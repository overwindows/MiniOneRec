"""Memory-bounded joint rich-representation and popularity readout fitting."""

from collections import Counter
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from probe_jev_readout import pair_indices, predict


BATCH_SIZE = 2048
RECIPES = (
    {"name": "joint_binary_control", "balanced": False, "pair_fraction": 0., "l2": .01},
    {"name": "joint_balanced_l2_0.01", "balanced": True, "pair_fraction": 0., "l2": .01},
    {"name": "joint_balanced_l2_0.1", "balanced": True, "pair_fraction": 0., "l2": .1},
    {"name": "joint_mixed_l2_0.01", "balanced": True, "pair_fraction": .5, "l2": .01},
)


def validation_impressions(records, reference_rows, reference_holdout):
    used = {row["impression_id"] for row in reference_rows + reference_holdout}
    last_day = max(record["day"] for record in records)
    selected = {
        record["impression_id"] for record in records
        if record["day"] == last_day and record["impression_id"] not in used
    }
    if not selected or len(selected) == len(records):
        raise ValueError("Cohort-disjoint selection requires nonempty fitting and validation groups")
    return last_day, selected


def select_joint(candidates, controls):
    keys = ("auc", "mrr", "ndcg5", "ndcg10")
    if (not candidates or not controls
            or any(not np.isfinite(arm["metrics"][key])
                   or not 0 <= arm["metrics"][key] <= 1
                   for arm in candidates + controls for key in keys)):
        raise ValueError("Joint selection requires finite candidate and control ranking metrics")
    best = max(candidates, key=lambda arm: arm["metrics"]["auc"])
    opens = all(
        best["metrics"]["auc"] - control["metrics"]["auc"] >= .005
        and all(best["metrics"][key] >= control["metrics"][key] for key in keys[1:])
        for control in controls
    )
    return best, opens


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Streamed readout exceeded the registered CPU deadline")


class StreamedObjective:
    def __init__(self, features, extras, margins, labels, groups, fit_mask, recipe, deadline=None):
        labels, margins = np.asarray(labels), np.asarray(margins)
        fit_mask = np.asarray(fit_mask)
        if (features.ndim != 2 or extras.ndim != 2 or not features.shape[1]
                or extras.shape[0] != len(features) or labels.shape != (len(features),)
                or margins.shape != labels.shape or len(groups) != len(features)
                or fit_mask.shape != labels.shape or fit_mask.dtype != bool
                or not fit_mask.any() or not np.isin(labels, (0, 1)).all()
                or set(np.unique(labels[fit_mask])) != {0, 1}
                or not np.isfinite(margins).all()
                or not 0 <= recipe["pair_fraction"] <= 1 or recipe["l2"] <= 0):
            raise ValueError("Expected finite aligned data, mixed fitting labels and valid recipe")
        self.features, self.extras = features, extras
        self.labels, self.margins = labels.astype(np.float64), margins.astype(np.float64)
        self.indices = np.flatnonzero(fit_mask)
        self.recipe, self.deadline = recipe, deadline
        selected_groups = [groups[index] for index in self.indices]
        if any(not isinstance(group, str) or not group for group in selected_groups):
            raise ValueError("Impression groups must be nonempty strings")
        sizes = Counter(selected_groups)
        self.weights = (
            np.asarray([1. / (len(sizes) * sizes[group]) for group in selected_groups])
            if recipe["balanced"] else np.full(len(self.indices), 1. / len(self.indices))
        )
        self.dimension = features.shape[1] + extras.shape[1]
        count, self.mean, m2 = 0, np.zeros(self.dimension), np.zeros(self.dimension)
        for start in range(0, len(self.indices), BATCH_SIZE):
            check_deadline(deadline)
            block = self.raw(self.indices[start:start + BATCH_SIZE])
            n = len(block)
            block_mean = block.mean(axis=0)
            delta = block_mean - self.mean
            m2 += np.square(block - block_mean).sum(axis=0) + delta ** 2 * count * n / (count + n)
            self.mean += delta * n / (count + n)
            count += n
        self.scale = np.sqrt(m2 / count).clip(min=1e-6)
        self.pairs = None
        if recipe["pair_fraction"]:
            left, right, weights = pair_indices(
                self.labels[self.indices], selected_groups, allow_unpaired=True,
            )
            self.pairs = (
                self.indices[left.numpy()], self.indices[right.numpy()], weights.numpy(),
            )

    def raw(self, indices):
        block = np.empty((len(indices), self.dimension), dtype=np.float64)
        dimension = self.features.shape[1]
        block[:, :dimension] = self.features[indices]
        block[:, dimension:] = self.extras[indices]
        if not np.isfinite(block).all():
            raise ValueError("Readout features must be finite")
        return block

    def standardized(self, indices):
        block = self.raw(indices)
        block -= self.mean
        block /= self.scale
        return block

    def value_gradient(self, parameters):
        check_deadline(self.deadline)
        weight, bias = parameters[:-1], parameters[-1]
        loss = .5 * self.recipe["l2"] * weight @ weight
        gradient = np.r_[self.recipe["l2"] * weight, 0.]
        fraction = self.recipe["pair_fraction"]
        if fraction < 1:
            for start in range(0, len(self.indices), BATCH_SIZE):
                check_deadline(self.deadline)
                indices = self.indices[start:start + BATCH_SIZE]
                x = self.standardized(indices)
                scores = self.margins[indices] + x @ weight + bias
                weights = (1. - fraction) * self.weights[start:start + BATCH_SIZE]
                targets = self.labels[indices]
                loss += weights @ (np.logaddexp(0., scores) - targets * scores)
                derivative = weights * (expit(scores) - targets)
                gradient[:-1] += x.T @ derivative
                gradient[-1] += derivative.sum()
        if self.pairs is not None:
            left, right, weights = self.pairs
            for start in range(0, len(left), BATCH_SIZE):
                check_deadline(self.deadline)
                p, n = left[start:start + BATCH_SIZE], right[start:start + BATCH_SIZE]
                difference = self.standardized(p) - self.standardized(n)
                scores = self.margins[p] - self.margins[n] + difference @ weight
                selected_weights = fraction * weights[start:start + BATCH_SIZE]
                loss += selected_weights @ np.logaddexp(0., -scores)
                gradient[:-1] -= difference.T @ (selected_weights * expit(-scores))
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError("Streamed readout objective or gradient is nonfinite")
        return float(loss), gradient


def fit_streamed(features, extras, margins, labels, groups, fit_mask, recipe, deadline=None):
    objective = StreamedObjective(
        features, extras, margins, labels, groups, fit_mask, recipe, deadline,
    )
    result = minimize(
        objective.value_gradient, np.zeros(objective.dimension + 1), jac=True, method="L-BFGS-B",
        options={"maxiter": 100, "maxls": 30, "gtol": 1e-6, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError("Streamed readout failed to converge: " + str(result.message))
    head = {
        "mean": objective.mean, "scale": objective.scale,
        "weight": result.x[:-1], "bias": np.asarray(result.x[-1]),
    }
    return head, {
        "loss": float(result.fun), "iterations": int(result.nit),
        "evaluations": int(result.nfev), "converged": True,
        "fitting_rows": len(objective.indices),
    }


def predict_streamed(head, features, extras, margins):
    if (features.ndim != 2 or extras.ndim != 2 or len(features) != len(extras)
            or np.asarray(margins).shape != (len(features),)
            or features.shape[1] + extras.shape[1] != len(head["weight"])):
        raise ValueError("Expected aligned joint-head features and margins")
    scores = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), BATCH_SIZE):
        end = start + BATCH_SIZE
        block = np.column_stack((features[start:end], extras[start:end])).astype(np.float64)
        scores[start:end] = predict(head, block, margins[start:end])
    if not np.isfinite(scores).all():
        raise FloatingPointError("Streamed readout predictions are nonfinite")
    return scores
