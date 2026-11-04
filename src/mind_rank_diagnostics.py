"""Read-only, complete-population error anatomy for existing ranking scores."""

import json
import math
from pathlib import Path

import numpy as np

from mind_cpu_ranker import candidate_rows, limits, validate_groups


def candidate_support(source, fitting, width, deadline, rss_limit):
    validate_groups(source, fitting)
    if not isinstance(width, int) or width < 2:
        raise ValueError("An exact one-based news identity space is required")
    supported = np.zeros(width, dtype=bool)
    exposures = positives = 0
    for start in range(0, len(fitting), 8192):
        limits(deadline, rss_limit)
        rows, _, _ = candidate_rows(source.offsets, fitting[start:start + 8192])
        news, labels = source.candidates[rows], source.labels[rows]
        if (np.any(news < 1) or np.any(news >= width)
                or not np.all((labels == 0) | (labels == 1))):
            raise ValueError("Invalid source candidate identity or binary label")
        supported[news] = True
        exposures += len(rows)
        positives += int(labels.sum())
    return supported, {"impressions": len(fitting), "exposures": exposures, "positives": positives}


def score_records(paths):
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if (not isinstance(record, list) or len(record) != 2 or type(record[0]) is not int
                        or not isinstance(record[1], list)
                        or any(type(value) not in (int, float) for value in record[1])):
                    raise ValueError("Integer source positions and finite numeric score lists required")
                yield record


def pair_wins(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all()
            or not np.all((labels == 0) | (labels == 1))):
        raise ValueError("Finite exact complete-group binary scores required")
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not len(positive) or not len(negative):
        raise ValueError("Undefined group AUC may not be dropped or defaulted")
    margin = positive[:, None] - negative
    return (margin > 0) + .5 * (margin == 0), margin


def analyze(source, fitting, heldout, width, streams, deadline, rss_limit):
    validate_groups(source, heldout)
    if np.intersect1d(fitting, heldout).size or not streams:
        raise ValueError("Disjoint fitting/heldout populations and named score streams required")
    supported, coverage = candidate_support(source, fitting, width, deadline, rss_limit)
    iterators = {name: iter(records) for name, records in streams.items()}
    cohorts, support_pairs = {}, {}
    all_auc = {name: [] for name in streams}
    bounds = []
    exposures = positives = cold_exposures = cold_positives = empty_histories = 0
    for position in heldout.tolist():
        limits(deadline, rss_limit)
        lo, hi = int(source.offsets[position]), int(source.offsets[position + 1])
        news, labels = source.candidates[lo:hi], source.labels[lo:hi]
        if np.any(news < 1) or np.any(news >= width):
            raise ValueError("Heldout candidate identity outside exact source vocabulary")
        warm = supported[news]
        history_length = int(np.count_nonzero(source.histories[position]))
        if history_length == 0:
            history_cohort = "history_empty"
        elif history_length < 10:
            history_cohort = "history_1_to_9"
        elif history_length < 30:
            history_cohort = "history_10_to_29"
        else:
            history_cohort = "history_30"
        exposure_cohort = ("all_candidates_exposure_supported" if warm.all()
                           else "no_candidates_exposure_supported" if not warm.any()
                           else "mixed_candidate_exposure_support")
        groups = ("all", history_cohort, exposure_cohort)
        for key in groups:
            if key not in cohorts:
                cohorts[key] = {"impressions": 0, "exposures": 0, "positives": 0,
                                "auc_sum": {name: [] for name in streams}}
            bucket = cohorts[key]
            bucket["impressions"] += 1
            bucket["exposures"] += len(news)
            bucket["positives"] += int(labels.sum())
        wins_by_model, margins = {}, []
        for name, records in iterators.items():
            record = next(records, None)
            if record is None or record[0] != position or len(record[1]) != len(news):
                raise ValueError("Missing/reordered source position or candidate count: " + name)
            wins, margin = pair_wins(record[1], labels)
            wins_by_model[name] = wins
            margins.append(margin)
            auc = float(wins.mean())
            all_auc[name].append(auc)
            for key in groups:
                cohorts[key]["auc_sum"][name].append(auc)
        for positive_known in (False, True):
            p = warm[labels == 1] == positive_known
            for negative_known in (False, True):
                n = warm[labels == 0] == negative_known
                key = f"positive_{'seen' if positive_known else 'unseen'}__negative_{'seen' if negative_known else 'unseen'}"
                if key not in support_pairs:
                    support_pairs[key] = {
                        "impressions_with_pair": 0, "pairs": 0,
                        "macro_auc_contribution_sum": {name: [] for name in streams}}
                bucket = support_pairs[key]
                count = int(p.sum() * n.sum())
                bucket["pairs"] += count
                bucket["impressions_with_pair"] += int(count > 0)
                for name, wins in wins_by_model.items():
                    bucket["macro_auc_contribution_sum"][name].append(
                        float(wins[np.ix_(p, n)].sum()) / wins.size)
        any_win = np.logical_or.reduce([margin > 0 for margin in margins])
        any_tie = np.logical_or.reduce([margin == 0 for margin in margins])
        bounds.append(float((any_win + .5 * (~any_win & any_tie)).mean()))
        exposures += len(news)
        positives += int(labels.sum())
        cold_exposures += int((~warm).sum())
        cold_positives += int(((labels == 1) & ~warm).sum())
        empty_histories += int(history_length == 0)
    for name, records in iterators.items():
        if next(records, None) is not None:
            raise ValueError("Extra scored impression: " + name)
    aucs = {name: math.fsum(values) / len(heldout) for name, values in all_auc.items()}
    for bucket in cohorts.values():
        bucket["auc"] = {name: math.fsum(values) / bucket["impressions"]
                         for name, values in bucket.pop("auc_sum").items()}
    for bucket in support_pairs.values():
        bucket["macro_auc_contribution"] = {
            name: math.fsum(values) / len(heldout)
            for name, values in bucket.pop("macro_auc_contribution_sum").items()}
    for name, auc in aucs.items():
        if abs(math.fsum(bucket["macro_auc_contribution"][name]
                         for bucket in support_pairs.values()) - auc) > 1e-12:
            raise ValueError("Exposure-support pair partition failed exact macro-AUC additivity")
    first = next(iter(all_auc))
    comparison = {}
    for name in list(all_auc)[1:]:
        difference = np.asarray(all_auc[name]) - np.asarray(all_auc[first])
        comparison[name + "_minus_" + first] = {
            "mean_auc_difference": math.fsum(difference) / len(heldout),
            "impressions_improved": int((difference > 1e-14).sum()),
            "impressions_worse": int((difference < -1e-14).sum()),
            "impressions_tied": int((np.abs(difference) <= 1e-14).sum())}
    return {
        "scope": "read_only_complete_training_day_error_anatomy_not_model_selection_or_dev_test",
        "fitting_source_coverage": coverage,
        "evaluation_coverage": {"impressions": len(heldout), "exposures": exposures, "positives": positives},
        "model_auc": aucs, "cohorts": cohorts, "exposure_support_pair_contributions": support_pairs,
        "candidate_exposures_without_earlier_candidate_exposure": cold_exposures,
        "positives_without_earlier_candidate_exposure": cold_positives,
        "impressions_with_empty_history": empty_histories, "paired_comparison": comparison,
        "nonnegative_convex_score_mixture_pairwise_oracle_upper_bound": math.fsum(bounds) / len(bounds),
        "oracle_bound_is_not_a_deployable_model_auc": True,
        "support_definition": "article appeared as candidate in fitting rows; not history/metadata coverage",
        "no_causal_attribution_between_different_recipes": True,
        "model_fits": 0, "gate_reopened": False, "dev_or_test_contents_read": False,
        "competition_test_goal_achieved": False}
