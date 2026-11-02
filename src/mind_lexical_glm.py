"""Full sparse lexical click likelihood with complete-exposure empirical priors."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy import optimize, sparse
from scipy.special import expit
import torch

from full_impression_contract import read_news
from full_impression_training import MANIFEST_SHA, PROOF_SHA, load_contract
from mind_click_graph import evaluate
from mind_cpu_ranker import (
    Corpus, Source, candidate_rows, digest, encode_behaviors, limits, validate_groups, write_json,
)


L2 = 1e-5
PRIOR_STRENGTH = 20.0
RSS_LIMIT = 3758096384
ENCODER_HASHES = {
    "corpus.json": "1a678cbb88976656001fe868c85edb58786c3db1a299d34520cc860b04262467",
    "lexical_encoder.npz": "823f5fc617506b210ca1e6f072eb8e7fe7e75357109f36febf92213ca3782de5",
}


def lexical_matrix(corpus, news, names=None):
    names = corpus.names if names is None else names
    if any(corpus._metadata_hash(news[name]) != corpus.metadata_sha256[name] for name in names):
        raise ValueError("Frozen unlabeled article metadata changed")
    parts = [sparse.csr_matrix((1, len(corpus.vectorizer.vocabulary_)), dtype=np.float64)]
    for start in range(0, len(names), 2048):
        texts = [news[name]["title"] + " " + news[name]["abstract"]
                 for name in names[start:start + 2048]]
        parts.append(corpus.vectorizer.transform(texts).astype(np.float64))
    text = sparse.vstack(parts, format="csr")
    rows = np.arange(len(names) + 1)
    identities = np.array([0] + [corpus.lookup[name] for name in names])
    category = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, corpus.category[identities])),
        shape=(len(rows), len(corpus.categories) + 1))
    subcategory = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, corpus.subcategory[identities])),
        shape=(len(rows), len(corpus.subcategories) + 1))
    return sparse.hstack((text, category, subcategory), format="csr")


def binomial_objective(parameters, matrix, exposures, clicks, l2=L2):
    if (parameters.shape != (matrix.shape[1] + 1,)
            or exposures.shape != clicks.shape or exposures.shape != (matrix.shape[0],)
            or np.any(exposures < 0) or np.any(clicks < 0) or np.any(clicks > exposures)
            or not np.isfinite(parameters).all() or not np.isfinite(matrix.data).all()
            or not np.isfinite(exposures).all() or not np.isfinite(clicks).all()
            or exposures.sum() <= 0 or not np.isfinite(l2) or l2 < 0):
        raise ValueError("Finite complete binomial counts and compatible lexical parameters required")
    logits = matrix @ parameters[1:] + parameters[0]
    total = exposures.sum()
    objective = np.sum(exposures * np.logaddexp(0, logits) - clicks * logits) / total
    objective += 0.5 * l2 * np.dot(parameters[1:], parameters[1:])
    residual = (exposures * expit(logits) - clicks) / total
    gradient = np.concatenate(([residual.sum()], np.asarray(matrix.T @ residual).ravel()))
    gradient[1:] += l2 * parameters[1:]
    return float(objective), gradient


def posterior_logits(content_logits, exposures, clicks):
    positive = np.full_like(clicks, -np.inf, dtype=np.float64)
    negative = np.full_like(clicks, -np.inf, dtype=np.float64)
    np.log(clicks, out=positive, where=clicks > 0)
    np.log(exposures - clicks, out=negative, where=exposures > clicks)
    return (np.logaddexp(positive, np.log(PRIOR_STRENGTH) - np.logaddexp(0, -content_logits))
            - np.logaddexp(negative, np.log(PRIOR_STRENGTH) - np.logaddexp(0, content_logits)))


class LexicalClickModel:
    def __init__(self, corpus, parameters, logits, coverage):
        dimensions = (len(corpus.vectorizer.vocabulary_) + len(corpus.categories)
                      + len(corpus.subcategories) + 3)
        if (parameters.shape != (dimensions,) or logits.shape != (len(corpus.names) + 1,)
                or not np.isfinite(parameters).all() or not np.isfinite(logits).all()):
            raise ValueError("Finite lexical parameters and exact article identities required")
        self.corpus, self.parameters, self.logits, self.coverage = corpus, parameters, logits, coverage

    @classmethod
    def fit(cls, source, groups, corpus, matrix, deadline, rss_limit):
        validate_groups(source, groups)
        width = len(corpus.names) + 1
        exposures, clicks = np.zeros(width), np.zeros(width)
        expected = source.counts(groups)
        for start in range(0, len(groups), 8192):
            limits(deadline, rss_limit)
            rows, _, _ = candidate_rows(source.offsets, groups[start:start + 8192])
            news, labels = source.candidates[rows], source.labels[rows]
            if (np.any(news < 1) or np.any(news >= width)
                    or not np.all((labels == 0) | (labels == 1))):
                raise ValueError("Every fitting candidate must have a known identity and binary click")
            exposures += np.bincount(news, minlength=width)
            clicks += np.bincount(news, weights=labels, minlength=width)
        if (int(exposures.sum()) != expected["exposures"] or int(clicks.sum()) != expected["positives"]
                or matrix.shape[0] != width):
            raise ValueError("Incomplete fitting aggregation")
        ratio = clicks.sum() / exposures.sum()
        if not 0 < ratio < 1:
            raise ValueError("Fitting requires both click classes")
        initial = np.zeros(matrix.shape[1] + 1)
        initial[0] = np.log(ratio / (1 - ratio))

        def objective(parameters):
            limits(deadline, rss_limit)
            return binomial_objective(parameters, matrix, exposures, clicks)

        fitted = optimize.minimize(objective, initial, jac=True, method="L-BFGS-B",
                                   options={"maxiter": 400, "gtol": 1e-6, "ftol": 1e-10})
        if not fitted.success:
            raise RuntimeError("Lexical likelihood optimizer failed: " + str(fitted.message))
        logits = posterior_logits(matrix @ fitted.x[1:] + fitted.x[0], exposures, clicks)
        mask = np.zeros(len(source.offsets) - 1, dtype=bool)
        mask[groups] = True
        from mind_click_graph import digest_bytes
        coverage = dict(expected, consumed_source_positions_sha256=digest_bytes(np.packbits(mask).tobytes()),
                        candidate_sampling="none", complete_aggregation_passes=1,
                        optimizer_iterations=int(fitted.nit), optimizer_converged=True,
                        final_objective=float(fitted.fun), lexical_features=matrix.shape[1])
        return cls(corpus, fitted.x, logits, coverage)

    def scores(self, histories, candidates, local_groups):
        if (histories.ndim != 2 or histories.shape[1] != 30 or candidates.ndim != 1
                or candidates.shape != local_groups.shape
                or not all(np.issubdtype(value.dtype, np.integer) for value in
                           (histories, candidates, local_groups))
                or np.any(histories < 0) or np.any(histories >= len(self.logits))
                or np.any(candidates < 1) or np.any(candidates >= len(self.logits))
                or np.any(local_groups < 0) or np.any(local_groups >= len(histories))):
            raise ValueError("Exact known history/candidate/group identities required")
        mask = histories != 0
        counts = np.maximum(mask.sum(axis=1), 1)
        mean = self.corpus.lexical[histories].sum(axis=1) / counts[:, None]
        mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        similarity = (self.corpus.lexical[candidates] * mean[local_groups]).sum(axis=1)
        history = histories[local_groups]
        valid = history != 0
        category = self.corpus.category[candidates, None]
        subcategory = self.corpus.subcategory[candidates, None]
        category_match = ((self.corpus.category[history] == category) & valid & (category != 0)).sum(axis=1)
        subcategory_match = ((self.corpus.subcategory[history] == subcategory)
                             & valid & (subcategory != 0)).sum(axis=1)
        repeated = ((history == candidates[:, None]) & valid).any(axis=1)
        return (self.logits[candidates] + similarity
                + 0.5 * category_match / counts[local_groups]
                + 0.5 * subcategory_match / counts[local_groups] - repeated)

    def save(self, directory):
        directory.mkdir()
        np.save(directory / "parameters.npy", self.parameters, allow_pickle=False)
        np.save(directory / "logits.npy", self.logits, allow_pickle=False)
        write_json(directory / "fit_coverage.json", self.coverage)
        write_json(directory / "model_config.json", {
            "hypothesis": "full_sparse_lexical_binomial_empirical_prior_v1",
            "l2": L2, "prior_strength": PRIOR_STRENGTH, "history_limit": 30,
            "cosine_weight": 1.0, "category_weight": 0.5, "subcategory_weight": 0.5,
            "previously_read_penalty": 1.0, "objective": "exact_aggregated_candidate_binomial_likelihood",
        })

    @classmethod
    def load(cls, directory, corpus):
        config = json.loads((directory / "model_config.json").read_bytes())
        if any(config[key] != value for key, value in {
            "l2": L2, "prior_strength": PRIOR_STRENGTH, "history_limit": 30,
            "cosine_weight": 1.0, "category_weight": 0.5,
            "subcategory_weight": 0.5, "previously_read_penalty": 1.0}.items()):
            raise ValueError("Saved lexical recipe differs")
        return cls(corpus, np.load(directory / "parameters.npy", allow_pickle=False),
                   np.load(directory / "logits.npy", allow_pickle=False),
                   json.loads((directory / "fit_coverage.json").read_bytes()))

    def extend(self, news):
        old = len(self.corpus.names)
        self.corpus.extend(news)
        if len(self.corpus.names) > old:
            matrix = lexical_matrix(self.corpus, news, self.corpus.names[old:])[1:]
            self.logits = np.concatenate((self.logits, matrix @ self.parameters[1:] + self.parameters[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "encoder", "train-news", "dev-news", "dev-behaviors", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Fresh output and bounded allocation required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    deadline = time.monotonic() + args.seconds
    for name, expected in ENCODER_HASHES.items():
        if digest(args.encoder / name) != expected:
            raise ValueError("Frozen unlabeled encoder bytes differ")
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if manifest["implementation_sha256"] != digest(Path(__file__).with_name("full_impression_contract.py")):
        raise ValueError("Pinned complete-source producer changed")
    if digest(args.train_news) != manifest["input_sha256"]["news.tsv"]:
        raise ValueError("Pinned full training-news metadata changed")
    corpus = Corpus.load(args.encoder)
    train_news = read_news(args.train_news)
    if sorted(train_news) != corpus.names:
        raise ValueError("Full source article identities differ")
    matrix = lexical_matrix(corpus, train_news)
    source = object.__new__(Source)
    for field, filename in (("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
                            ("labels", "candidate_labels.npy"), ("histories", "histories.npy"),
                            ("days", "training_days.npy")):
        setattr(source, field, arrays[filename])
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(Path(__file__)), "manifest_sha256": MANIFEST_SHA,
        "encoder_sha256": ENCODER_HASHES, "encoder_fits_use_no_click_labels": True,
        "old_ranker_weights_or_priors_read": False, "gate_auc": 0.69, "seconds": args.seconds,
        "objective": "exact_aggregated_candidate_binomial_likelihood",
        "l2": L2, "prior_strength": PRIOR_STRENGTH, "optimizer_max_iterations": 400,
        "lexical_features": matrix.shape[1], "source_news": len(corpus.names),
        "official_dev_or_test_contents_read": False, "new_gpu_allocations": 0,
    })
    earlier = np.flatnonzero(source.days < source.days.max())
    heldout = np.flatnonzero(source.days == source.days.max())
    if (source.counts(earlier) != {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}
            or source.counts(heldout) != {"impressions": 431517, "exposures": 17400106, "positives": 671044}):
        raise ValueError("Exact earlier/heldout training populations differ")
    model = LexicalClickModel.fit(source, earlier, corpus, matrix, deadline, RSS_LIMIT)
    model.save(args.output / "internal_selection")
    internal = evaluate(model, source, heldout, args.output / "internal_scores.jsonl", deadline, RSS_LIMIT)
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if internal["metrics"]["auc"] < 0.69:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_fit": False, "official_dev_contents_read": False,
            "competition_test_goal_achieved": False, "model_fits": 1,
        })
        return
    model = LexicalClickModel.fit(source, np.arange(len(source.days)), corpus, matrix, deadline, RSS_LIMIT)
    if any(model.coverage[key] != value for key, value in {
        "impressions": 2232748, "exposures": 83507374, "positives": 3383656}.items()):
        raise ValueError("Final lexical fit did not consume all training candidates")
    model.save(args.output / "all_training_fit")
    if (digest(args.dev_behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"
            or digest(args.dev_news) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"):
        raise ValueError("Pinned official-dev bytes differ")
    dev_news = read_news(args.dev_news)
    model.extend(train_news | dev_news)
    dev = encode_behaviors(args.dev_behaviors, corpus, 376471, 14085557)
    if int(dev.labels.sum()) != 574845:
        raise ValueError("Official-dev positive population differs")
    official = evaluate(model, dev, np.arange(376471), args.output / "official_dev_scores.jsonl",
                        deadline, RSS_LIMIT)
    if (digest(args.dev_behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"
            or digest(args.dev_news) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"):
        raise ValueError("Official-dev inputs changed while scoring")
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_fit_and_official_dev_scored",
        "internal_validation": internal, "official_dev": official, "model_fits": 2,
        "final_all_training_fit": True, "official_dev_contents_read": True,
        "competition_test_contents_read": False, "competition_test_goal_achieved": False,
    })


if __name__ == "__main__":
    main()
