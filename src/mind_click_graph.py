"""Complete-candidate exposure priors plus a bounded, train-only co-click graph."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import sparse
import torch

from full_impression_training import MANIFEST_SHA, PROOF_SHA, load_contract
from mind_cpu_ranker import (
    Source, candidate_rows, digest, encode_behaviors, limits, validate_groups, write_json,
)
from full_impression_contract import read_news
from mind_utils import auc_score, mrr_score, ndcg_score


TOP_K = 256
GRAPH_WEIGHT = 4.0
GRAPH_SHRINKAGE = 10.0
PRIOR_STRENGTH = 20.0
GATE_AUC = 0.69
RSS_LIMIT = 3758096384


def history_matrix(histories, width):
    if (histories.ndim != 2 or histories.shape[1] != 30
            or not np.issubdtype(histories.dtype, np.integer)
            or np.any(histories < 0) or np.any(histories >= width)):
        raise ValueError("History requires thirty ordered, in-range integer news identities")
    indices = np.array(histories, dtype=np.int32).ravel()
    values = (indices != 0).astype(np.float32)
    pointers = np.arange(len(histories) + 1, dtype=np.int64) * 30
    matrix = sparse.csr_matrix((values, indices, pointers), shape=(len(histories), width))
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


class ClickGraph:
    def __init__(self, graph, logits, coverage):
        if (not sparse.isspmatrix_csr(graph) or graph.shape[0] != graph.shape[1]
                or logits.shape != (graph.shape[0],)
                or not np.isfinite(graph.data).all() or not np.isfinite(logits).all()
                or np.any(graph.data < 0) or graph[0].nnz or graph[:, 0].nnz
                or np.any(graph.diagonal() != 0)):
            raise ValueError("Finite positive graph with zero padding/self edges required")
        self.graph, self.logits, self.coverage = graph, logits, coverage

    @classmethod
    def fit(cls, source, groups, width, deadline, rss_limit, top_k=TOP_K):
        validate_groups(source, groups)
        if width < 2 or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("Positive graph width and neighbor bound required")
        groups = np.sort(groups)
        expected = source.counts(groups)
        positive_ids = np.empty(expected["positives"], dtype=np.int32)
        positive_offsets = np.empty(len(groups) + 1, dtype=np.int64)
        positive_offsets[0] = 0
        exposures = np.zeros(width, dtype=np.float64)
        clicks = np.zeros(width, dtype=np.float64)
        consumed, positive_position = 0, 0
        for start in range(0, len(groups), 8192):
            limits(deadline, rss_limit)
            selected = groups[start:start + 8192]
            rows, local, sizes = candidate_rows(source.offsets, selected)
            news, labels = source.candidates[rows], source.labels[rows]
            if (np.any(news < 1) or np.any(news >= width)
                    or not np.all((labels == 0) | (labels == 1))):
                raise ValueError("Complete fitting candidates need known IDs and binary labels")
            exposures += np.bincount(news, minlength=width)
            clicks += np.bincount(news, weights=labels, minlength=width)
            positive = labels == 1
            counts = np.bincount(local[positive], minlength=len(selected))
            if np.any(counts == 0) or np.any(counts == sizes):
                raise ValueError("Every complete fitting impression needs both click classes")
            positive_ids[positive_position:positive_position + int(counts.sum())] = news[positive]
            positive_offsets[start + 1:start + len(selected) + 1] = positive_position + np.cumsum(counts)
            positive_position += int(counts.sum())
            consumed += len(rows)
        if (consumed != expected["exposures"] or positive_position != expected["positives"]
                or int(exposures.sum()) != consumed or int(clicks.sum()) != positive_position):
            raise ValueError("Incomplete all-impression/all-candidate fitting pass")
        positive_matrix = sparse.csr_matrix(
            (np.ones(positive_position, dtype=np.float64), positive_ids, positive_offsets),
            shape=(len(groups), width))
        positive_matrix.sum_duplicates()
        histories = history_matrix(source.histories[groups], width)
        transpose = histories.T.tocsr()
        del histories
        row_clicks = np.diff(positive_offsets).astype(np.float64)
        history_clicks = np.asarray(transpose @ row_clicks).ravel()
        maximum = width * min(top_k, width - 1)
        indices = np.empty(maximum, dtype=np.int32)
        values = np.empty(maximum, dtype=np.float32)
        pointers = np.zeros(width + 1, dtype=np.int64)
        position = 0
        for start in range(0, width, 256):
            limits(deadline, rss_limit)
            end = min(start + 256, width)
            counts = (transpose[start:end] @ positive_matrix).tocsr()
            counts.sum_duplicates()
            for local, row in enumerate(range(start, end)):
                lo, hi = counts.indptr[local:local + 2]
                columns = counts.indices[lo:hi]
                observed = counts.data[lo:hi].astype(np.float64)
                expected_pairs = history_clicks[row] * clicks[columns] / positive_position
                weights = np.log((observed + GRAPH_SHRINKAGE) / (expected_pairs + GRAPH_SHRINKAGE))
                eligible = (weights > 0) & (columns != row) & (columns != 0) & (row != 0)
                columns, weights = columns[eligible], weights[eligible]
                order = np.lexsort((columns, -weights))[:top_k]
                columns, weights = columns[order], weights[order]
                order = np.argsort(columns)
                number = len(columns)
                indices[position:position + number] = columns[order]
                values[position:position + number] = weights[order]
                position += number
                pointers[row + 1] = position
            if start % 8192 == 0:
                print(json.dumps({"graph_rows": end, "graph_edges": position,
                                  "fitting_coverage": expected}), flush=True)
        graph = sparse.csr_matrix((values[:position], indices[:position], pointers), shape=(width, width))
        graph.prune()
        ratio = positive_position / consumed
        probability = (clicks + PRIOR_STRENGTH * ratio) / (exposures + PRIOR_STRENGTH)
        logits = np.log(probability / (1 - probability))
        seen = np.zeros(len(source.offsets) - 1, dtype=bool)
        seen[groups] = True
        coverage = dict(expected, consumed_source_positions_sha256=digest_bytes(np.packbits(seen).tobytes()),
                        complete_aggregation_passes=1, candidate_sampling="none",
                        graph_edges=graph.nnz, top_k=top_k)
        limits(deadline, rss_limit)
        return cls(graph, logits, coverage)

    def scores(self, histories, candidates, local_groups):
        if (candidates.ndim != 1 or local_groups.shape != candidates.shape
                or not np.issubdtype(candidates.dtype, np.integer)
                or not np.issubdtype(local_groups.dtype, np.integer)
                or np.any(candidates < 1) or np.any(candidates >= len(self.logits))
                or np.any(local_groups < 0) or np.any(local_groups >= len(histories))):
            raise ValueError("Scoring requires exact known candidate/group identities")
        history = history_matrix(histories, len(self.logits))
        lengths = np.asarray(history.sum(axis=1)).ravel()
        personalized = (history @ self.graph).tocsr()
        personalized.sort_indices()
        selected = np.asarray(personalized[local_groups, candidates]).ravel()
        return self.logits[candidates] + GRAPH_WEIGHT * selected / np.maximum(lengths[local_groups], 1)

    def save(self, directory):
        directory.mkdir()
        sparse.save_npz(directory / "graph.npz", self.graph)
        np.save(directory / "logits.npy", self.logits, allow_pickle=False)
        write_json(directory / "fit_coverage.json", self.coverage)
        write_json(directory / "model_config.json", {
            "hypothesis": "complete_candidate_prior_plus_shrunk_positive_lift_history_click_graph_v1",
            "top_k": self.coverage["top_k"], "graph_weight": GRAPH_WEIGHT,
            "graph_shrinkage": GRAPH_SHRINKAGE, "prior_strength": PRIOR_STRENGTH,
            "history_limit": 30, "self_edges": False, "neural_optimizer_epochs": 0,
        })

    @classmethod
    def load(cls, directory):
        config = json.loads((directory / "model_config.json").read_bytes())
        if (config["graph_weight"] != GRAPH_WEIGHT or config["graph_shrinkage"] != GRAPH_SHRINKAGE
                or config["prior_strength"] != PRIOR_STRENGTH or config["history_limit"] != 30
                or config["self_edges"] or config["neural_optimizer_epochs"]):
            raise ValueError("Saved graph recipe differs")
        return cls(sparse.load_npz(directory / "graph.npz"),
                   np.load(directory / "logits.npy", allow_pickle=False),
                   json.loads((directory / "fit_coverage.json").read_bytes()))

    def extend(self, width):
        if width < len(self.logits):
            raise ValueError("Inference identity inventory cannot shrink")
        self.graph.resize((width, width))
        self.logits = np.concatenate((self.logits, np.full(width - len(self.logits), self.logits[0])))


def digest_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def evaluate(model, source, groups, path, deadline, rss_limit):
    validate_groups(source, groups)
    values, exposures, positives = [], 0, 0
    with path.open("x", encoding="utf-8") as handle:
        for start in range(0, len(groups), 256):
            limits(deadline, rss_limit)
            selected = groups[start:start + 256]
            rows, local, sizes = candidate_rows(source.offsets, selected)
            scores = model.scores(source.histories[selected], source.candidates[rows], local)
            if not np.isfinite(scores).all():
                raise FloatingPointError("Graph scores must be finite")
            labels = source.labels[rows]
            offset = 0
            for index, size in zip(selected.tolist(), sizes.tolist(), strict=True):
                target, score = labels[offset:offset + size], scores[offset:offset + size]
                if not np.all((target == 0) | (target == 1)) or not 0 < int(target.sum()) < size:
                    raise ValueError("Every evaluated impression requires both binary click classes")
                values.append((auc_score(target, score), mrr_score(target, score),
                               ndcg_score(target, score, 5), ndcg_score(target, score, 10)))
                identity = source.impression_ids[index] if hasattr(source, "impression_ids") else index
                json.dump([identity, score.tolist()], handle, allow_nan=False)
                handle.write("\n")
                offset += size
            exposures += len(rows)
            positives += int(labels.sum())
    expected = source.counts(groups)
    if expected != {"impressions": len(values), "exposures": exposures, "positives": positives}:
        raise ValueError("Incomplete graph evaluation population")
    metrics = {name: math.fsum(row[column] for row in values) / len(values)
               for column, name in enumerate(("auc", "mrr", "ndcg5", "ndcg10"))}
    return dict(expected, metrics=metrics, scores_sha256=digest(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--dev-behaviors", type=Path, required=True)
    parser.add_argument("--dev-news", type=Path, required=True)
    parser.add_argument("--train-news", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Fresh output and positive bounded allocation required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started, deadline = time.monotonic(), time.monotonic() + args.seconds
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if manifest["implementation_sha256"] != digest(Path(__file__).with_name("full_impression_contract.py")):
        raise ValueError("Pinned source producer differs")
    source = object.__new__(Source)
    source.manifest = manifest
    for field, name in (("candidates", "candidate_news.npy"), ("labels", "candidate_labels.npy"),
                        ("offsets", "candidate_offsets.npy"), ("histories", "histories.npy"),
                        ("days", "training_days.npy")):
        setattr(source, field, arrays[name])
    source.fingerprints = {}
    for path in (args.contract / "manifest.json", args.contract / "verification.json",
                 *(args.contract / name for name in manifest["sha256"])):
        stat = path.stat()
        source.fingerprints[path] = (stat.st_size, stat.st_mtime_ns)
    names = json.loads((args.contract / "news.ids.json").read_bytes())
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "registered_at": datetime.now(timezone.utc).isoformat(), "source_sha256": digest(Path(__file__)),
        "manifest_sha256": MANIFEST_SHA, "source_proof_sha256": PROOF_SHA,
        "gate_auc": GATE_AUC, "top_k": TOP_K, "graph_weight": GRAPH_WEIGHT,
        "graph_shrinkage": GRAPH_SHRINKAGE, "prior_strength": PRIOR_STRENGTH,
        "seconds": args.seconds, "rss_limit_bytes": RSS_LIMIT,
        "neural_epochs": 0, "final_fit_requires_every_training_impression_and_candidate": True,
        "dev_or_test_contents_read": False, "new_gpu_allocations": 0,
    })
    earlier = np.flatnonzero(source.days < source.days.max())
    heldout = np.flatnonzero(source.days == source.days.max())
    if (source.counts(earlier) != {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}
            or source.counts(heldout) != {"impressions": 431517, "exposures": 17400106, "positives": 671044}):
        raise ValueError("Pinned earlier/heldout training populations differ")
    model = ClickGraph.fit(source, earlier, len(names) + 1, deadline, RSS_LIMIT)
    model.save(args.output / "internal_selection")
    internal = evaluate(model, source, heldout, args.output / "internal_scores.jsonl", deadline, RSS_LIMIT)
    source.unchanged()
    if internal["metrics"]["auc"] < GATE_AUC:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_fit": False, "official_dev_contents_read": False,
            "model_fits": 1, "competition_test_goal_achieved": False,
        })
        return
    del model
    model = ClickGraph.fit(source, np.arange(len(source.days)), len(names) + 1, deadline, RSS_LIMIT)
    if any(model.coverage[key] != value for key, value in {
        "impressions": 2232748, "exposures": 83507374, "positives": 3383656}.items()):
        raise ValueError("Final fit did not use the complete training population")
    model.save(args.output / "all_training_fit")
    source.unchanged()
    if (digest(args.dev_behaviors) != "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"
            or digest(args.dev_news) != "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"
            or digest(args.train_news) != manifest["input_sha256"]["news.tsv"]):
        raise ValueError("Bound official-dev/training-news inputs differ")
    train_news, dev_news = read_news(args.train_news), read_news(args.dev_news)
    if sorted(train_news) != names:
        raise ValueError("Raw training news identities differ")
    if any(train_news[name] != item for name, item in dev_news.items() if name in train_news):
        raise ValueError("Shared train/dev news metadata changed")
    names.extend(sorted(set(dev_news) - set(names)))
    lookup = {name: index + 1 for index, name in enumerate(names)}
    model.extend(len(names) + 1)
    dev = encode_behaviors(args.dev_behaviors, SimpleNamespace(lookup=lookup), 376471, 14085557)
    if int(dev.labels.sum()) != 574845:
        raise ValueError("Official-dev click population differs")
    official = evaluate(model, dev, np.arange(376471), args.output / "official_dev_scores.jsonl",
                        deadline, RSS_LIMIT)
    source.unchanged()
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_fit_and_official_dev_scored",
        "internal_validation": internal, "official_dev": official, "model_fits": 2,
        "final_all_training_fit": True, "official_dev_contents_read": True,
        "competition_test_contents_read": False, "competition_test_goal_achieved": False,
        "elapsed_seconds": time.monotonic() - started,
    })


if __name__ == "__main__":
    main()
