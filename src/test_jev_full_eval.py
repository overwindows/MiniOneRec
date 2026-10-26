import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from behavior_jev_readout import build_counts, save_counts
from evaluate_full_jev_readout import evaluate
from probe_jev_readout import metrics
from scale_jev_readout import sha256_file


class FullTrainingEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        log = root / "train.tsv"
        log.write_text("".join(f"T{i}\tU\tt\t\tN1-1 N0-0\n" for i in range(30)))
        counts, totals = build_counts(log, set())
        roots = [root / "full", root / "retained"]
        results = []
        for index, trained in enumerate(roots):
            trained.mkdir()
            save_counts(trained / "population.npz", counts, totals)
            np.savez(
                trained / "correction.head.npz", mean=np.zeros(4), scale=np.ones(4),
                weight=np.asarray([0., 0., .1 if index == 0 else -.1, 0.]), bias=np.asarray(0.),
            )
            results.append({
                "selected": "correction", "behavior_selected": "correction",
                "selected_head_sha256": sha256_file(trained / "correction.head.npz"),
                "behavior_head_sha256": sha256_file(trained / "correction.head.npz"),
                "population_sha256": sha256_file(trained / "population.npz"),
                "reference_head_sha256": "b" * 64, "text_head_sha256": "a" * 64,
                "training_only": True, "full_training": True, "training_pairs": 469264,
                "training_impressions": 156965, "train_holdout_impressions": 0,
                "artificial_pair_cap": None, "quality_result": None, "minimum_dev_start": 31000,
            })
            (trained / "result.json").write_text(json.dumps(results[-1]))
        dev = root / "data" / "dev"
        dev.mkdir(parents=True)
        (dev / "news.tsv").write_text("fixture-news")
        (dev / "behaviors.tsv").write_text(
            "".join(f"{i}\tU\tt\t\tN1-1 N0-0\n" for i in range(31000))
            + "registered\tU\tt\t\tN1-1 N0-0\n"
        )
        raw = root / "base.scores.jsonl"
        raw.write_text(json.dumps({
            "impression_id": "registered", "candidate_ids": ["N1", "N0"], "labels": [1, 0],
            "frozen_margins": [1., -1.], "selected_margins": [0., 0.],
            "reference_margins": [0., 0.],
        }) + "\n")
        baseline = {
            "source_revision": "source", "head_sha256": "a" * 64, "reference_head_sha256": "b" * 64,
            "dev_start_zero_based": 31000, "dev_impressions": 1, "dev_candidates": 2,
            "selected_metrics": metrics([1, 0], [0., 0.], ["a", "a"]),
            "reference_metrics": metrics([1, 0], [0., 0.], ["a", "a"]),
            "frozen": metrics([1, 0], [1., -1.], ["a", "a"]),
            "data_sha256": {"dev/" + name: sha256_file(dev / name) for name in ("news.tsv", "behaviors.tsv")},
        }
        base = root / "base.json"
        base.write_text(json.dumps(baseline))
        return [
            roots[0], sha256_file(roots[0] / "result.json"),
            roots[1], sha256_file(roots[1] / "result.json"),
            base, sha256_file(base), raw, sha256_file(raw), root / "data", root / "out",
            "source", 31000, 1,
        ]

    def test_both_frozen_corrections_are_applied_to_their_own_text_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            with patch("builtins.print"):
                result = evaluate(*args)
            self.assertEqual(result["selected_metrics"]["auc"], 1.)
            self.assertEqual(result["reference_metrics"]["auc"], 0.)
            self.assertEqual(result["frozen"]["auc"], 1.)
            record = json.loads((args[9] / "dev.scores.jsonl").read_text())
            np.testing.assert_allclose(record["selected_margins"], -np.array(record["reference_margins"]))
            with self.assertRaises(FileExistsError):
                evaluate(*args)

    def test_every_pinned_input_and_both_populations_are_checked(self):
        for target in ("full/result.json", "retained/result.json", "base.json",
                       "base.scores.jsonl", "full/population.npz", "retained/population.npz"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                with (root / target).open("ab") as handle:
                    handle.write(b"x")
                with self.assertRaises(ValueError):
                    evaluate(*args)

    def test_incomplete_training_and_consumed_dev_windows_are_rejected(self):
        for key, value in (
            ("training_pairs", 262143), ("training_impressions", 156709),
            ("train_holdout_impressions", 256), ("artificial_pair_cap", 131071),
            ("quality_result", {"auc": .9}), ("full_training", False), ("minimum_dev_start", 36000),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                path = args[0] / "result.json"
                result = json.loads(path.read_bytes())
                result[key] = value
                path.write_text(json.dumps(result))
                args[1] = sha256_file(path)
                with self.assertRaises(ValueError):
                    evaluate(*args)
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            args[-2] = 26000
            with self.assertRaises(ValueError):
                evaluate(*args)

    def test_reference_identity_metrics_and_raw_scores_cannot_be_substituted(self):
        for key, value in (
            ("reference_head_sha256", "c" * 64), ("head_sha256", "c" * 64),
            ("reference_metrics", dict(auc=.9, mrr=1., ndcg5=1., ndcg10=1.)),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                result = json.loads(args[4].read_bytes())
                result[key] = value
                args[4].write_text(json.dumps(result))
                args[5] = sha256_file(args[4])
                with self.assertRaises(ValueError):
                    evaluate(*args)
        for key, value in (("reference_margins", [float("nan"), 0.]),
                           ("candidate_ids", ["N0", "N1"]), ("labels", [0, 1])):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                args = self.fixture(Path(directory))
                record = json.loads(args[6].read_text())
                record[key] = value
                args[6].write_text(json.dumps(record) + "\n")
                args[7] = sha256_file(args[6])
                with self.assertRaises(ValueError):
                    evaluate(*args)


if __name__ == "__main__":
    unittest.main()
