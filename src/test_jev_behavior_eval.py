import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from behavior_jev_readout import build_counts, save_counts
from evaluate_jev_behavior import evaluate
from probe_jev_readout import metrics
from scale_jev_readout import sha256_file


class BehaviorEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        trained = root / "trained"
        trained.mkdir()
        training_log = root / "train.tsv"
        training_log.write_text("".join(
            f"train-{index}\tU\tt\t\tN1-1 N0-0\n" for index in range(30)
        ))
        counts, totals = build_counts(training_log, set())
        save_counts(trained / "population.npz", counts, totals)
        name = "behavior_pairwise_l2_0.01"
        np.savez(
            trained / (name + ".head.npz"), mean=np.zeros(4), scale=np.ones(4),
            weight=np.asarray([0., 0., .1, 0.]), bias=np.asarray(0.),
        )
        result = {
            "training_only": True, "behavior_head_opens_dev_gate": True,
            "selected": name, "minimum_dev_start": 26000,
            "reference_head_sha256": "b" * 64,
            "selected_head_sha256": sha256_file(trained / (name + ".head.npz")),
            "population_sha256": sha256_file(trained / "population.npz"),
        }
        (trained / "result.json").write_text(json.dumps(result))
        (trained / "train-selection.json").write_text(json.dumps({
            "selected": name, "minimum_dev_start": 26000,
            "candidates": [
                {"name": "scaled_reference", "validation": dict(auc=.7, mrr=.3, ndcg5=.4, ndcg10=.5)},
                {"name": name, "validation": dict(auc=.72, mrr=.31, ndcg5=.41, ndcg10=.51)},
            ],
        }))
        dev = root / "data" / "dev"
        dev.mkdir(parents=True)
        (dev / "news.tsv").write_text("fixture-news")
        (dev / "behaviors.tsv").write_text(
            "".join(f"{i}\tU\tt\t\tN1-1 N0-0\n" for i in range(26000))
            + "registered\tU\tt\t\tN1-1 N0-0\n"
        )
        raw = root / "base.scores.jsonl"
        raw.write_text(json.dumps({
            "impression_id": "registered", "candidate_ids": ["N1", "N0"], "labels": [1, 0],
            "frozen_margins": [1., -1.], "selected_margins": [0., 0.],
        }) + "\n")
        baseline = {
            "source_revision": "base-source", "head_sha256": "b" * 64,
            "dev_start_zero_based": 26000, "dev_impressions": 1, "dev_candidates": 2,
            "selected_metrics": metrics([1, 0], [0., 0.], ["a", "a"]),
            "frozen": metrics([1, 0], [1., -1.], ["a", "a"]),
            "data_sha256": {"dev/" + name: sha256_file(dev / name) for name in ("news.tsv", "behaviors.tsv")},
        }
        base_result = root / "base.json"
        base_result.write_text(json.dumps(baseline))
        return [
            trained, sha256_file(trained / "result.json"),
            base_result, sha256_file(base_result), raw, sha256_file(raw),
            root / "data", root / "out", "evaluation-source", 26000, 1,
        ]

    def test_frozen_correction_exact_window_and_artifact_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with patch("builtins.print"):
                result = evaluate(*args)
            self.assertEqual(result["selected_metrics"]["auc"], 1.)
            self.assertEqual(result["reference_metrics"]["auc"], .5)
            self.assertEqual(result["dev_candidates"], 2)
            self.assertEqual(result["base_result_sha256"], args[3])
            self.assertEqual(result["base_scores_sha256"], args[5])
            self.assertEqual(result["reference_head_sha256"], "b" * 64)
            record = json.loads((root / "out" / "dev.scores.jsonl").read_text())
            self.assertEqual(record["candidate_ids"], ["N1", "N0"])
            self.assertGreater(record["selected_margins"][0], record["selected_margins"][1])
            with self.assertRaises(FileExistsError):
                evaluate(*args)

    def test_unpinned_inputs_and_population_tampering_are_rejected(self):
        for target in ("trained/result.json", "base.json", "base.scores.jsonl", "trained/population.npz"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                with (root / target).open("ab") as handle:
                    handle.write(b"x")
                with self.assertRaises(ValueError):
                    evaluate(*args)

    def test_zero_correction_preserves_retained_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.fixture(Path(directory))
            result_path = args[0] / "result.json"
            trained = json.loads(result_path.read_bytes())
            head = args[0] / (trained["selected"] + ".head.npz")
            np.savez(head, mean=np.zeros(4), scale=np.ones(4), weight=np.zeros(4), bias=np.asarray(0.))
            trained["selected_head_sha256"] = sha256_file(head)
            result_path.write_text(json.dumps(trained))
            args[1] = sha256_file(result_path)
            with patch("builtins.print"):
                result = evaluate(*args)
            self.assertEqual(result["selected_metrics"], result["reference_metrics"])

    def test_base_identity_window_and_metrics_must_reproduce(self):
        for key, value in (
            ("head_sha256", "x" * 64), ("dev_start_zero_based", 25000),
            ("dev_candidates", 3), ("selected_metrics", dict(auc=.9, mrr=1., ndcg5=1., ndcg10=1.)),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                baseline = json.loads(args[2].read_bytes())
                baseline[key] = value
                args[2].write_text(json.dumps(baseline))
                args[3] = sha256_file(args[2])
                with self.assertRaises(ValueError):
                    evaluate(*args)

    def test_raw_order_labels_coverage_and_nonfinite_scores_are_rejected(self):
        for key, value in (
            ("candidate_ids", ["N0", "N1"]), ("labels", [0, 1]),
            ("selected_margins", [0.]), ("selected_margins", [float("nan"), 0.]),
        ):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                record = json.loads(args[4].read_text())
                record[key] = value
                args[4].write_text(json.dumps(record) + "\n")
                args[5] = sha256_file(args[4])
                with self.assertRaises(ValueError):
                    evaluate(*args)

    def test_failed_gate_or_consumed_window_cannot_be_evaluated(self):
        for failed in ("flag", "candidate", "window"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                if failed == "flag":
                    path = args[0] / "result.json"
                    result = json.loads(path.read_bytes())
                    result["behavior_head_opens_dev_gate"] = False
                    path.write_text(json.dumps(result))
                    args[1] = sha256_file(path)
                elif failed == "candidate":
                    path = args[0] / "train-selection.json"
                    selection = json.loads(path.read_bytes())
                    selection["candidates"][1]["validation"]["mrr"] = .29
                    path.write_text(json.dumps(selection))
                else:
                    args[-2] = 25000
                with self.assertRaises(ValueError):
                    evaluate(*args)


if __name__ == "__main__":
    unittest.main()
