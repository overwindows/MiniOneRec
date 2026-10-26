import tempfile
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from probe_jev_readout import (
    behavior_rows, extract, fit_head, metrics, pair_indices, predict, sample_training_rows,
)
from evaluate_jev_readout import load_head
import evaluate_jev_readout as readout_eval
from scale_jev_readout import check_cached_model, load_cached_features, write_holdout_scores


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.features = np.asarray([[-2, 1], [2, 1], [-1, 1], [1, 1]], dtype=float)
        self.base = np.zeros(4)
        self.labels = np.asarray([0, 1, 0, 1])
        self.groups = ["a", "a", "b", "b"]

    def test_zero_residual_preserves_frozen_margins_and_ranking(self):
        scores = np.asarray([1.5, -2, 3, -0.5])
        np.testing.assert_array_equal(predict(None, self.features, scores), scores)
        head = {
            "mean": self.features.mean(0), "scale": np.ones(2),
            "weight": np.zeros(2), "bias": np.asarray(0.0),
        }
        np.testing.assert_array_equal(predict(head, self.features, scores), scores)

    def test_both_objectives_learn_and_round_trip_without_backbone_updates(self):
        original = self.features.copy()
        for objective in ("binary", "pairwise"):
            with self.subTest(objective=objective):
                head, loss = fit_head(
                    self.features, self.base, self.labels, self.groups, 0.1, objective,
                )
                self.assertTrue(np.isfinite(loss))
                scores = predict(head, self.features, self.base)
                self.assertEqual(metrics(self.labels, scores, self.groups)["auc"], 1.0)
                self.assertGreater(scores[1], scores[0])
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "head.npz"
                    np.savez(path, **head)
                    with np.load(path, allow_pickle=False) as stored:
                        restored = dict(stored)
                    np.testing.assert_array_equal(
                        predict(restored, self.features, self.base), scores,
                    )
        np.testing.assert_array_equal(self.features, original)

    def test_pairs_are_impression_local_and_equally_weighted(self):
        labels = [0, 1, 0, 0, 1]
        groups = ["a", "a", "b", "b", "b"]
        left, right, weight = pair_indices(labels, groups)
        self.assertAlmostEqual(float(weight.sum()), 1.0)
        for p, n in zip(left, right):
            self.assertEqual(groups[p], groups[n])
            self.assertEqual(labels[p], 1)
            self.assertEqual(labels[n], 0)
        self.assertAlmostEqual(float(weight[:1].sum()), 0.5)
        self.assertAlmostEqual(float(weight[1:].sum()), 0.5)

    def test_metrics_convert_full_inputs_once_and_preserve_ties(self):
        labels = [1, 0, 0, 0, 1, 0]
        scores = [1., 0., 0., 1., 0., 0.]
        groups = ["a"] * 3 + ["b"] * 3
        with patch("probe_jev_readout.np.asarray", wraps=np.asarray) as convert:
            actual = metrics(labels, scores, groups)
        self.assertEqual(sum(call.args[0] is labels for call in convert.call_args_list), 1)
        self.assertEqual(sum(call.args[0] is scores for call in convert.call_args_list), 1)
        for name, value in {"auc": .625, "mrr": 2 / 3, "ndcg5": .75, "ndcg10": .75}.items():
            self.assertAlmostEqual(actual[name], value)
        self.assertEqual(actual, metrics(np.asarray(labels), np.asarray(scores), groups))

    def test_invalid_objectives_nonfinite_and_degenerate_groups_fail(self):
        with self.assertRaises(ValueError):
            fit_head(self.features, self.base, self.labels, self.groups, 0, "binary")
        bad = self.features.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_head(bad, self.base, self.labels, self.groups, 1, "binary")
        with self.assertRaises(ValueError):
            pair_indices([1, 1], ["a", "a"])
        with self.assertRaises(ValueError):
            metrics([1, 1], [0, 0], ["a", "a"])

    def test_extraction_captures_the_actual_output_head_input_and_margin(self):
        class Scorer:
            cand_tokens = [0, 1]

            def __init__(self):
                self.model = torch.nn.Module()
                self.model.lm_head = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.model.lm_head.weight.copy_(torch.tensor([[1.0, 0], [-1.0, 0]]))

            @torch.inference_mode()
            def score_yes_batch(self, states, chunk_size):
                hidden = torch.tensor(states, dtype=torch.float32).unsqueeze(1)
                return self.model.lm_head(hidden)[:, 0].softmax(-1)[:, 0].tolist()

        rows = [{"state": row.tolist()} for row in self.features]
        features, margins = extract(Scorer(), rows, None, "unit", chunk_size=2)
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(margins, 2 * self.features[:, 0])

    def test_behavior_window_is_exact_and_incomplete_windows_fail(self):
        news = {
            key: {"text": key, "category": "cat"} for key in ("H", "P", "N")
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            path.write_text(
                "1\tU\ttime\tH\tP-1 N-0\n2\tU\ttime\tH\tN-0 P-1\n",
                encoding="utf-8",
            )
            records = behavior_rows(path, news, start=1, count=1)
            self.assertEqual(records[0][0], "2")
            self.assertEqual([row["candidate_id"] for row in records[0][1]], ["N", "P"])
            with self.assertRaises(ValueError):
                behavior_rows(path, news, start=1, count=2)

    def test_confirmation_head_is_checksum_pinned_and_shape_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.npz"
            head, _ = fit_head(
                self.features, self.base, self.labels, self.groups, 0.1, "binary",
            )
            np.savez(path, **head)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored, _ = load_head(path, digest)
            np.testing.assert_array_equal(restored["weight"], head["weight"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_head(path, "0" * 64)
            head["scale"][0] = 0
            np.savez(path, **head)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "dimensions or numeric"):
                load_head(path, digest)

    def test_data_scaling_preserves_grouped_prefix_and_excludes_holdout(self):
        article = {"text": "title", "category": "cat"}
        samples = [
            {"impression_id": group, "candidate_id": f"N{index}", "label": label,
             "history": [article], "candidate": article}
            for index, (group, label) in enumerate([
                ("a", 0), ("heldout", 1), ("b", 0), ("a", 1),
                ("b", 1), ("b", 0), ("heldout", 0), ("single", 1),
            ])
        ]
        small = sample_training_rows(samples, {"heldout"}, 4)
        large = sample_training_rows(samples, {"heldout"}, 5)
        self.assertEqual(len(small), 2)
        self.assertEqual(len(large), 5)
        self.assertEqual(large[:len(small)], small)
        self.assertEqual([r["impression_id"] for r in large], ["a", "a", "b", "b", "b"])
        with self.assertRaises(ValueError):
            sample_training_rows(samples, {"heldout"}, 1)
        with self.assertRaises(ValueError):
            sample_training_rows(samples, {"heldout"}, 0)

    def test_feature_cache_requires_exact_row_identity_order_and_finite_shape(self):
        rows = [
            {"impression_id": "a", "candidate_id": f"N{i}", "label": i}
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "train.rows.json"
            metadata.write_text(json.dumps(rows), encoding="utf-8")
            np.savez(root / "train.features.npz", features=self.features[:2], margins=self.base[:2])
            actual, margins = load_cached_features(root, "train", rows, 2)
            np.testing.assert_array_equal(actual, self.features[:2])
            np.testing.assert_array_equal(margins, self.base[:2])
            with self.assertRaisesRegex(ValueError, "deterministic training sample"):
                load_cached_features(root, "train", rows[::-1], 2)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                load_cached_features(root, "train", rows, 3)
            np.savez(root / "train.features.npz", features=np.full((2, 2), np.nan), margins=self.base[:2])
            with self.assertRaisesRegex(ValueError, "numeric"):
                load_cached_features(root, "train", rows, 2)

    def test_feature_cache_is_replayed_against_the_actual_frozen_model(self):
        with patch("scale_jev_readout.extract", return_value=(self.features, self.base)):
            check_cached_model(None, [{}] * 4, self.features, self.base, "unit")
        with patch("scale_jev_readout.extract", return_value=(self.features + 1, self.base)):
            with self.assertRaisesRegex(ValueError, "frozen-model replay"):
                check_cached_model(None, [{}] * 4, self.features, self.base, "unit")

    def test_training_holdout_artifact_preserves_all_three_conditions(self):
        records = [("a", [
            {"candidate_id": "N0", "label": 0}, {"candidate_id": "N1", "label": 1},
        ])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_holdout_scores(root, records, self.base[:2], self.base[:2] + 1, self.base[:2] + 2)
            row = json.loads((root / "train-holdout.scores.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["candidate_ids"], ["N0", "N1"])
            self.assertEqual(row["labels"], [0, 1])
            self.assertEqual(row["frozen_margins"], [0, 0])
            self.assertEqual(row["reference_margins"], [1, 1])
            self.assertEqual(row["scaled_margins"], [2, 2])

    def test_three_arm_evaluator_reuses_one_feature_pass_and_preserves_two_arm_default(self):
        for with_reference in (False, True):
            with self.subTest(with_reference=with_reference), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                head_dir, dev = root / "heads", root / "data" / "dev"
                head_dir.mkdir()
                dev.mkdir(parents=True)
                for name in ("news.tsv", "behaviors.tsv"):
                    (dev / name).write_text("fixture", encoding="utf-8")
                head = dict(mean=np.zeros(2), scale=np.ones(2), weight=np.asarray([1., 0.]), bias=np.asarray(0.))
                selected_path = head_dir / "scaled.head.npz"
                np.savez(selected_path, **head)
                head["weight"] = -head["weight"]
                reference_path = head_dir / "reference.head.npz"
                np.savez(reference_path, **head)
                (head_dir / "train-selection.json").write_text(json.dumps({"selected": "scaled"}))
                (head_dir / "loader-parity.json").write_text("{}")
                (head_dir / "result.json").write_text('{"training_only":true}')
                records = [
                    (group, [
                        {"impression_id": group, "candidate_id": f"N{index + offset}",
                         "label": index, "state": "fixture"}
                        for index in (0, 1)
                    ]) for group, offset in (("a", 0), ("b", 2))
                ]
                args = [
                    "evaluate_jev_readout", "--model-path", "unused", "--data-root", str(root / "data"),
                    "--head-path", str(selected_path), "--head-sha256",
                    hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                    "--output", str(root / "out"), "--source-revision", "fixture",
                    "--dev-start", "11000", "--dev-impressions", "2",
                ]
                if with_reference:
                    args += [
                        "--reference-head", str(reference_path), "--reference-head-sha256",
                        hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                    ]
                scorer = SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))
                with (
                    patch("sys.argv", args),
                    patch.object(readout_eval, "JevScorer", return_value=scorer),
                    patch.object(readout_eval, "load_news", return_value={}),
                    patch.object(readout_eval, "behavior_rows", return_value=records),
                    patch.object(readout_eval, "extract", return_value=(self.features, self.base)) as forward,
                ):
                    readout_eval.main()
                self.assertEqual(forward.call_count, 1)
                result = json.loads((root / "out" / "result.json").read_text())
                self.assertEqual(result["selected_metrics"]["auc"], 1)
                self.assertEqual(result["frozen"]["auc"], 0.5)
                raw = json.loads((root / "out" / "dev.scores.jsonl").read_text().splitlines()[0])
                self.assertEqual("reference_margins" in raw, with_reference)
                if with_reference:
                    self.assertEqual(result["reference_metrics"]["auc"], 0)
                    self.assertEqual(result["delta_vs_reference"]["auc"], 1)
                    self.assertEqual(result["primary_comparison"], "selected_minus_reference")
                else:
                    self.assertNotIn("reference_metrics", result)
                (head_dir / "train-selection.json").write_text(json.dumps({
                    "selected": "scaled", "minimum_dev_start": 16000,
                }))
                args[args.index("--output") + 1] = str(root / "overlap")
                with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "overlaps"):
                    readout_eval.main()

    def test_three_arm_evaluator_rejects_previously_used_windows(self):
        with patch("sys.argv", [
            "evaluate_jev_readout", "--model-path", "unused", "--data-root", "unused",
            "--head-path", "unused", "--head-sha256", "unused", "--output", "unused",
            "--source-revision", "fixture", "--reference-head", "unused",
            "--reference-head-sha256", "unused", "--dev-start", "6000",
        ]), self.assertRaises(SystemExit):
            readout_eval.main()

    def test_pipeline_executor_mounts_both_checksum_pinned_heads(self):
        path = (
            Path(__file__).resolve().parents[1] / "pipeline" / "components"
            / "mind_jev_eval" / "pipeline_executor.py"
        )
        spec = importlib.util.spec_from_file_location("readout_pipeline_executor_test", path)
        executor = importlib.util.module_from_spec(spec)
        with patch("logging.FileHandler", return_value=logging.NullHandler()):
            spec.loader.exec_module(executor)
        args = [
            "pipeline_executor", "--config", "unused.json", "--mount-dir", "/mnt/fixture",
            "--readout-head", "shares/scaled.head.npz", "--readout-head-sha256", "scaled-digest",
            "--comparison-head", "shares/reference.head.npz",
            "--comparison-head-sha256", "reference-digest",
            "--comparison-dev-start", "16000",
            "--readout-holdout-cache", "shares/original-cache",
        ]
        with patch("sys.argv", args), patch.object(executor, "PipelineExecutor") as factory:
            factory.return_value.run.return_value = True
            with self.assertRaises(SystemExit) as result:
                executor.main()
        self.assertEqual(result.exception.code, 0)
        variables = factory.call_args.args[1]
        self.assertEqual(variables["readout_head"], "/mnt/fixture/shares/scaled.head.npz")
        self.assertEqual(variables["comparison_head"], "/mnt/fixture/shares/reference.head.npz")
        self.assertEqual(variables["comparison_head_sha256"], "reference-digest")
        self.assertEqual(variables["comparison_dev_start"], 16000)
        self.assertEqual(variables["readout_holdout_cache"], "/mnt/fixture/shares/original-cache")


if __name__ == "__main__":
    unittest.main()
