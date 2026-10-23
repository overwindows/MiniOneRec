import tempfile
import hashlib
from pathlib import Path
import unittest

import numpy as np
import torch

from probe_jev_readout import behavior_rows, extract, fit_head, metrics, pair_indices, predict
from evaluate_jev_readout import load_head


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


if __name__ == "__main__":
    unittest.main()
