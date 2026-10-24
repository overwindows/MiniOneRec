import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

import nonlinear_jev_readout as nonlinear
from probe_jev_readout import metrics, predict
from refine_jev_readout import REFERENCE_NAME
import test_jev_refinement as refinement_tests


class NonlinearTests(unittest.TestCase):
    def head(self):
        return {
            "mean": np.zeros(2), "scale": np.ones(2), "weight": np.zeros(2),
            "bias": np.asarray(0.), "hidden_weight": np.asarray([[1.], [0.]], dtype=np.float32),
            "hidden_bias": np.zeros(1, dtype=np.float32), "output_weight": np.zeros(1, dtype=np.float32),
            "output_bias": np.asarray(0., dtype=np.float32),
        }

    def test_zero_correction_preserves_reference_and_round_trip(self):
        x = np.asarray([[-2., 1.], [2., -1.], [-1., -1.], [1., 1.]])
        base = np.asarray([1., -1., 1., -1.])
        head = self.head()
        head["weight"] = np.asarray([.3, -.2])
        np.testing.assert_array_equal(nonlinear.predict_nonlinear(head, x, base), predict(head, x, base))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.npz"
            np.savez(path, **head)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored, _ = nonlinear.load_nonlinear_head(path, digest)
            np.testing.assert_array_equal(
                nonlinear.predict_nonlinear(restored, x, base), predict(head, x, base),
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                nonlinear.load_nonlinear_head(path, "0" * 64)

    def test_portable_prediction_matches_torch_network(self):
        head = self.head()
        head["output_weight"][:] = 2.
        head["output_bias"] = np.asarray(-.25, dtype=np.float32)
        x = np.asarray([[-2., 1.], [2., -1.], [-1., -1.], [1., 1.]], dtype=np.float32)
        actual = nonlinear.predict_nonlinear(head, x, np.zeros(4))
        hidden = torch.relu(torch.from_numpy(x) @ torch.from_numpy(head["hidden_weight"]))
        expected = (hidden @ torch.from_numpy(head["output_weight"]) - .25).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_actual_training_learns_nonlinear_xor_without_changing_reference(self):
        torch.set_num_threads(2)
        x = np.tile(np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (8, 1))
        labels = (x[:, 0] * x[:, 1] > 0).astype(int)
        reference = {key: value.copy() for key, value in self.head().items() if key in nonlinear.REFERENCE_FIELDS}
        original = {key: value.copy() for key, value in reference.items()}
        with patch("builtins.print"):
            checkpoints = list(nonlinear.fit_checkpoints(
                x, np.zeros(len(x)), labels, reference, checkpoints=(100, 200),
            ))
        self.assertEqual([value[0] for value in checkpoints], [100, 200])
        score = nonlinear.predict_nonlinear(checkpoints[-1][1], x, np.zeros(len(x)))
        self.assertEqual(metrics(labels, score, ["group"] * len(x))["auc"], 1.)
        for key in reference:
            np.testing.assert_array_equal(reference[key], original[key])
        saved = checkpoints[0][1]["hidden_weight"].copy()
        checkpoints[-1][1]["hidden_weight"][:] = 0
        np.testing.assert_array_equal(checkpoints[0][1]["hidden_weight"], saved)

    def test_determinism_deadline_and_invalid_training_inputs(self):
        x = np.asarray([[-1., 0.], [1., 0.]])
        reference = {key: value.copy() for key, value in self.head().items() if key in nonlinear.REFERENCE_FIELDS}
        with patch("builtins.print"):
            first = list(nonlinear.fit_checkpoints(x, np.zeros(2), np.asarray([0, 1]), reference, (2,)))
            second = list(nonlinear.fit_checkpoints(x, np.zeros(2), np.asarray([0, 1]), reference, (2,)))
        for key in first[0][1]:
            np.testing.assert_array_equal(first[0][1][key], second[0][1][key])
        with self.assertRaises(TimeoutError):
            list(nonlinear.fit_checkpoints(
                x, np.zeros(2), np.asarray([0, 1]), reference, (2,), time.monotonic() - 1,
            ))
        for labels, epochs in (([1, 1], (2,)), ([0, 1], (2, 1)), ([0, 1], ())):
            with self.assertRaises(ValueError):
                list(nonlinear.fit_checkpoints(x, np.zeros(2), np.asarray(labels), reference, epochs))

    def test_invalid_heads_are_rejected(self):
        for key, value in (
            ("scale", np.zeros(2)), ("weight", np.zeros(3)), ("hidden_bias", np.zeros(2)),
            ("output_bias", np.asarray(np.nan)), ("output_weight", np.zeros(1, dtype=int)),
        ):
            with self.subTest(key=key):
                head = self.head()
                head[key] = value
                with self.assertRaises(ValueError):
                    nonlinear.validate_head(head)
        head = self.head()
        head["unknown"] = np.asarray(1.)
        with self.assertRaises(ValueError):
            nonlinear.validate_head(head)

    def test_strict_gate_requires_material_auc_and_all_secondary_nonregression(self):
        reference = {"name": REFERENCE_NAME, "validation": dict(auc=.7, mrr=.3, ndcg5=.4, ndcg10=.5)}
        for auc, mrr, opens in ((.7, .3, False), (.704, .31, False), (.71, .29, False), (.71, .3, True)):
            candidate = {"name": "mlp", "validation": dict(auc=auc, mrr=mrr, ndcg5=.4, ndcg10=.5)}
            selected, _ = nonlinear.select_candidate([reference, candidate])
            self.assertEqual(selected["name"] != REFERENCE_NAME, opens)
        selected, best = nonlinear.select_candidate([
            reference,
            {"name": "second", "validation": dict(auc=.71, mrr=.31, ndcg5=.41, ndcg10=.51)},
            {"name": "best", "validation": dict(auc=.72, mrr=.29, ndcg5=.41, ndcg10=.51)},
        ])
        self.assertEqual(best["name"], "best")
        self.assertEqual(selected["name"], REFERENCE_NAME)

    def test_invalid_loader_parity_is_rejected_before_training(self):
        for difference in (float("nan"), float("inf"), -.1, .3):
            with self.subTest(difference=difference), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = refinement_tests.RefinementTests().fixture(root)
                parity = json.loads((root / "parity.json").read_bytes())
                parity["loader_max_margin_difference"] = difference
                (root / "parity.json").write_text(json.dumps(parity))
                with self.assertRaisesRegex(ValueError, "parity"):
                    nonlinear.train(*args, root / "parity.json", root / "out", "revision")

    def test_train_only_artifacts_gate_and_reference_byte_preservation(self):
        for succeeds in (False, True):
            with self.subTest(succeeds=succeeds), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = refinement_tests.RefinementTests().fixture(root)
                head = self.head()
                head["output_weight"][:] = 4. if succeeds else 0.
                with (
                    patch("nonlinear_jev_readout.fit_checkpoints", return_value=iter([(4, head, .5)])),
                    patch("builtins.print"),
                ):
                    result = nonlinear.train(*args, root / "parity.json", root / "out", "revision")
                self.assertTrue(result["training_only"])
                self.assertEqual(result["nonlinear_head_opens_dev_gate"], succeeds)
                self.assertEqual(
                    (root / "out" / (REFERENCE_NAME + ".head.npz")).read_bytes(),
                    (root / "head.npz").read_bytes(),
                )
                self.assertEqual(
                    result["selected_head_sha256"],
                    hashlib.sha256((root / "out" / (result["selected"] + ".head.npz")).read_bytes()).hexdigest(),
                )
                with self.assertRaises(FileExistsError):
                    nonlinear.train(*args, root / "parity.json", root / "out", "revision")


if __name__ == "__main__":
    unittest.main()
