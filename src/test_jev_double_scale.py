import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from probe_jev_readout import metrics
import scale_jev_readout as scale


class DoubleScaleTests(unittest.TestCase):
    def test_doubling_uses_full_reference_prefix_and_separate_original_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, holdout, train = root / "reference", root / "holdout", root / "data" / "train"
            for path in (reference, holdout, train):
                path.mkdir(parents=True)
            (train / "news.tsv").write_text("fixture")
            (train / "behaviors.tsv").write_text("fixture\n" * 257)
            rows = [
                {"impression_id": str(i // 2 if i < 131068 else 65534 if i < 131071 else 65535),
                 "candidate_id": str(i), "label": i % 2}
                for i in range(131073)
            ]
            x = np.asarray([[2 * r["label"] - 1, 1] for r in rows], dtype=np.float32)
            base = np.zeros(len(rows), dtype=np.float32)
            np.savez(reference / "train.features.npz", features=x[:131071], margins=base[:131071])
            (reference / "train.rows.json").write_text(json.dumps(rows[:131071]))
            records = [
                ("h" + str(i), [
                    {"impression_id": "h" + str(i), "candidate_id": str(y), "label": y}
                    for y in (0, 1)
                ]) for i in range(256)
            ]
            holdout_rows = [r for _, group in records for r in group]
            hx = np.tile(np.asarray([[-1, 1], [1, 1]], dtype=np.float32), (256, 1))
            hb = np.zeros(512, dtype=np.float32)
            np.savez(holdout / "train_holdout.features.npz", features=hx, margins=hb)
            (holdout / "train_holdout.rows.json").write_text(json.dumps(holdout_rows))
            head = dict(mean=np.zeros(2), scale=np.ones(2), weight=np.zeros(2), bias=np.asarray(0.))
            head_path = reference / "binary_l2_0.01_scaled.head.npz"
            np.savez(head_path, **head)
            head_sha = scale.sha256_file(head_path)
            validation = metrics([r["label"] for r in holdout_rows], hb,
                                 [r["impression_id"] for r in holdout_rows])
            original = {
                "selected": "binary_l2_0.01_scaled", "training_pairs": 131071,
                "train_holdout_impressions": 256, "training_only": True,
                "scaled_head_opens_dev_gate": True,
                "train_features_sha256": scale.sha256_file(reference / "train.features.npz"),
                "feature_cache_sha256": {
                    name: scale.sha256_file(holdout / name)
                    for name in ("train_holdout.features.npz", "train_holdout.rows.json")
                },
                "data_sha256": {
                    "train/" + name: scale.sha256_file(train / name)
                    for name in ("news.tsv", "behaviors.tsv")
                },
            }
            (reference / "result.json").write_text(json.dumps(original))
            (reference / "train-selection.json").write_text(json.dumps({
                "selected": original["selected"],
                "candidates": [{"name": original["selected"], "validation": validation}],
            }))
            (reference / "loader-parity.json").write_text("{}")
            args = [
                "scale", "--double-training", "--holdout-cache-root", str(holdout),
                "--model-path", "fixture", "--data-root", str(root / "data"),
                "--reference-head", str(head_path), "--reference-head-sha256", head_sha,
                "--output", str(root / "out"), "--source-revision", "fixture",
            ]
            with (
                patch("sys.argv", args),
                patch.object(scale, "DOUBLE_REFERENCE_HEAD_SHA256", head_sha),
                patch.object(scale, "DOUBLE_REFERENCE_RESULT_SHA256", scale.sha256_file(reference / "result.json")),
                patch.object(scale, "JevScorer", return_value=SimpleNamespace(
                    model=SimpleNamespace(parameters=lambda: []), tokenizer=None,
                )),
                patch.object(scale, "load_news", return_value={}),
                patch.object(scale, "behavior_rows", return_value=records),
                patch.object(scale, "JEVNoulPointwiseDataset", return_value=SimpleNamespace(samples=[])),
                patch.object(scale, "sample_training_rows", return_value=rows) as sampler,
                patch.object(scale, "check_cached_model") as replay,
                patch.object(scale, "extract", return_value=(x[131071:], base[131071:])) as extract,
            ):
                scale.main()
            self.assertEqual(sampler.call_args.args[2], 262144)
            self.assertEqual(replay.call_count, 2)
            self.assertEqual(len(extract.call_args.args[1]), 2)
            result = json.loads((root / "out" / "result.json").read_bytes())
            self.assertTrue(result["training_only"])
            self.assertEqual(result["cached_training_pairs"], 131071)
            self.assertEqual(result["new_training_pairs"], 2)
            self.assertEqual(result["selected"], "binary_l2_0.01_doubled")
            self.assertTrue(result["scaled_head_opens_dev_gate"])
            protocol = json.loads((root / "out" / "protocol.json").read_bytes())
            self.assertEqual(protocol["training_pair_limit"], 262144)
            self.assertEqual(protocol["minimum_dev_start"], 21000)
            self.assertEqual(json.loads((root / "out" / "train-selection.json").read_bytes())["minimum_dev_start"], 21000)
            self.assertEqual(protocol["dev_evaluation"], "none; selection uses training holdout only")

    def test_doubling_requires_original_holdout_and_exact_reference(self):
        for extras in ([], ["--holdout-cache-root", "fixture"]):
            with self.subTest(extras=extras), patch("sys.argv", [
                "scale", "--double-training", "--model-path", "unused", "--data-root", "unused",
                "--reference-head", "unused", "--reference-head-sha256",
                "wrong" if extras else scale.DOUBLE_REFERENCE_HEAD_SHA256,
                "--output", "unused", "--source-revision", "fixture", *extras,
            ]), self.assertRaises(SystemExit):
                scale.main()


if __name__ == "__main__":
    unittest.main()
