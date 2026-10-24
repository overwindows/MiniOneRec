import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from probe_jev_readout import metrics, predict
from refine_jev_readout import load_inputs, refine, REFERENCE_NAME
from scale_jev_readout import sha256_file


class RefinementTests(unittest.TestCase):
    def fixture(self, root):
        x = np.asarray([[-2, 1], [2, 1], [-1, 1], [1, 1]], dtype=np.float32)
        rows = [
            {"impression_id": group, "candidate_id": str(i), "label": i % 2}
            for i, group in enumerate(("a", "a", "b", "b"))
        ]
        holdout = [dict(row, impression_id="holdout-" + row["impression_id"]) for row in rows]
        base = np.asarray([1, -1, 1, -1], dtype=np.float32)
        for name, metadata in (("train", rows), ("train_holdout", holdout)):
            np.savez(root / (name + ".features.npz"), features=x, margins=base)
            (root / (name + ".rows.json")).write_text(json.dumps(metadata), encoding="utf-8")
        (root / "expected.json").write_text(json.dumps(rows), encoding="utf-8")
        head = {"mean": np.zeros(2), "scale": np.ones(2), "weight": np.zeros(2), "bias": np.asarray(0.)}
        np.savez(root / "head.npz", **head)
        head_sha = sha256_file(root / "head.npz")
        values = metrics([r["label"] for r in holdout], base, [r["impression_id"] for r in holdout])
        provenance = {
            "training_only": True, "scaled_head_opens_dev_gate": True, "scaled_head_sha256": head_sha,
            "train_features_sha256": sha256_file(root / "train.features.npz"),
            "feature_cache_sha256": {
                name: sha256_file(root / name)
                for name in ("train_holdout.features.npz", "train_holdout.rows.json")
            },
            "training_pairs": 4, "training_impressions": 2,
            "holdout_candidates": 4, "train_holdout_impressions": 2,
            "scaled": values, "data_sha256": {"train/behaviors.tsv": "fixture"},
        }
        (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
        (root / "parity.json").write_text(json.dumps({
            "head_rows_identical": True, "loader_max_margin_difference": 0.,
            "training_max_margin_difference": 0.,
        }), encoding="utf-8")
        return [
            root, root / "provenance.json", sha256_file(root / "provenance.json"),
            root / "head.npz", head_sha, root / "expected.json",
        ]

    def test_exact_cached_search_and_train_only_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with patch("refine_jev_readout.CANDIDATES", (("binary", .1), ("pairwise", .1))):
                result = refine(*args, root / "parity.json", root / "out", "revision")
            self.assertTrue(result["refined_head_opens_dev_gate"])
            self.assertEqual(result["selected"], "binary_l2_0.1_refined")
            self.assertEqual(result["selected_metrics"]["auc"], 1.)
            self.assertTrue(result["training_only"])
            self.assertEqual(result["selected_head_sha256"], sha256_file(
                root / "out" / (result["selected"] + ".head.npz"),
            ))
            records = [json.loads(line) for line in (root / "out" / "train-holdout.scores.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(set(records[0]["candidate_margins"]), {
                REFERENCE_NAME, "binary_l2_0.1_refined", "pairwise_l2_0.1_refined",
            })
            with np.load(root / "out" / (result["selected"] + ".head.npz")) as stored:
                head = dict(stored)
            with np.load(root / "out" / "train-holdout-spotcheck.npz") as stored:
                np.testing.assert_allclose(
                    predict(head, stored["features"], stored["margins"]), stored["selected"],
                )
            with self.assertRaises(FileExistsError):
                refine(*args, root / "parity.json", root / "out", "revision")

    def test_reference_wins_ties_and_retains_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.fixture(root)
            with np.load(root / "head.npz") as stored:
                head = dict(stored)
            with (
                patch("refine_jev_readout.CANDIDATES", (("binary", .1),)),
                patch("refine_jev_readout.fit_head", return_value=(head, 0.)),
            ):
                result = refine(*args, root / "parity.json", root / "out", "revision")
            self.assertEqual(result["selected"], REFERENCE_NAME)
            self.assertFalse(result["refined_head_opens_dev_gate"])
            self.assertEqual(
                (root / "out" / (REFERENCE_NAME + ".head.npz")).read_bytes(),
                (root / "head.npz").read_bytes(),
            )

    def test_pinned_artifacts_and_sample_mutation_are_rejected(self):
        for name in ("train.features.npz", "train_holdout.features.npz",
                     "train_holdout.rows.json", "head.npz", "expected.json", "provenance.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                with (root / name).open("ab") as handle:
                    handle.write(b"x")
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    load_inputs(*args)

    def test_matching_hashes_do_not_allow_leakage_or_incorrect_reference_metrics(self):
        for mutation in ("leakage", "metrics", "coverage"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = self.fixture(root)
                provenance = json.loads((root / "provenance.json").read_bytes())
                if mutation == "leakage":
                    holdout = json.loads((root / "train_holdout.rows.json").read_bytes())
                    for row in holdout:
                        row["impression_id"] = row["impression_id"].replace("holdout-", "")
                    (root / "train_holdout.rows.json").write_text(json.dumps(holdout))
                    provenance["feature_cache_sha256"]["train_holdout.rows.json"] = sha256_file(
                        root / "train_holdout.rows.json",
                    )
                elif mutation == "metrics":
                    provenance["scaled"]["auc"] = .5
                else:
                    provenance["training_pairs"] = 5
                (root / "provenance.json").write_text(json.dumps(provenance))
                args[2] = sha256_file(root / "provenance.json")
                with self.assertRaises(ValueError):
                    load_inputs(*args)


if __name__ == "__main__":
    unittest.main()
