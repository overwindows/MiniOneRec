import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from behavior_jev_readout import (
    behavior_features, build_counts, impression_fold, load_counts, save_counts, train,
)
from scale_jev_readout import sha256_file
import test_jev_refinement as refinement_tests


class BehaviorTests(unittest.TestCase):
    def write(self, path, heldout_label=1, target_label=1):
        records = [
            f"train-{index}\tU\ttime\t\tN0-0 N1-1 N2-0 N3-1"
            for index in range(30)
        ]
        records.extend([
            f"target\tU\ttime\t\tN0-0 NEW-{target_label}",
            f"heldout\tU\ttime\t\tN0-0 HELD-{heldout_label}",
        ])
        path.write_text("\n".join(records) + "\n", encoding="utf-8")

    def test_fold_is_stable_and_features_are_finite_for_unseen_items(self):
        self.assertEqual(impression_fold("target"), impression_fold("target"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            self.write(path)
            counts, totals = build_counts(path, {"heldout"})
            rows = [{"impression_id": "target", "candidate_id": "UNKNOWN"}]
            features = behavior_features(rows, counts, totals, True)
            np.testing.assert_array_equal(features, np.zeros((1, 4)))
            self.assertNotIn("HELD", counts)
            self.assertEqual(int(totals[0, 0]), 122)
            self.assertEqual(int(totals[0, 1]), 61)

    def test_own_fold_and_entire_holdout_label_changes_cannot_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = [{"impression_id": "target", "candidate_id": "NEW"}]
            observed = []
            for target, heldout in ((0, 0), (0, 1), (1, 0), (1, 1)):
                path = root / f"{target}-{heldout}.tsv"
                self.write(path, heldout, target)
                counts, totals = build_counts(path, {"heldout"})
                observed.append(behavior_features(row, counts, totals, True))
            for value in observed[1:]:
                np.testing.assert_array_equal(value, observed[0])

    def test_full_training_statistics_have_exact_smoothed_ctr_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            self.write(path)
            counts, totals = build_counts(path, {"heldout"})
            x = behavior_features([{"impression_id": "heldout", "candidate_id": "N1"}], counts, totals, False)[0]
            prior = 61 / 122
            ctr = (30 + 20 * prior) / (30 + 20)
            expected = [np.log1p(1e6 * 30 / 122), np.log1p(1e6 * 30 / 61),
                        np.log(ctr / (1 - ctr)) - np.log(prior / (1 - prior)), 1.]
            np.testing.assert_allclose(x, expected, atol=1e-14)

    def test_statistics_round_trip_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root / "behaviors.tsv")
            counts, totals = build_counts(root / "behaviors.tsv", {"heldout"})
            path = root / "population.npz"
            save_counts(path, counts, totals)
            restored, actual = load_counts(path, sha256_file(path))
            np.testing.assert_array_equal(actual, totals)
            for key in counts:
                np.testing.assert_array_equal(counts[key], restored[key])
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_counts(path, "0" * 64)
            with np.load(path) as data:
                altered = dict(data)
            altered["counts"] = altered["counts"].copy()
            altered["counts"][0, 0, 0] += 1
            np.savez(path, **altered)
            with self.assertRaisesRegex(ValueError, "count identities"):
                load_counts(path, sha256_file(path))

    def test_malformed_rows_missing_ids_and_cached_label_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            for text in ("bad\n", "a\tU\tt\t\tN-2\n", "a\tU\tt\t\tN-0 N-1\n",
                         "a\tU\tt\t\tN-0 P-1\na\tU\tt\t\tN-0 P-1\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    build_counts(path, set())
            self.write(path)
            with self.assertRaises(ValueError):
                build_counts(path, {"absent"})
            with self.assertRaises(ValueError):
                build_counts(path, {"heldout"}, [
                    {"impression_id": "target", "candidate_id": "NEW", "label": 0},
                ])

    def test_train_only_end_to_end_artifacts_and_reference_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = refinement_tests.RefinementTests().fixture(root)
            path = root / "behaviors.tsv"
            path.write_text(
                "a\tU\tt\t\t0-0 1-1\nb\tU\tt\t\t2-0 3-1\n"
                "holdout-a\tU\tt\t\t0-0 1-1\nholdout-b\tU\tt\t\t2-0 3-1\n"
                + "".join(f"extra-{i}\tU\tt\t\t0-0 1-1 2-0 3-1\n" for i in range(20)),
            )
            provenance = json.loads((root / "provenance.json").read_bytes())
            provenance["data_sha256"]["train/behaviors.tsv"] = sha256_file(path)
            (root / "provenance.json").write_text(json.dumps(provenance))
            args[2] = sha256_file(root / "provenance.json")
            zero = dict(mean=np.zeros(4), scale=np.ones(4), weight=np.zeros(4), bias=np.asarray(0.))
            with (
                patch("behavior_jev_readout.fit_head", return_value=(zero, 0.)),
                patch("builtins.print"),
            ):
                result = train(*args, path, root / "out", "source")
            self.assertTrue(result["training_only"])
            self.assertFalse(result["behavior_head_opens_dev_gate"])
            self.assertEqual(result["selected_head_sha256"], args[4])
            self.assertEqual(result["minimum_dev_start"], 26000)
            selection = json.loads((root / "out" / "train-selection.json").read_bytes())
            self.assertEqual(len(selection["candidates"]), 3)
            with self.assertRaises(FileExistsError):
                train(*args, path, root / "out", "source")


if __name__ == "__main__":
    unittest.main()
