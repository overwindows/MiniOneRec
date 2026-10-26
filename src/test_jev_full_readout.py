import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from behavior_jev_readout import build_counts
from full_jev_readout import (
    BEHAVIOR_NAME, TEXT_NAME, fit_full, identities, key, legacy_prefix, pending_rows, read_cache,
)
from probe_jev_readout import sample_training_rows
from scale_jev_readout import sha256_file


class FullReadoutTests(unittest.TestCase):
    def samples(self):
        article = {"text": "fixture", "category": "cat"}
        return [
            dict(impression_id=group, candidate_id=f"N{index}", label=label,
                 history=[article], candidate=article)
            for index, (group, label) in enumerate([
                ("a", 1), ("former-holdout", 1), ("unpaired", 1),
                ("b", 0), ("a", 0), ("former-holdout", 0), ("b", 1),
            ])
        ]

    def test_full_rows_include_every_group_and_former_holdout(self):
        rows = identities(self.samples())
        self.assertEqual(len(rows), 7)
        self.assertEqual({row["impression_id"] for row in rows}, {"a", "b", "former-holdout", "unpaired"})
        self.assertEqual([row["label"] for row in rows if row["impression_id"] == "unpaired"], [1])
        with self.assertRaises(ValueError):
            identities(self.samples() + [self.samples()[0]])
        with self.assertRaises(ValueError):
            identities([])

    def test_cache_prefix_exactly_matches_previous_sampling(self):
        samples = self.samples()
        expected = sample_training_rows(samples, {"former-holdout"}, 5)
        expected = [{name: row[name] for name in ("impression_id", "candidate_id", "label")} for row in expected]
        self.assertEqual(legacy_prefix(identities(samples), {"former-holdout"}, 5), expected)

    def test_shards_cover_all_rows_without_group_filtering_or_duplicates(self):
        rows = identities(self.samples())
        covered = {key(rows[0])}
        shards = []
        while len(covered) < len(rows):
            chosen = pending_rows(rows, covered, limit=2)
            self.assertTrue(chosen)
            self.assertFalse({key(row) for row in chosen} & covered)
            shards.extend(chosen)
            covered.update(key(row) for row in chosen)
        self.assertEqual(shards, rows[1:])
        self.assertEqual(pending_rows(rows, covered), [])

    def test_cache_requires_exact_order_hash_dtype_shape_and_finite_values(self):
        rows = identities(self.samples())[:2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.rows.json").write_text(json.dumps(rows))
            features, margins = np.ones((2, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)
            np.savez(root / "train.features.npz", features=features, margins=margins)
            digest = sha256_file(root / "train.features.npz")
            actual, base = read_cache(root, rows, digest, dimension=2)
            np.testing.assert_array_equal(actual, features)
            np.testing.assert_array_equal(base, margins)
            for expected, checksum in ((rows[::-1], digest), (rows, "0" * 64)):
                with self.assertRaises(ValueError):
                    read_cache(root, expected, checksum, dimension=2)
            for invalid in (features.astype(np.float64), np.full((2, 2), np.nan, dtype=np.float32), features[:, :1]):
                np.savez(root / "train.features.npz", features=invalid, margins=margins)
                with self.assertRaises(ValueError):
                    read_cache(root, rows, sha256_file(root / "train.features.npz"), dimension=2)

    def test_fixed_full_fit_uses_all_rows_without_training_selection_metric(self):
        rows = [
            {"impression_id": group, "candidate_id": news, "label": label}
            for group, news, label in (
                ("a", "P", 1), ("a", "N", 0), ("positive-only", "P", 1), ("negative-only", "N", 0),
            )
        ]
        features = np.asarray([[1, 0], [-1, 1], [2, 0], [-2, 1]], dtype=np.float32)
        margins = np.zeros(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "counts.tsv"
            log.write_text("".join(f"background-{i}\tU\tt\t\tP-1 N-0\n" for i in range(30)))
            counts, totals = build_counts(log, set())
            result = fit_full(rows, features, margins, counts, totals, root / "fit", "fixture", [])
            self.assertEqual(result["training_pairs"], 4)
            self.assertEqual(result["training_impressions"], 3)
            self.assertEqual(result["train_holdout_impressions"], 0)
            self.assertIsNone(result["artificial_pair_cap"])
            self.assertIsNone(result["quality_result"])
            self.assertEqual(result["unpaired_impressions"], 2)
            self.assertEqual(result["pairwise_supervised_impressions"], 1)
            self.assertEqual(json.loads((root / "fit" / "train.rows.json").read_bytes()), rows)
            with np.load(root / "fit" / "train-head-margins.npz", allow_pickle=False) as stored:
                text, hybrid = stored["text"], stored["hybrid"]
            with np.load(root / "fit" / (TEXT_NAME + ".head.npz"), allow_pickle=False) as head:
                expected_loss = np.logaddexp(0, (1 - 2 * np.asarray([1, 0, 1, 0])) * text).mean()
                expected_loss += .005 * np.square(head["weight"]).sum()
            self.assertAlmostEqual(result["text_training_loss"], expected_loss)
            with np.load(root / "fit" / (BEHAVIOR_NAME + ".head.npz"), allow_pickle=False) as head:
                pair_loss = np.logaddexp(0, hybrid[1] - hybrid[0]) + .005 * np.square(head["weight"]).sum()
            self.assertAlmostEqual(result["behavior_training_loss"], pair_loss)
            self.assertEqual(result["text_head_sha256"], sha256_file(root / "fit" / (TEXT_NAME + ".head.npz")))
            with self.assertRaises(FileExistsError):
                fit_full(rows, features, margins, counts, totals, root / "fit", "fixture", [])


if __name__ == "__main__":
    unittest.main()
