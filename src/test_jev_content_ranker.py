import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from behavior_jev_readout import build_counts
from jev_content_ranker import (
    BASE_COLUMNS, FEATURE_NAMES, Content, assembled_features, behavior_records,
    category_codes, content_features, fit_model, ranking_metrics, read_news,
)


class ContentRankerTests(unittest.TestCase):
    def news(self):
        return {
            f"N{index}": {
                "category": "sports" if index % 2 else "politics",
                "subcategory": "football" if index % 2 else "election",
                "text": ("sports football championship team game score " if index % 2
                         else "politics election government policy vote debate ") + str(index),
            }
            for index in range(12)
        }

    def test_strict_news_and_behavior_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "news.tsv"
            path.write_text("N1\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\n")
            self.assertEqual(read_news(path)["N1"]["text"], "title abstract")
            for text in ("N1\tcat\n", path.read_text() * 2):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    read_news(path)
            path = Path(directory) / "behaviors.tsv"
            good = "A\tU\t11/14/2019 1:00:00 PM\tN1 N2\tN1-1 N2-0\n"
            path.write_text(good)
            self.assertEqual(next(behavior_records(path))["day"], "2019-11-14")
            for text in (good * 2, good.replace("N2-0", "N1-0"), good.replace("N2-0", "N2-2")):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    list(behavior_records(path))

    def test_content_features_have_exact_order_and_cold_start_flags(self):
        news = self.news()
        encoder = Content.fit(news, components=4)
        records = [
            {"impression_id": "a", "history": ("N1",),
             "items": [("N1", 1), ("N2", 0), ("UNKNOWN", 0)]},
            {"impression_id": "b", "history": (), "items": [("N2", 1)]},
        ]
        rows, values = content_features(records, encoder.table(news))
        self.assertEqual(values.shape, (4, 20))
        self.assertEqual([row["candidate_id"] for row in rows], ["N1", "N2", "UNKNOWN", "N2"])
        self.assertAlmostEqual(values[0, 14], 1., places=5)
        self.assertEqual(values[0, 6], 1.)
        self.assertEqual(values[1, 6], 0.)
        self.assertEqual(values[2, 4], 1.)
        self.assertEqual(values[0, 19], 1.)
        np.testing.assert_array_equal(values[3, 5:], np.zeros(15))
        self.assertTrue(np.isfinite(values).all())

    def test_transforming_new_news_does_not_refit_vocabulary_or_categories(self):
        news = self.news()
        encoder = Content.fit(news, components=4)
        vocabulary = dict(encoder.vectorizer.vocabulary_)
        category = dict(encoder.categories)
        extra = {"NEW": {"category": "brandnew", "subcategory": "unseen", "text": "neverobservedtoken"}}
        table = encoder.table({**news, **extra})
        self.assertEqual(encoder.vectorizer.vocabulary_, vocabulary)
        self.assertEqual(encoder.categories, category)
        self.assertEqual(table[2][table[0]["NEW"]], 0)
        np.testing.assert_array_equal(table[1][table[0]["NEW"]], np.zeros(4))

    def test_temporal_validation_and_own_fold_labels_cannot_enter_fit_features(self):
        row = [{"impression_id": "target", "candidate_id": "N1", "label": 1}]
        validation = [{"impression_id": "heldout", "candidate_id": "N1", "label": 0}]
        observed_fit, observed_validation = [], []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            for own, heldout in ((0, 0), (0, 1), (1, 0), (1, 1)):
                text = "".join(
                    f"train-{index}\tU\tt\t\tN1-1 N2-0\n" for index in range(50)
                )
                text += f"target\tU\tt\t\tN1-{own} N2-0\nheldout\tU\tt\t\tN1-{heldout} N2-0\n"
                path.write_text(text)
                counts, totals = build_counts(path, {"heldout"})
                observed_fit.append(assembled_features(
                    row, np.zeros((1, 20)), [0.25], counts, totals, True,
                ))
                if own == 0:
                    observed_validation.append(assembled_features(
                        validation, np.zeros((1, 20)), [0.25], counts, totals, False,
                    ))
            for value in observed_fit[1:]:
                np.testing.assert_array_equal(value, observed_fit[0])
            np.testing.assert_array_equal(observed_validation[0], observed_validation[1])

    def test_rank_metrics_are_impression_macro_averages_and_report_unpaired(self):
        rows = [
            {"impression_id": name, "label": label}
            for name, label in (("a", 1), ("a", 0), ("b", 1), ("b", 0), ("c", 1))
        ]
        result = ranking_metrics(rows, [1., 0., 0., 1., 9.])
        self.assertEqual(result["metrics"]["auc"], .5)
        self.assertEqual(result["scored_impressions"], 2)
        self.assertEqual(result["single_class_impressions"], 1)
        with self.assertRaises(ValueError):
            ranking_metrics(rows, [float("nan")] * 5)

    def test_small_model_fit_is_finite_and_uses_fixed_iterations(self):
        rows = [
            {"impression_id": str(index // 2), "label": index % 2}
            for index in range(400)
        ]
        features = np.random.default_rng(42).normal(size=(400, len(FEATURE_NAMES))).astype(np.float32)
        features[:, 1:3] = 0
        recipe = {"columns": BASE_COLUMNS, "leaves": 3, "iterations": 2}
        model = fit_model(recipe, features, rows)
        scores = model.decision_function(features[:, BASE_COLUMNS])
        self.assertEqual(model.n_iter_, 2)
        self.assertFalse(model.early_stopping)
        self.assertTrue(np.isfinite(scores).all())

    def test_category_encoding_is_bounded_and_no_raw_ids_are_features(self):
        news = {
            str(index): {"category": str(index), "subcategory": str(index)}
            for index in range(300)
        }
        self.assertEqual(max(category_codes(news, "category").values()), 254)
        self.assertFalse(any(name in FEATURE_NAMES for name in ("impression_id", "candidate_id", "label")))


if __name__ == "__main__":
    unittest.main()
