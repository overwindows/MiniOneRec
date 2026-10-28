import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

import evaluate_jev_content as evaluator
from behavior_jev_readout import build_counts, save_counts
from evaluate_jev_content import checked_json, evaluate, paired_intervals
from jev_content_ranker import Content, digest, fit_model, write_json
import test_jev_content_ranker as ranker_tests


class ContentEvaluationTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_preserves_pairing(self):
        baseline = np.tile([.6, .3, .4, .5], (20, 1))
        candidate = baseline + np.asarray([.02, .01, -.01, 0.])
        first = paired_intervals(candidate, baseline)
        self.assertEqual(first, paired_intervals(candidate, baseline))
        for key, delta in zip(("auc", "mrr", "ndcg5", "ndcg10"), (.02, .01, -.01, 0.)):
            self.assertAlmostEqual(first[key]["delta"], delta)
            np.testing.assert_allclose(first[key]["ci95"], [delta, delta], atol=1e-14)
        with self.assertRaises(ValueError):
            paired_intervals(np.full((20, 4), np.nan), baseline)
        with self.assertRaises(ValueError):
            paired_intervals(np.zeros((0, 4)), np.zeros((0, 4)))

    def test_manifest_changes_are_rejected_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"selected": "frozen"}))
            checksum = digest(path)
            self.assertEqual(checked_json(path, checksum)["selected"], "frozen")
            path.write_text(json.dumps({"selected": "changed"}))
            with self.assertRaisesRegex(ValueError, "checksum"):
                checked_json(path, checksum)

    def test_complete_frozen_evaluation_and_source_order_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, reference = root / "training", root / "reference"
            training.mkdir()
            reference.mkdir()
            news = ranker_tests.ContentRankerTests().news()
            encoder = Content.fit(news, components=4)
            rows = [{"impression_id": str(index // 2), "label": index % 2} for index in range(400)]
            features = np.zeros((400, 25), dtype=np.float32)
            features[:, 0] = [row["label"] for row in rows]
            recipe = {"name": "content_15", "columns": tuple(range(25)), "leaves": 3, "iterations": 2}
            model = fit_model(recipe, features, rows)
            joblib.dump({
                "model": model, "content": {
                    "vectorizer": encoder.vectorizer, "svd": encoder.svd,
                    "categories": encoder.categories, "subcategories": encoder.subcategories,
                }, "news": news, "recipe": recipe,
            }, training / "model.joblib")
            counts_path = root / "counts.tsv"
            counts_path.write_text("".join(f"T{i}\tU\tt\t\tN1-1 N2-0\n" for i in range(30)))
            counts, totals = build_counts(counts_path, set())
            for path in (training, reference):
                save_counts(path / "population.npz", counts, totals)
            np.savez(reference / "retained.head.npz", mean=np.zeros(4), scale=np.ones(4),
                     weight=np.zeros(4), bias=np.asarray(0.))
            head_sha = digest(reference / "retained.head.npz")
            selection = {
                "selected": "content_15",
                "candidates": [
                    {"name": "frozen_popularity", "metrics": {"auc": .6}},
                    {"name": "content_15", "metrics": {"auc": .7}},
                ],
            }
            write_json(training / "train-selection.json", selection)
            write_json(training / "result.json", {
                "stage": "full_training_fitted", "training_only": True, "opens_dev_gate": True,
                "selected": "content_15", "training_impressions": 156965, "training_rows": 469264,
                "final_refit_holdout_impressions": 0,
                "train_selection_sha256": digest(training / "train-selection.json"),
                "model_sha256": digest(training / "model.joblib"),
                "population_sha256": digest(training / "population.npz"),
            })
            write_json(reference / "result.json", {
                "selected": "retained", "selected_head_sha256": head_sha,
                "population_sha256": digest(reference / "population.npz"),
            })
            news_path = root / "news.tsv"
            news_path.write_text("".join(
                f"{name}\t{article['category']}\t{article['subcategory']}\t{article['text']}\t\turl\t[]\t[]\n"
                for name, article in news.items()
            ))
            behaviors = root / "behaviors.tsv"
            behaviors.write_text("".join(
                f"D{i}\tU\t11/15/2019 1:00:00 PM\tN1\tN1-1 N2-0\n" for i in range(2)
            ))
            raw = [
                {"impression_id": f"D{i}", "candidate_ids": ["N1", "N2"], "labels": [1, 0],
                 "frozen_margins": [1., 0.], "selected_margins": [1., 0.],
                 "reference_margins": [1., 0.]} for i in range(2)
            ]
            raw_path = root / "scores.jsonl"
            raw_path.write_text("".join(json.dumps(record) + "\n" for record in raw))
            base_path = root / "base.json"
            metrics = {"auc": 1., "mrr": 1., "ndcg5": 1., "ndcg10": 1.}
            write_json(base_path, {
                "dev_start_zero_based": 0, "dev_impressions": 2, "dev_candidates": 4,
                "source_revision": evaluator.CONTROL_SOURCE,
                "head_sha256": evaluator.FULL_TEXT_SHA,
                "reference_head_sha256": evaluator.RETAINED_TEXT_SHA,
                "data_sha256": {"dev/news.tsv": digest(news_path), "dev/behaviors.tsv": digest(behaviors)},
                "frozen": metrics, "selected_metrics": metrics, "reference_metrics": metrics,
            })
            args = (training, digest(training / "result.json"), reference,
                    digest(reference / "result.json"), base_path, digest(base_path))
            with patch.multiple(
                evaluator, DEV_START=0, DEV_IMPRESSIONS=2, DEV_NEWS_SHA=digest(news_path),
                DEV_BEHAVIORS_SHA=digest(behaviors), RETAINED_BEHAVIOR_SHA=head_sha,
            ), patch("builtins.print"):
                result = evaluate(*args, raw_path, digest(raw_path), news_path, behaviors, root / "out")
                self.assertEqual(result["dev_candidates"], 4)
                self.assertEqual(result["metrics"]["frozen"]["auc"], 1.)
                self.assertFalse(result["project_target_achieved"])
                self.assertEqual(len(result["metrics"]), 5)
                raw[0]["candidate_ids"].reverse()
                raw_path.write_text("".join(json.dumps(record) + "\n" for record in raw))
                with self.assertRaisesRegex(ValueError, "identities"):
                    evaluate(*args, raw_path, digest(raw_path), news_path, behaviors, root / "bad")


if __name__ == "__main__":
    unittest.main()
