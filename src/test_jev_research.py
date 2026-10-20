import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from audit_jev_inputs import audit_states, dev_states
from evaluate_mind_jev import evaluate, main
from test_jev_decision import CharacterTokenizer


class BatchTokenizer(CharacterTokenizer):
    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text, **kwargs) for text in texts]}


class ResearchAuditTests(unittest.TestCase):
    def test_audit_covers_every_candidate_and_detects_over_budget(self):
        states = [(("1", "N1", 1), "short"), (("2", "N2", 0), "long" * 300)]
        result = audit_states(iter(states), BatchTokenizer(), budget=1000, batch_size=1)
        self.assertEqual(result["candidate_count"], 2)
        for contract in result["contracts"].values():
            self.assertEqual(contract["over_budget"], 1)
        repeated = audit_states(iter(states), BatchTokenizer(), budget=1000)
        self.assertEqual(result, repeated)
        with self.assertRaisesRegex(ValueError, "no candidates"):
            audit_states(iter([]), BatchTokenizer(), 1000)

    def test_dev_candidate_coverage_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "behaviors.tsv"
            path.write_text("1\tU1\ttime\tN0\tN1-1 N2-0\n", encoding="utf-8")
            news = {nid: {"text": nid} for nid in ("N0", "N1", "N2")}
            coverage = {}
            rows = list(dev_states(path, news, 30, 1, coverage))
            self.assertEqual(len(rows), 2)
            self.assertEqual(coverage, {"impressions": 1, "missing_candidates": 0, "missing_history": 0})
            with self.assertRaisesRegex(ValueError, "fewer impressions"):
                list(dev_states(path, news, 30, 2, {}))
            del news["N2"]
            with self.assertRaisesRegex(ValueError, "absent from news"):
                list(dev_states(path, news, 30, 1, {}))


class ResearchEvaluationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        behaviors = root / "behaviors.tsv"
        behaviors.write_text("1\tU1\ttime\tN0\tN1-1 N2-0 N3-0\n", encoding="utf-8")
        news = {nid: {"text": nid} for nid in ("N0", "N1", "N2", "N3")}
        args = SimpleNamespace(
            behaviors_path=str(behaviors), max_history=30, max_impressions=1,
            max_prompt_tokens=4095, model_path="fixture-model", use_abstract=True, seed=42,
        )
        return news, args

    def test_raw_scores_keep_ties_and_full_precision_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            news, args = self.fixture(directory)
            scorer = SimpleNamespace(
                prompt_contract="legacy", score_yes_batch=lambda states: [0.8, 0.8, 0.2],
            )
            output = str(Path(directory) / "predictions.txt")
            metrics = evaluate(scorer, news, args, output)
            self.assertEqual(metrics["auc"], 0.75)
            self.assertEqual(metrics["candidates"], 3)
            records = [json.loads(line) for line in Path(output).with_suffix(".scores.jsonl").read_text().splitlines()]
            self.assertEqual(records[0]["scores"], [0.8, 0.8, 0.2])
            self.assertEqual(records[0]["candidate_ids"], ["N1", "N2", "N3"])
            self.assertEqual(records[0]["labels"], [1, 0, 0])
            self.assertFalse(Path(output).with_suffix(".scores.jsonl.partial").exists())
            self.assertEqual(json.loads(Path(output).with_suffix(".metrics.json").read_text()), metrics)

    def test_invalid_scores_and_missing_candidates_cannot_be_neutralized(self):
        with tempfile.TemporaryDirectory() as directory:
            news, args = self.fixture(directory)
            for scores in ([0.8], [float("nan"), 0.5, 0.5], [0.8, -1.0, 0.2]):
                scorer = SimpleNamespace(prompt_contract="legacy", score_yes_batch=lambda states: scores)
                with self.subTest(scores=scores), self.assertRaises((ValueError, FloatingPointError)):
                    evaluate(scorer, news, args, "")
            del news["N2"]
            with self.assertRaisesRegex(ValueError, "Missing candidate"):
                evaluate(scorer, news, args, "")

    def test_both_prompt_controls_load_model_once_and_isolate_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            news, args = self.fixture(directory)
            output = str(Path(directory) / "predictions.txt")
            scorer = SimpleNamespace(prompt_contract="legacy", score_yes_batch=lambda states: [0.8, 0.8, 0.2])
            with (
                patch("evaluate_mind_jev.JevScorer", return_value=scorer) as loader,
                patch("evaluate_mind_jev.load_news", return_value=news),
                patch.object(sys, "argv", [
                    "evaluate", "--model_path", "fixture-model", "--behaviors_path", args.behaviors_path,
                    "--news_path", "fixture-news", "--prompt_contract", "both",
                    "--max_prompt_tokens", "4095", "--max_impressions", "1", "--output_file", output,
                ]),
            ):
                main()
            loader.assert_called_once()
            for contract in ("legacy", "native_click"):
                path = Path(directory) / f"predictions_{contract}.scores.jsonl"
                self.assertEqual(json.loads(path.read_text())["prompt_contract"], contract)


if __name__ == "__main__":
    unittest.main()
