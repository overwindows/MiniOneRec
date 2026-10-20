import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from jev_decision import (
    BINARY_CRITERIA, candidate_decision_loss, decision_token_ids,
    encode_decision_prompt, render_noul_prompt,
)
from evaluate_mind_jev import JevScorer, build_jev_state
from jev_noul_data import JEVNoulPointwiseDataset
from sft_mind_jev import JEVWithinModelDistillTrainer


class CharacterTokenizer:
    all_special_ids = [0]
    chat_template = "test-template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "<user>" + messages[0]["content"] + "</user><assistant>\n"

    def encode(self, text, add_special_tokens=False, max_length=None, truncation=False):
        ids = [ord(char) + 10 for char in text]
        return ids[:max_length] if truncation else ids


class RecordingModel(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.called = None

    def forward(self, **kwargs):
        self.called = kwargs
        return SimpleNamespace(logits=self.values.expand(kwargs["input_ids"].shape[0], 1, -1))


def batch():
    return {
        "input_ids": torch.tensor([[0, 5, 6, 2], [7, 8, 9, 3]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, -100, 2], [-100, -100, -100, 3]]),
    }


class DecisionLossTests(unittest.TestCase):
    def test_binary_loss_matches_candidate_ce_and_ignores_other_classes(self):
        model = RecordingModel([1000.0, -1000.0, 2.0, -1.0, 900.0])
        inputs = batch()
        loss, output = candidate_decision_loss(model, inputs, (2, 3))
        expected = F.cross_entropy(model.values[[2, 3]].expand(2, -1), torch.tensor([0, 1]))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(model.values.grad).all())
        self.assertEqual(model.values.grad[[0, 1, 4]].abs().sum().item(), 0)
        torch.testing.assert_close(model.called["input_ids"], inputs["input_ids"][:, :-1])
        torch.testing.assert_close(model.called["position_ids"], torch.tensor([[0, 0, 1], [0, 1, 2]]))
        self.assertNotIn("labels", model.called)
        self.assertEqual(model.called["logits_to_keep"], 1)
        self.assertFalse(model.called["use_cache"])
        self.assertEqual(output["logits"].shape, (2, 2))
        torch.testing.assert_close(output["logits"].softmax(-1)[:, 0],
                                   (output["logits"][:, 0] - output["logits"][:, 1]).sigmoid())

    def test_bfloat16_candidate_loss_is_fp32_and_finite(self):
        model = RecordingModel([0, 0, 1000, -1000]).to(torch.bfloat16)
        loss, _ = candidate_decision_loss(model, batch(), (2, 3))
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(model.values.grad).all())

    def test_invalid_batches_fail_before_forward(self):
        cases = []
        for key, row, column, value in [
            ("labels", 0, 0, 2), ("labels", 0, -1, 4),
            ("input_ids", 0, -1, 3), ("attention_mask", 0, -1, 0),
            ("attention_mask", 1, 1, 0),
        ]:
            inputs = batch()
            inputs[key][row, column] = value
            cases.append(inputs)
        for inputs in cases:
            with self.subTest(inputs=inputs):
                model = RecordingModel([0, 0, 0, 0])
                with self.assertRaises(ValueError):
                    candidate_decision_loss(model, inputs, (2, 3))
                self.assertIsNone(model.called)

    def test_nonfinite_decision_logits_raise(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                candidate_decision_loss(RecordingModel([0, 0, value, 0]), batch(), (2, 3))

    def test_unsupported_kl_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "enable_kl=False"):
            JEVWithinModelDistillTrainer(loss_type="decision", enable_kl=True)


class PromptTests(unittest.TestCase):
    def test_native_criteria_and_legacy_prompt(self):
        native = render_noul_prompt("state", "native_click")
        parsed = json.loads(native.split("\n\n", 1)[1].split("\nReturn only", 1)[0])
        self.assertEqual(parsed["criteria"], [
            {"label": label, "description": candidate["description"]}
            for label, candidate in zip(("A", "B"), BINARY_CRITERIA)
        ])
        self.assertIn("click", parsed["instructions"])
        legacy = render_noul_prompt("state")
        self.assertIn('"description": "yes"', legacy)
        self.assertIn("matches their interests", legacy)
        with self.assertRaises(ValueError):
            render_noul_prompt("state", "typo")

    def test_boundaries_and_no_silent_native_truncation(self):
        tokenizer = CharacterTokenizer()
        ids = encode_decision_prompt(tokenizer, "prompt")
        self.assertEqual(decision_token_ids(tokenizer), (75, 76))
        with self.assertRaisesRegex(ValueError, "never silently truncated"):
            encode_decision_prompt(tokenizer, "prompt", max_length=3)
        self.assertEqual(encode_decision_prompt(tokenizer, "prompt", max_length=3, truncate=True), ids[:3])
        original = tokenizer.encode
        with patch.object(tokenizer, "encode", side_effect=lambda text, **kw: [999] if text.endswith("\nA") else original(text, **kw)):
            with self.assertRaisesRegex(ValueError, "answer boundary"):
                encode_decision_prompt(tokenizer, "prompt")

    def test_invalid_candidate_tokens_are_rejected(self):
        tokenizer = CharacterTokenizer()
        for tokens in ([7, 8], [7], [0]):
            with self.subTest(tokens=tokens), patch.object(tokenizer, "encode", return_value=tokens):
                with self.assertRaises(ValueError):
                    decision_token_ids(tokenizer)

    def test_real_dataset_and_scorer_share_the_same_prompt_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "news.tsv").write_text(
                "N0\tcat\tsub\tHistory\tAbstract\n"
                "N1\tcat\tsub\tPositive\tAbstract\n"
                "N2\tcat\tsub\tNegative\tAbstract\n", encoding="utf-8",
            )
            (root / "behaviors.tsv").write_text(
                "1\tU1\ttime\tN0\tN1-1 N2-0\n", encoding="utf-8",
            )
            for contract in ("legacy", "native_click"):
                with self.subTest(contract=contract):
                    dataset = JEVNoulPointwiseDataset(
                        str(root / "behaviors.tsv"), str(root / "news.tsv"),
                        CharacterTokenizer(), max_len=4096, prompt_contract=contract,
                        hard_neg_ratio=1.0,
                    )
                    scorer = object.__new__(JevScorer)
                    scorer.tokenizer = dataset.tokenizer
                    scorer.prompt_contract = contract
                    self.assertEqual(len(dataset), 2)
                    for index, sample in enumerate(dataset.samples):
                        item = dataset[index]
                        state = build_jev_state(sample["history"], sample["candidate"])
                        self.assertEqual(item["input_ids"][:-1].tolist(), scorer._encode(state))
                        self.assertEqual((item["labels"] != -100).sum().item(), 1)
                        self.assertEqual(item["labels"][-1].item(), 75 if sample["label"] else 76)
            with self.assertRaisesRegex(ValueError, "never silently truncated"):
                JEVNoulPointwiseDataset(
                    str(root / "behaviors.tsv"), str(root / "news.tsv"),
                    CharacterTokenizer(), max_len=16, prompt_contract="native_click",
                )

    def test_scorer_restores_checkpoint_prompt_mode_or_explicit_override(self):
        config = SimpleNamespace(model_type="qwen3_5_text", jev_prompt_contract="native_click")
        model = RecordingModel([0, 0, 0, 0])
        with (
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
            patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as loader,
            patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()),
        ):
            scorer = JevScorer("test-model", device="cpu")
            self.assertEqual(scorer.prompt_contract, "native_click")
            scorer = JevScorer("test-model", device="cpu", prompt_contract="legacy")
            self.assertEqual(scorer.prompt_contract, "legacy")
            del config.jev_prompt_contract
            self.assertEqual(JevScorer("test-model", device="cpu").prompt_contract, "legacy")
            config.jev_prompt_contract = "unknown"
            with self.assertRaisesRegex(ValueError, "Unknown JEV prompt contract"):
                JevScorer("test-model", device="cpu")
            self.assertEqual(loader.call_count, 3)


class QwenIntegrationTests(unittest.TestCase):
    def test_actual_hybrid_model_forward_backward_and_checkpoint_metadata(self):
        from transformers import AutoConfig, Qwen3_5ForCausalLM, Qwen3_5TextConfig, TrainingArguments

        torch.manual_seed(7)
        config = Qwen3_5TextConfig(
            vocab_size=128, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=2,
            layer_types=["linear_attention", "full_attention"],
            pad_token_id=0, eos_token_id=1,
        )
        config.jev_prompt_contract = "native_click"
        config.jev_loss_type = "decision"
        model = Qwen3_5ForCausalLM(config)
        model.gradient_checkpointing_enable()
        model.train()
        with tempfile.TemporaryDirectory() as directory:
            trainer = JEVWithinModelDistillTrainer(
                model=model, enable_kl=False, loss_type="decision", decision_token_ids=(2, 3),
                args=TrainingArguments(output_dir=directory, report_to="none", use_cpu=True),
            )
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
            initial = trainer.compute_loss(model, batch()).item()
            for _ in range(3):
                optimizer.zero_grad()
                loss, outputs = trainer.compute_loss(model, batch(), return_outputs=True)
                self.assertEqual(outputs["logits"].shape, (2, 2))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertLess(trainer.compute_loss(model, batch()).item(), initial)
            model.save_pretrained(directory)
            restored = AutoConfig.from_pretrained(directory, local_files_only=True)
            self.assertEqual(restored.jev_prompt_contract, "native_click")
            self.assertEqual(restored.jev_loss_type, "decision")
            legacy = object.__new__(JEVWithinModelDistillTrainer)
            legacy.loss_type = "full_vocab"
            expected = model(**batch(), return_dict=True).loss
            torch.testing.assert_close(legacy.compute_loss(model, batch()), expected)
            trainer.train_dataset = [
                {key: value[row] for key, value in batch().items()} for row in range(2)
            ]
            trainer.args.max_steps = 2
            trainer.args.per_device_train_batch_size = 1
            trainer.args.gradient_accumulation_steps = 2
            trainer.args.save_strategy = "no"
            trainer.args.disable_tqdm = True
            forward_calls = []
            hook = model.register_forward_pre_hook(lambda _model, _args: forward_calls.append(None))
            self.addCleanup(hook.remove)
            result = trainer.train()
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            self.assertEqual(trainer.state.global_step, 2)
            self.assertEqual(len(forward_calls), 4)


if __name__ == "__main__":
    unittest.main()
