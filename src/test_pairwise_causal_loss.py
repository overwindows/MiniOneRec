from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent))
from jev_decision import candidate_decision_loss
from sft_mind_pointwise_pairwise import _Collator, _PairwiseTrainer, train


def batch():
    return {
        "input_ids": torch.tensor([[0, 5, 6, 2], [7, 8, 9, 3]]),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, -100, 2], [-100, -100, -100, 3]]),
        "impression_id": ["I1", "I1"],
        "label": torch.tensor([1, 0]),
    }


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([[9., 0., 2., 1.], [7., 0., 3., 5.]]))
        self.config = SimpleNamespace()
        self.called = None

    def forward(self, **kwargs):
        self.called = kwargs
        logits = self.values[:, None, :]
        return SimpleNamespace(logits=logits, loss=self.values.sum() * 0 + 3)


def trainer(model=None, weight=0.5):
    result = object.__new__(_PairwiseTrainer)
    result._yes_id, result._no_id, result._pw = 2, 3, weight
    result._warned_missing_pairs = False
    return result, model or RecordingModel()


def test_target_free_full_vocab_ce_and_margin_bt_match_independent_formula():
    runner, model = trainer()
    inputs = batch()
    original = {key: value.clone() if torch.is_tensor(value) else list(value)
                for key, value in inputs.items()}
    loss, output = runner.compute_loss(model, inputs, return_outputs=True)
    expected = F.cross_entropy(model.values, torch.tensor([2, 3]))
    expected += 0.5 * F.softplus(torch.tensor(-3.))
    torch.testing.assert_close(loss, expected)
    assert output["loss"] is loss
    torch.testing.assert_close(output["logits"], model.values[:, [2, 3]])
    torch.testing.assert_close(model.called["input_ids"], inputs["input_ids"][:, :-1])
    assert "labels" not in model.called and model.called["logits_to_keep"] == 1
    for key, value in original.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(inputs[key], value)
        else:
            assert inputs[key] == value
    loss.backward()
    assert torch.isfinite(model.values.grad).all()


def test_full_vocab_extension_does_not_change_default_binary_decision_loss():
    _, model = trainer()
    inputs = {key: value for key, value in batch().items() if key not in ("impression_id", "label")}
    full_loss, _ = candidate_decision_loss(model, inputs, (2, 3), loss_type="full_vocab")
    binary_loss, _ = candidate_decision_loss(model, inputs, (2, 3))
    torch.testing.assert_close(full_loss, F.cross_entropy(model.values, torch.tensor([2, 3])))
    torch.testing.assert_close(binary_loss, F.cross_entropy(model.values[:, [2, 3]], torch.tensor([0, 1])))
    assert full_loss > binary_loss
    with pytest.raises(ValueError, match="Unknown candidate loss"):
        candidate_decision_loss(model, inputs, (2, 3), loss_type="typo")


def test_all_pairs_are_impression_balanced_and_permutation_invariant():
    runner, _ = trainer()
    scores = torch.tensor([2., 0., -2., -4., 1., 3., 2.], requires_grad=True)
    labels = torch.tensor([1, 0, 0, 0, 1, 1, 0])
    impressions = ["A", "A", "A", "A", "B", "B", "B"]
    expected = (F.softplus(torch.tensor([-2., -4., -6.])).mean()
                + F.softplus(torch.tensor([1., -1.])).mean()) / 2
    actual = runner._pairwise_term(scores, labels, impressions)
    torch.testing.assert_close(actual, expected)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    torch.testing.assert_close(actual, runner._pairwise_term(
        scores[permutation], labels[permutation], [impressions[i] for i in permutation]))
    actual.backward()
    assert torch.all(scores.grad[labels == 1] < 0)
    assert torch.all(scores.grad[labels == 0] > 0)


def test_cross_impression_pairs_are_never_invented_and_no_rng_is_consumed():
    runner, model = trainer()
    inputs = batch()
    inputs["impression_id"] = ["A", "B"]
    rng = torch.random.get_rng_state().clone()
    with pytest.warns(RuntimeWarning, match="CE only"):
        loss = runner.compute_loss(model, inputs)
    torch.testing.assert_close(loss, F.cross_entropy(model.values, torch.tensor([2, 3])))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_extreme_incorrect_rank_has_finite_unsaturated_gradient():
    runner, _ = trainer()
    scores = torch.tensor([-1000., 1000.], requires_grad=True)
    loss = runner._pairwise_term(scores, torch.tensor([1, 0]), ["A", "A"])
    torch.testing.assert_close(loss, torch.tensor(2000.))
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-1., 1.]))


@pytest.mark.parametrize("key,value", [
    ("impression_id", ["A"]), ("impression_id", ["", "A"]),
    ("label", torch.tensor([1, 2])), ("label", torch.tensor([0, 1])),
])
def test_invalid_metadata_fails_before_model_forward(key, value):
    runner, model = trainer()
    inputs = batch()
    inputs[key] = value
    with pytest.raises(ValueError):
        runner.compute_loss(model, inputs)
    assert model.called is None


def test_missing_metadata_is_not_silently_downgraded_to_ce():
    runner, model = trainer()
    inputs = batch()
    del inputs["impression_id"]
    with pytest.raises(ValueError, match="requires impression_id"):
        runner.compute_loss(model, inputs)
    assert model.called is None


def test_zero_weight_preserves_legacy_ce_forward():
    runner, model = trainer(weight=0)
    loss = runner.compute_loss(model, batch())
    torch.testing.assert_close(loss, torch.tensor(3.))
    assert "labels" in model.called and "impression_id" not in model.called


@pytest.mark.parametrize("weight", (-1., float("nan"), float("inf")))
def test_invalid_weights_fail_before_trainer_initialization(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _PairwiseTrainer(2, 3, pairwise_loss_weight=weight)


def test_existing_final_checkpoint_is_never_overwritten(tmp_path):
    (tmp_path / "final_checkpoint").mkdir()
    with patch("sft_mind_pointwise_pairwise.AutoModelForCausalLM.from_pretrained") as loader:
        with pytest.raises(ValueError, match="overwrite"):
            train(output_dir=str(tmp_path))
        loader.assert_not_called()


def test_failed_trial_artifacts_are_not_reused_without_explicit_compatible_resume(tmp_path):
    (tmp_path / "checkpoint-1").mkdir()
    with pytest.raises(ValueError, match="prior trial artifacts"):
        train(output_dir=str(tmp_path))

@pytest.mark.parametrize("tokens", (([], [3]), ([2, 4], [3]), ([2], [2]), ([0], [3])))
def test_invalid_answer_tokens_fail_before_loading_model_weights(tmp_path, tokens):
    tokenizer = SimpleNamespace(
        eos_token="eos", eos_token_id=1, all_special_ids=[0, 1],
        encode=lambda answer, **kwargs: tokens[answer == " No"])
    with (patch("sft_mind_pointwise_pairwise.AutoTokenizer.from_pretrained", return_value=tokenizer),
          patch("sft_mind_pointwise_pairwise.AutoModelForCausalLM.from_pretrained") as loader):
        with pytest.raises(ValueError, match="distinct single"):
            train(base_model="fixture", output_dir=str(tmp_path))
        loader.assert_not_called()


def test_legacy_optimizer_resume_is_rejected_before_loading_model_weights(tmp_path):
    tokenizer = SimpleNamespace(
        eos_token="eos", eos_token_id=1, all_special_ids=[0, 1],
        encode=lambda answer, **kwargs: [3 if answer == " No" else 2])
    with (patch("sft_mind_pointwise_pairwise.AutoTokenizer.from_pretrained", return_value=tokenizer),
          patch("sft_mind_pointwise_pairwise.AutoConfig.from_pretrained", return_value=SimpleNamespace()),
          patch("sft_mind_pointwise_pairwise.AutoModelForCausalLM.from_pretrained") as loader):
        with pytest.raises(ValueError, match="incompatible optimizer"):
            train(base_model="fixture", output_dir=str(tmp_path), resume_from_checkpoint="legacy")
        loader.assert_not_called()


def test_actual_qwen3_causal_ce_matches_prompt_only_forward_and_has_gradients():
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, pad_token_id=0, eos_token_id=1, attention_dropout=0.0))
    inputs = batch()
    positions = inputs["attention_mask"].cumsum(-1) - 1
    positions.masked_fill_(inputs["attention_mask"] == 0, 0)
    with torch.no_grad():
        reference = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                          position_ids=positions, labels=inputs["labels"], use_cache=False)
    prompt_inputs = {key: value for key, value in inputs.items()
                     if key not in ("impression_id", "label")}
    ce, _ = candidate_decision_loss(model, prompt_inputs, (2, 3), loss_type="full_vocab")
    torch.testing.assert_close(ce, reference.loss)
    runner, _ = trainer()
    loss = runner.compute_loss(model, inputs)
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_real_trainer_keeps_metadata_in_cpu_dataloader_and_takes_one_step(tmp_path):
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, pad_token_id=0, eos_token_id=1))
    inputs = batch()
    features = [{key: value[index] for key, value in inputs.items()} for index in range(2)]
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to="none",
        remove_unused_columns=False, per_device_train_batch_size=2,
        max_steps=1, save_strategy="no", logging_strategy="no")
    runner = _PairwiseTrainer(
        2, 3, model=model, args=args, train_dataset=features,
        data_collator=_Collator(SimpleNamespace(pad_token_id=0)))
    actual = next(iter(runner.get_train_dataloader()))
    assert sorted(actual["impression_id"]) == ["I1", "I1"]
    assert not runner.model_accepts_loss_kwargs
    assert torch.isfinite(runner.compute_loss(model, actual))
    runner.train()
    assert runner.state.global_step == 1
