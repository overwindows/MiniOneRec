import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import MINDPointwiseSFTDataset
from jev_decision import candidate_decision_loss
from mind_impression_data import MINDImpressionSFTDataset
from mind_pairwise_loss import complete_impression_loss, within_impression_bt
from sft_mind_pointwise_pairwise import _Collator
from sft_mind_pointwise_pairwise import CompleteImpressionTrainer, train
from mind_impression_training import CompleteImpressionCollator
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments


class Tokenizer:
    all_special_ids = [0, 1]
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "eos"

    def encode(self, text, max_length=None, truncation=False, **kwargs):
        if text in (" Yes", " No"):
            return [2 if text == " Yes" else 3]
        ids = [4 + ord(character) % 12 for character in text]
        return ids[:max_length] if truncation else ids

    def save_pretrained(self, directory):
        (Path(directory) / "fixture_tokenizer.txt").write_text("synthetic tokenizer")


@pytest.fixture
def sources(tmp_path):
    news = tmp_path / "news.tsv"
    behaviors = tmp_path / "behaviors.tsv"
    news.write_text("N0\tc\ts\tHistory\tA\turl\t[]\t[]\n"
                    "N1\tc\ts\tClick\tA\turl\t[]\t[]\n"
                    "N2\tc\ts\tNegative\tA\turl\t[]\t[]\n"
                    "N3\td\ts\tOther\tA\turl\t[]\t[]\n")
    behaviors.write_text("A\tU1\tt\tN0\tN1-1 N2-0 N2-0\nB\tU2\tt\t\tN3-0 N2-0\n")
    return behaviors, news


def test_every_candidate_exposure_and_one_class_impression_is_retained(sources):
    behaviors, news = sources
    dataset = MINDImpressionSFTDataset(behaviors, news, Tokenizer(), max_len=4096)
    assert len(dataset) == 2 and not hasattr(dataset, "samples")
    assert dataset.offsets.typecode == "Q"
    manifest = dataset.source_manifest
    assert (manifest["impressions"], manifest["exposures"], manifest["positives"]) == (2, 5, 1)
    assert manifest["candidate_sampling"] == "none" and manifest["max_candidates"] == 3
    assert manifest["behaviors_sha256"] == hashlib.sha256(behaviors.read_bytes()).hexdigest()
    assert manifest["news_sha256"] == hashlib.sha256(news.read_bytes()).hexdigest()
    assert [feature["label"] for feature in dataset[0]["candidates"]] == [1, 0, 0]
    assert [feature["label"] for feature in dataset[1]["candidates"]] == [0, 0]
    assert dataset[0]["expected_candidates"] == 3


def test_shared_encoder_preserves_existing_pointwise_candidate_tokens(sources):
    behaviors, news = sources
    complete = MINDImpressionSFTDataset(behaviors, news, Tokenizer(), max_len=4096)
    legacy = MINDPointwiseSFTDataset(
        str(behaviors), str(news), Tokenizer(), max_len=4096,
        neg_ratio=100, negative_sampling="uniform", return_meta=True)
    expected = sorted((tuple(item["input_ids"].tolist()), tuple(item["labels"].tolist()))
                      for item in complete[0]["candidates"])
    actual = sorted((tuple(legacy[index]["input_ids"].tolist()),
                     tuple(legacy[index]["labels"].tolist())) for index in range(len(legacy)))
    assert actual == expected


@pytest.mark.parametrize("row", [
    "A\tU\tt\t\tN1", "A\tU\tt\t\tN1-2", "A\tU\tt\t\tMissing-0",
    "A\tU\tt\tMissing\tN1-1", "A\tU\tt\t\t", "\n",
])
def test_malformed_or_uncovered_source_rows_never_disappear(sources, row):
    behaviors, news = sources
    behaviors.write_text(row)
    with pytest.raises(ValueError):
        MINDImpressionSFTDataset(behaviors, news, Tokenizer())


def test_duplicate_impressions_and_duplicate_news_are_rejected(sources):
    behaviors, news = sources
    behaviors.write_text("A\tU\tt\t\tN1-1\nA\tU\tt\t\tN2-0\n")
    with pytest.raises(ValueError, match="Duplicate source impression"):
        MINDImpressionSFTDataset(behaviors, news, Tokenizer())
    behaviors.write_text("A\tU\tt\t\tN1-1\n")
    news.write_text(news.read_text() + "N1\tc\ts\tDuplicate\tA\turl\t[]\t[]\n")
    with pytest.raises(ValueError, match="complete record"):
        MINDImpressionSFTDataset(behaviors, news, Tokenizer())


@pytest.mark.parametrize("which", (0, 1))
def test_source_mutation_invalidates_an_existing_index(sources, which):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer())
    sources[which].write_text(sources[which].read_text() + "\n")
    with pytest.raises(ValueError, match="source changed"):
        dataset[0]


class MarginModel(torch.nn.Module):
    def __init__(self, dropout=0):
        super().__init__()
        self.table = torch.nn.Embedding(16, 8)
        self.dropout = torch.nn.Dropout(dropout)
        self.forward_sizes = []

    def forward(self, input_ids, **kwargs):
        assert "labels" not in kwargs
        assert kwargs["logits_to_keep"] == 1
        assert torch.all(input_ids[:, -1] >= 4)
        self.forward_sizes.append(input_ids.shape[0])
        return SimpleNamespace(logits=self.dropout(self.table(input_ids[:, -1]))[:, None, :])


def impression():
    labels = torch.tensor([1, 0, 1, 0, 0, 0, 0])
    targets = torch.where(labels == 1, 2, 3)
    inputs = torch.stack((torch.full((7,), 12), torch.arange(4, 11), targets), dim=1)
    supervision = torch.full_like(inputs, -100)
    supervision[:, -1] = targets
    return {"input_ids": inputs, "labels": supervision, "attention_mask": torch.ones_like(inputs),
            "label": labels, "impression_id": ["A"] * 7, "expected_candidates": 7}


def direct_loss(model, inputs, chunk_size):
    fields = ("input_ids", "labels", "attention_mask")
    ce_sums, logits = [], []
    for start in range(0, 7, chunk_size):
        chunk = {field: inputs[field][start:start + chunk_size] for field in fields}
        ce, output = candidate_decision_loss(model, chunk, (2, 3), loss_type="full_vocab")
        ce_sums.append(ce * chunk["input_ids"].shape[0])
        logits.append(output["logits"])
    logits = torch.cat(logits)
    return (torch.stack(ce_sums).sum() / 7
            + 0.5 * within_impression_bt(logits[:, 0] - logits[:, 1],
                                        inputs["label"], inputs["impression_id"]))


@pytest.mark.parametrize("chunk_size", (1, 2, 3, 7, 20))
def test_loss_and_all_candidate_gradients_match_a_full_impression_forward(chunk_size):
    torch.manual_seed(7)
    model = MarginModel()
    reference = copy.deepcopy(model)
    inputs = impression()
    expected = direct_loss(reference, inputs, 7)
    expected.backward()
    loss, outputs = complete_impression_loss(
        model, inputs, (2, 3), chunk_size=chunk_size, pairwise_weight=0.5)
    loss.backward()
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(model.table.weight.grad, reference.table.weight.grad,
                               atol=1e-6, rtol=1e-6)
    assert outputs["logits"].shape == (7, 2)
    assert max(model.forward_sizes) <= min(7, chunk_size)
    assert torch.all(model.table.weight.grad[4:11].abs().sum(-1) > 0)


def test_stochastic_recomputation_preserves_rng_and_matches_unchunked_graph_retention():
    torch.manual_seed(1)
    model = MarginModel(dropout=0.3)
    reference = copy.deepcopy(model)
    torch.manual_seed(37)
    expected = direct_loss(reference, impression(), 2)
    expected.backward()
    rng = torch.random.get_rng_state()
    torch.manual_seed(37)
    loss, _ = complete_impression_loss(
        model, impression(), (2, 3), chunk_size=2, pairwise_weight=0.5)
    loss.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(model.table.weight.grad, reference.table.weight.grad)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("change", ("partial", "mixed", "target", "mask"))
def test_invalid_or_partial_impressions_fail_before_any_candidate_forward(change):
    inputs = impression()
    if change == "partial":
        inputs["expected_candidates"] = 8
    elif change == "mixed":
        inputs["impression_id"][-1] = "B"
    elif change == "target":
        inputs["label"][-1] = 1
    else:
        inputs["attention_mask"][-1, -1] = 0
    model = MarginModel()
    with pytest.raises(ValueError):
        complete_impression_loss(model, inputs, (2, 3), chunk_size=2, pairwise_weight=0.5)
    assert model.forward_sizes == []


def test_one_class_impression_is_complete_ce_not_a_skipped_example():
    inputs = impression()
    inputs["label"].zero_()
    inputs["input_ids"][:, -1] = inputs["labels"][:, -1] = 3
    model = MarginModel()
    loss, _ = complete_impression_loss(
        model, inputs, (2, 3), chunk_size=2, pairwise_weight=0.5)
    expected = F.cross_entropy(model.table(inputs["input_ids"][:, -2]),
                              torch.full((7,), 3))
    torch.testing.assert_close(loss, expected)


def test_checkpointing_does_not_retain_full_candidate_hidden_activations():
    class ActivationModel(MarginModel):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Embedding(16, 1024)
            self.projection = torch.nn.Linear(1024, 8)

        def forward(self, input_ids, **kwargs):
            hidden = self.table(input_ids[:, -1]).relu()
            return SimpleNamespace(logits=self.projection(hidden)[:, None, :])

    retained = []

    def capture(tensor):
        if tensor.is_floating_point():
            retained.append(tensor.numel())
        return tensor

    model = ActivationModel()
    with torch.autograd.graph.saved_tensors_hooks(capture, lambda tensor: tensor):
        loss, _ = complete_impression_loss(
            model, impression(), (2, 3), chunk_size=2, pairwise_weight=0.5)
    assert max(retained) < 1024
    loss.backward()
    assert torch.isfinite(model.projection.weight.grad).all()


def tiny_qwen():
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        pad_token_id=0, eos_token_id=1))


def full_trainer(dataset, output, *, train_dataset=None):
    return CompleteImpressionTrainer(
        yes_token_id=2, no_token_id=3, model=tiny_qwen(),
        expected_source=dataset.source_manifest, candidate_chunk_size=1,
        train_dataset=dataset if train_dataset is None else train_dataset,
        data_collator=CompleteImpressionCollator(_Collator(Tokenizer())),
        args=TrainingArguments(
            output_dir=str(output), use_cpu=True, report_to="none",
            num_train_epochs=2, per_device_train_batch_size=1, remove_unused_columns=False,
            gradient_accumulation_steps=2, save_strategy="no", logging_strategy="no",
            disable_tqdm=True),
    )


def test_native_trainer_consumes_every_exposure_once_in_each_fixed_epoch(sources, tmp_path):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer(), max_len=4096)
    runner = full_trainer(dataset, tmp_path / "model")
    runner.train()
    assert runner.state.global_step == 2
    assert len(runner.epoch_coverage) == 2
    for record in runner.epoch_coverage:
        assert (record["impressions"], record["exposures"], record["positives"]) == (2, 5, 1)
        assert record["impression_ids_sha256"] == dataset.source_manifest["impression_ids_sha256"]
        assert record["optimizer_steps"] == 1


def test_incomplete_epoch_never_passes_source_coverage(sources, tmp_path):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer(), max_len=4096)
    runner = full_trainer(dataset, tmp_path / "model", train_dataset=[dataset[0]])
    with pytest.raises(ValueError, match="does not exactly cover"):
        runner.train()
    assert runner.epoch_coverage == []


def test_duplicate_source_consumption_fails_before_second_impression_forward(sources, tmp_path):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer(), max_len=4096)
    runner = full_trainer(dataset, tmp_path / "model", train_dataset=[dataset[0], dataset[0]])
    with pytest.raises(ValueError, match="more than once"):
        runner.train()
    assert runner.epoch_coverage == []


@pytest.mark.parametrize("override", [
    {"eval_behaviors_path": "dev.tsv"}, {"sample": 1}, {"neg_ratio": 4.},
    {"micro_batch_size": 2}, {"resume_from_checkpoint": "old"},
    {"deepspeed_config": "zero2.json"},
])
def test_unqualified_full_source_options_fail_before_weight_loading(tmp_path, override):
    arguments = {"full_impression_training": True, "micro_batch_size": 1,
                 "output_dir": str(tmp_path), **override}
    with patch("sft_mind_pointwise_pairwise.AutoModelForCausalLM.from_pretrained") as loader:
        with pytest.raises(ValueError, match="Complete-source mode requires"):
            train(**arguments)
        loader.assert_not_called()


def test_collator_does_not_merge_or_drop_complete_impressions(sources):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer(), max_len=4096)
    collator = CompleteImpressionCollator(_Collator(Tokenizer()))
    with pytest.raises(ValueError, match="batch_size=1"):
        collator([dataset[0], dataset[1]])
    partial = dataset[0]
    partial["candidates"].pop()
    with pytest.raises(ValueError, match="were dropped"):
        collator([partial])


def test_complete_source_encoding_never_silently_truncates_the_candidate_prompt(sources):
    dataset = MINDImpressionSFTDataset(*sources, Tokenizer(), max_len=2)
    with pytest.raises(ValueError, match="no silent truncation"):
        dataset[0]


def test_full_training_entrypoint_saves_only_after_complete_fixed_epoch_proofs(sources, tmp_path):
    behaviors, news = sources
    output = tmp_path / "full-entrypoint"
    model = tiny_qwen()

    def cpu_arguments(**kwargs):
        kwargs["use_cpu"] = True
        kwargs["bf16"] = False
        kwargs["disable_tqdm"] = True
        return TrainingArguments(**kwargs)

    with (patch("sft_mind_pointwise_pairwise.AutoTokenizer.from_pretrained", return_value=Tokenizer()),
          patch("sft_mind_pointwise_pairwise.AutoConfig.from_pretrained", return_value=model.config),
          patch("sft_mind_pointwise_pairwise.AutoModelForCausalLM.from_pretrained", return_value=model),
          patch("sft_mind_pointwise_pairwise.transformers.TrainingArguments", side_effect=cpu_arguments)):
        train(
            base_model="fixture-model", train_behaviors_path=str(behaviors),
            train_news_path=str(news), output_dir=str(output), full_impression_training=True,
            micro_batch_size=1, batch_size=2, candidate_chunk_size=1,
            num_epochs=2, cutoff_len=4096, learning_rate=0.001)
    config = json.loads((output / "final_checkpoint" / "config.json").read_text())
    assert config["mind_pairwise_objective"] == "complete_impression_causal_margin_bt_v1"
    assert config["mind_pairwise_answer_token_ids"] == [2, 3]
    assert config["mind_candidate_chunk_size"] == 1
    assert config["mind_training_source"]["exposures"] == 5
    assert len(config["mind_training_epoch_coverage"]) == 2
    assert all(record["exposures"] == 5 and record["impressions"] == 2
               for record in config["mind_training_epoch_coverage"])
