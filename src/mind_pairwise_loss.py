"""Candidate margins and exact within-impression loss across forward chunks."""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from jev_decision import candidate_decision_loss, decision_prompt_inputs


def validate_pairwise_metadata(labels, impressions, batch_size):
    if (not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.numel() != batch_size
            or not torch.all((labels == 0) | (labels == 1))
            or not isinstance(impressions, (list, tuple)) or len(impressions) != batch_size
            or any(not isinstance(impression, str) or not impression for impression in impressions)):
        raise ValueError("Pairwise metadata needs one binary label and impression ID per candidate")


def within_impression_bt(scores, labels, impressions):
    if scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all():
        raise ValueError("Pairwise scores must be a finite vector")
    validate_pairwise_metadata(labels, impressions, scores.numel())
    groups = {}
    for index, impression in enumerate(impressions):
        groups.setdefault(impression, []).append(index)
    labels = labels.to(scores.device)
    losses = []
    for indices in groups.values():
        indices = torch.tensor(indices, device=scores.device)
        positive = scores[indices[labels[indices] == 1]]
        negative = scores[indices[labels[indices] == 0]]
        if positive.numel() and negative.numel():
            losses.append(F.softplus(negative[None, :] - positive[:, None]).mean())
    return torch.stack(losses).mean() if losses else None


def complete_impression_loss(model, inputs, token_ids, *, chunk_size, pairwise_weight):
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("Candidate chunk size must be a positive integer")
    if not math.isfinite(pairwise_weight) or pairwise_weight < 0:
        raise ValueError("Pairwise weight must be finite and nonnegative")
    count = inputs["input_ids"].shape[0]
    validate_pairwise_metadata(inputs["label"], inputs["impression_id"], count)
    expected = inputs["expected_candidates"]
    if (not isinstance(expected, int) or isinstance(expected, bool) or expected != count
            or len(set(inputs["impression_id"])) != 1):
        raise ValueError("A complete-impression batch must retain exactly one source impression")
    model_inputs = {key: value for key, value in inputs.items()
                    if key not in ("label", "impression_id", "expected_candidates")}
    decision_prompt_inputs(model_inputs, token_ids)
    if not torch.all(model_inputs["labels"][:, -1] == torch.where(
            inputs["label"].to(model_inputs["labels"].device) == 1, token_ids[0], token_ids[1])):
        raise ValueError("Complete-impression labels disagree with their answer tokens")
    fields = tuple(model_inputs)

    def forward_chunk(*tensors):
        chunk = dict(zip(fields, tensors))
        ce, output = candidate_decision_loss(model, chunk, token_ids, loss_type="full_vocab")
        return ce * chunk["input_ids"].shape[0], output["logits"]

    ce_sums, logits = [], []
    for start in range(0, count, chunk_size):
        tensors = tuple(model_inputs[key][start:start + chunk_size] for key in fields)
        if torch.is_grad_enabled():
            ce, scores = checkpoint(
                forward_chunk, *tensors, use_reentrant=False, preserve_rng_state=True)
        else:
            ce, scores = forward_chunk(*tensors)
        ce_sums.append(ce)
        logits.append(scores)
    logits = torch.cat(logits)
    loss = torch.stack(ce_sums).sum() / count
    bt = within_impression_bt(
        logits[:, 0] - logits[:, 1], inputs["label"], inputs["impression_id"])
    if bt is not None:
        loss = loss + pairwise_weight * bt
    if not torch.isfinite(loss):
        raise FloatingPointError("Complete-impression loss contains NaN/Inf")
    return loss, {"loss": loss, "logits": logits}
