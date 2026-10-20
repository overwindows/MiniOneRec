"""Shared JEV prompt contract and full-depth candidate-only decision loss."""

import json

import torch
import torch.nn.functional as F


PROMPT_CONTRACTS = ("legacy", "native_click")
LOSS_TYPES = ("full_vocab", "decision")
LABELS = ("A", "B")
BINARY_CRITERIA = (
    {"id": "yes", "description": "The stated proposition is true."},
    {"id": "no", "description": "The stated proposition is false."},
)


def render_noul_prompt(state, prompt_contract="legacy"):
    if prompt_contract not in PROMPT_CONTRACTS:
        raise ValueError(f"Unknown JEV prompt contract: {prompt_contract}")
    if prompt_contract == "native_click":
        instructions = (
            "Judge the following proposition using the user's reading history "
            "and the candidate article: the user would click the candidate news "
            "article if it were shown. A no-click label does not necessarily "
            "mean the user dislikes the article."
        )
        descriptions = [candidate["description"] for candidate in BINARY_CRITERIA]
    else:
        instructions = (
            "Based on the user's reading history, decide whether they would "
            "read the candidate article. Answer yes if the article matches "
            "their interests, no otherwise."
        )
        descriptions = ["yes", "no"]
    task = {
        "primitive": "noul",
        "instructions": instructions,
        "criteria": [
            {"label": label, "description": description}
            for label, description in zip(LABELS, descriptions)
        ],
    }
    return (
        "Shared state:\n" + state + "\n\n"
        + json.dumps(task, ensure_ascii=False, sort_keys=True)
        + "\nReturn only the selected letter: A, B.\nAnswer:"
    )


def decision_token_ids(tokenizer):
    tokens = [tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
    if any(len(token) != 1 for token in tokens):
        raise ValueError("JEV A/B labels must each encode as one token")
    ids = [token[0] for token in tokens]
    if len(set(ids)) != 2 or any(token in tokenizer.all_special_ids for token in ids):
        raise ValueError("JEV A/B IDs must be distinct, non-special tokens")
    return tuple(ids)


def encode_decision_prompt(tokenizer, prompt, *, max_length=None, truncate=False):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(formatted, add_special_tokens=False)
    if not ids:
        raise ValueError("JEV decision prompt is empty")
    for label, token in zip(LABELS, decision_token_ids(tokenizer)):
        if tokenizer.encode(formatted + label, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"JEV {label} token is not stable at the answer boundary")
    if max_length is not None and len(ids) > max_length:
        if not truncate:
            raise ValueError(
                f"JEV prompt has {len(ids)} tokens, exceeding budget {max_length}; "
                "native prompts are never silently truncated"
            )
        ids = tokenizer.encode(
            formatted, max_length=max_length, truncation=True, add_special_tokens=False,
        )
    return ids


def candidate_decision_loss(model, inputs, token_ids):
    """Read only the final prompt logits; the appended target is never forwarded."""
    if len(token_ids) != 2 or len(set(token_ids)) != 2 or min(token_ids) < 0:
        raise ValueError("Decision loss requires two distinct nonnegative A/B IDs")
    ids, labels, mask = (inputs[key] for key in ("input_ids", "labels", "attention_mask"))
    if ids.ndim != 2 or ids.shape != labels.shape or ids.shape != mask.shape:
        raise ValueError("JEV inputs, labels and attention mask must share shape [B, L]")
    if ids.shape[0] == 0 or ids.shape[1] < 2:
        raise ValueError("JEV decision batch needs a nonempty prompt and answer")
    if not torch.all((labels != -100).sum(dim=1) == 1) or not torch.all(labels[:, -1] != -100):
        raise ValueError("JEV requires exactly one supervised answer at the final position")
    if not torch.all(ids[:, -1] == labels[:, -1]):
        raise ValueError("JEV appended answer must match its supervised label")
    if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[:, 1:] < mask[:, :-1]):
        raise ValueError("JEV decision loss requires a binary, left-padded attention mask")
    if not torch.all(mask[:, -2:] == 1):
        raise ValueError("JEV prompt and answer must both be unmasked")
    yes_id, no_id = token_ids
    if not torch.all((labels[:, -1] == yes_id) | (labels[:, -1] == no_id)):
        raise ValueError("JEV decision targets must be A or B")
    unexpected = set(inputs) - {"input_ids", "labels", "attention_mask", "position_ids"}
    if unexpected:
        raise ValueError(f"Unsupported JEV decision inputs: {sorted(unexpected)}")
    prompt_mask = mask[:, :-1]
    positions = prompt_mask.long().cumsum(dim=-1) - 1
    positions.masked_fill_(prompt_mask == 0, 0)
    if "position_ids" in inputs:
        if inputs["position_ids"].shape != ids.shape:
            raise ValueError("JEV position IDs must match the input shape")
        positions = inputs["position_ids"][:, :-1]
    output = model(
        input_ids=ids[:, :-1],
        attention_mask=prompt_mask,
        position_ids=positions,
        use_cache=False,
        return_dict=True,
        logits_to_keep=1,
    )
    if output.logits.ndim != 3 or output.logits.shape[:2] != (ids.shape[0], 1):
        raise ValueError("JEV model must return exactly one logit position per prompt")
    if max(token_ids) >= output.logits.shape[-1]:
        raise ValueError("JEV decision token ID exceeds the model vocabulary")
    logits = output.logits[:, 0, [yes_id, no_id]].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("JEV A/B decision logits contain NaN/Inf")
    targets = (labels[:, -1] == yes_id).float()
    loss = F.binary_cross_entropy_with_logits(logits[:, 0] - logits[:, 1], targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("JEV candidate decision loss contains NaN/Inf")
    return loss, {"loss": loss, "logits": logits}
