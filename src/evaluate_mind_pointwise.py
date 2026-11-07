"""
Evaluate MIND models trained with point-wise SFT (Yes/No classification).

This evaluation scores each candidate independently by computing P("Yes" | prompt)
and uses those scores to rank candidates within each impression.

Key Features:
- Point-wise scoring: Each candidate evaluated independently
- Scores P(" Yes") vs P(" No") for each (history, candidate) pair
- Ranks candidates by Yes probability
- Uses official MIND metrics (AUC, MRR, nDCG@5, nDCG@10)
- Supports Flash Attention 2 for faster inference

Usage:
    python evaluate_mind_pointwise.py \
        --model_path output_dir/sft_mind_pointwise_*/final_checkpoint \
        --behaviors_path ../data/MIND/dev/behaviors.tsv \
        --news_path ../data/MIND/dev/news.tsv \
        --flash_attn \
        --max_impressions 1000  # Optional: for quick testing

    # Quick mode (500 impressions for fast iteration)
    python evaluate_mind_pointwise.py --model_path ... --quick
"""

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from mind_utils import (
    load_news,
    build_pointwise_prompt,
    build_pointwise_prompt_subcategory,
    parse_behaviors_line,
    auc_score,
    mrr_score,
    ndcg_score,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _common_prefix_len(seqs: List[List[int]]) -> int:
    first = seqs[0]
    n = min(len(s) for s in seqs)
    for s in seqs[1:]:
        i = 0
        while i < n and s[i] == first[i]:
            i += 1
        n = i
    return n


def _yes_no_scores(model, last_hidden, yes_token_id, no_token_id, temperature, legacy_bf16):
    """Score = logP(" Yes") - logP(" No") at the answer position, shape (B,).

    The softmax normaliser cancels, so this is (logit_yes - logit_no) / T and
    only two lm_head rows are needed. They are applied in fp32: bf16 logits
    near |20| have a 0.125 step, which puts ~57% of candidates into within-
    impression ties (AUC gives ties half credit, discarding ranking signal).
    legacy_bf16 reproduces the old full-vocab bf16 computation for comparison
    with earlier results.
    """
    lm_head = model.get_output_embeddings()
    if legacy_bf16:
        logits = lm_head(last_hidden)
        if temperature != 1.0:
            logits = logits / temperature
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[:, yes_token_id] - log_probs[:, no_token_id]
    w = lm_head.weight[[yes_token_id, no_token_id]].float()  # (2, H)
    logits = last_hidden.float() @ w.T
    if getattr(lm_head, "bias", None) is not None:
        logits = logits + lm_head.bias[[yes_token_id, no_token_id]].float()
    return (logits[:, 0] - logits[:, 1]) / temperature


def _score_with_prefix_cache(model, all_prompt_ids, device, yes_token_id, no_token_id,
                             pack_tokens, temperature, legacy_bf16):
    """Shared-prefix scoring: identical tokens/positions to the plain path.

    The prefix is the longest common token prefix of all prompts (computed on
    the already-tokenized prompts, so no tokenizer boundary effects), capped so
    every prompt keeps >= 1 suffix token. It is run once (batch 1); the
    candidate suffixes are then packed back to back into one batch-1 sequence
    behind it, with a block mask so each suffix attends to the prefix and
    causally to itself only, and position ids restarting at len(prefix) for
    each suffix. Packing keeps the prefix KV at batch 1 (expanding it per
    candidate made the KV concat ~40% of GPU time). Suffixes are packed in
    chunks of <= pack_tokens tokens; the cache is cropped back to the prefix
    after each chunk.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    p = min(_common_prefix_len(all_prompt_ids), min(len(s) for s in all_prompt_ids) - 1)
    decoder = model.get_decoder()
    suffixes = [ids[p:] for ids in all_prompt_ids]

    chunks, cur, cur_len = [], [], 0
    for sfx in suffixes:
        if cur and cur_len + len(sfx) > pack_tokens:
            chunks.append(cur)
            cur, cur_len = [], 0
        cur.append(sfx)
        cur_len += len(sfx)
    chunks.append(cur)

    # cuDNN SDPA rebuilds an execution plan for every new shape (~1 ms CPU per
    # call here, since shapes change every impression); flash/mem-efficient don't.
    with torch.no_grad(), sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        cache = None
        if p > 0:
            prefix = torch.tensor([all_prompt_ids[0][:p]], dtype=torch.long, device=device)
            cache = decoder(input_ids=prefix, use_cache=True).past_key_values

        scores = []
        for chunk in chunks:
            lens = torch.tensor([len(sfx) for sfx in chunk], device=device)
            n = int(lens.sum())
            seg = torch.repeat_interleave(torch.arange(len(chunk), device=device), lens)
            starts = torch.cumsum(lens, 0) - lens
            offset = torch.arange(n, device=device) - starts[seg]
            idx = torch.arange(n, device=device)
            own = (seg[:, None] == seg[None, :]) & (idx[None, :] <= idx[:, None])
            mask = torch.cat([own.new_ones(n, p), own], dim=1)[None, None]  # (1,1,n,p+n), True = attend
            input_ids = torch.tensor([[t for sfx in chunk for t in sfx]], dtype=torch.long, device=device)
            hidden = decoder(
                input_ids=input_ids,
                position_ids=(offset + p)[None],
                attention_mask=mask,
                past_key_values=cache,
                use_cache=cache is not None,
            ).last_hidden_state[0]
            if cache is not None:
                cache.crop(p)
            last = hidden[starts + lens - 1]  # (len(chunk), H)
            scores.extend(_yes_no_scores(
                model, last, yes_token_id, no_token_id, temperature, legacy_bf16).cpu().tolist())
    return scores


_LINEAR_STATE_FP32_PATCHED = False


def _patch_recurrent_state_fp32():
    """Make the linear-attention cache keep the recurrent state in fp32.

    The chunked delta-rule kernel computes the state in fp32 and returns an
    fp32 final state, but `LinearAttentionLayer.lazy_initialization` sizes the
    recurrent-state buffer from `self.dtype`, which was set from the CONV
    state (bf16), so `update_recurrent_state`'s `copy_` rounds it to bf16.

    In the plain path no cache is involved, so the state stays fp32 end to
    end. In the prefix path the prefix's state is written to the cache and
    read back as the suffix's `initial_state`, so without this the ~1100-token
    prefix would be rounded to bf16 (0.4% relative error) before it reaches
    the suffix — and that state feeds the output directly.

    The patch must be installed BEFORE the prefix forward, otherwise the
    prefix's own state has already been rounded on the way into the cache.

    The conv state stays bf16: it is a 4-token sliding window of projections
    (not an accumulated sum), and the plain path computes it in bf16 too.
    """
    global _LINEAR_STATE_FP32_PATCHED
    if _LINEAR_STATE_FP32_PATCHED:
        return
    from transformers.cache_utils import LinearAttentionAndFullAttentionLayer, LinearAttentionLayer

    def _update_recurrent_state(self, recurrent_states, **kwargs):
        if not self.is_recurrent_states_initialized:
            self.lazy_initialization(recurrent_states=recurrent_states)
        # Assign rather than copy_: the buffer's dtype is bf16 (inherited from
        # the conv state), and `copy_` would round the fp32 state on the way in.
        self.recurrent_states = recurrent_states.to(torch.float32).clone()
        return self.recurrent_states

    LinearAttentionLayer.update_recurrent_state = _update_recurrent_state
    # `LinearAttentionAndFullAttentionLayer` (the "hybrid" layer type) defines
    # its OWN `update_recurrent_state`, so patching only the base class would
    # leave those layers rounding to bf16. Patch both.
    if hasattr(LinearAttentionAndFullAttentionLayer, "update_recurrent_state"):
        LinearAttentionAndFullAttentionLayer.update_recurrent_state = _update_recurrent_state
    _LINEAR_STATE_FP32_PATCHED = True


def _expand_cache_batch(cache, batch_size):
    """Replicate a batch-1 cache across the batch dimension, in place.

    Returns a snapshot that `_restore_cache_batch` puts back.

    Linear-attention layers keep a conv state and a recurrent state instead of
    a KV tensor, and their `crop()` is a no-op, so they are expanded here
    explicitly.

    The full-attention KV is expanded as a VIEW: `DynamicLayer.update` cats
    the new keys onto it, which materialises a fresh tensor anyway, so copying
    the prefix KV up front would only double the peak memory. The linear
    states must be real copies, because `update_conv_state` /
    `update_recurrent_state` write into them with `copy_`, which is undefined
    on a stride-0 expanded view.
    """
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    saved = []
    for layer in cache.layers:
        linear = isinstance(layer, LinearAttentionCacheLayerMixin)
        conv = getattr(layer, "conv_states", None) if linear else None
        rec = getattr(layer, "recurrent_states", None) if linear else None
        # A "hybrid" layer carries BOTH a linear state and a full-attention KV.
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None and keys.dim() != 4:
            keys = values = None
        saved.append((layer, conv, rec, keys, values))

        if conv is not None:
            layer.conv_states = conv.expand(batch_size, *([-1] * (conv.dim() - 1))).contiguous()
        if rec is not None:
            layer.recurrent_states = rec.expand(batch_size, *([-1] * (rec.dim() - 1))).contiguous()
        if keys is not None:
            layer.keys = keys.expand(batch_size, *([-1] * (keys.dim() - 1)))
        if values is not None:
            layer.values = values.expand(batch_size, *([-1] * (values.dim() - 1)))
    return saved


def _restore_cache_batch(cache, saved):
    """Undo `_expand_cache_batch`, so the next group starts from the prefix state.

    The linear states were written in place (`copy_` into the expanded copy),
    and the full-attention KV was replaced by a fresh concatenated tensor, so
    restoring the saved references puts the cache back to the batch-1 prefix.
    """
    for layer, conv, rec, keys, values in saved:
        if conv is not None:
            layer.conv_states = conv
        if rec is not None:
            layer.recurrent_states = rec
        if keys is not None:
            layer.keys = keys
        if values is not None:
            layer.values = values


def _score_with_prefix_cache_hybrid(model, all_prompt_ids, device, yes_token_id, no_token_id,
                                     pad_token_id, group_size, temperature, legacy_bf16):
    """Shared-prefix scoring for hybrid (GatedDeltaNet + full attention) backbones.

    The full-attention path packs the candidate suffixes into ONE sequence and
    masks them apart. That is impossible here: the linear-attention layers
    carry a causal conv state and a recurrent state, both sequential over the
    token axis, so a packed sequence would let one candidate's suffix bleed
    into the next.

    Instead each candidate's suffix is its own batch row. The prefix is still
    run once (batch 1) and its states are replicated across the batch:
      * full-attention layers: the prefix KV is expanded to (G, H, P, D);
      * linear-attention layers: `conv_states` (G, conv_dim, K) and
        `recurrent_states` (G, HV, K, V) are expanded to match.
    Every row therefore sees exactly the prefix it would have seen in the plain
    path, and the suffix is causally masked within its own row.

    The suffix is RIGHT-padded. The linear-attention recurrence is exactly
    causal (a position's state depends only on the tokens at or before it), so
    trailing pads cannot affect the hidden state we read back (each row's last
    real token). Left padding would instead push the pads in FRONT of the real
    tokens, where the recurrence would consume them first.

    The cache is restored from the batch-1 snapshot before every group, so a
    group's padding tokens can never leak into the next group's conv or
    recurrent state.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    p = max(min(_common_prefix_len(all_prompt_ids),
                min(len(s) for s in all_prompt_ids) - 1), 0)
    decoder = model.get_decoder()

    # cuDNN SDPA rebuilds an execution plan for every new shape (~1 ms CPU per
    # call here, since shapes change every impression); flash/mem-efficient don't.
    with torch.no_grad(), sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        cache = None
        if p > 0:
            # Patch the cache layer CLASS before the prefix forward, so the
            # prefix's own recurrent state is stored in fp32 (the kernel's
            # native precision) instead of being rounded to bf16 on the way in.
            _patch_recurrent_state_fp32()
            prefix = torch.tensor([all_prompt_ids[0][:p]], dtype=torch.long, device=device)
            cache = decoder(input_ids=prefix, use_cache=True).past_key_values

        scores = []
        for start in range(0, len(all_prompt_ids), group_size):
            group = all_prompt_ids[start:start + group_size]
            suffixes = [ids[p:] for ids in group]
            lens = [len(s) for s in suffixes]
            L = max(lens)
            G = len(group)

            rows = [list(sfx) + [pad_token_id] * (L - len(sfx)) for sfx in suffixes]
            input_ids = torch.tensor(rows, dtype=torch.long, device=device)
            # The suffix continues the prefix's positions, exactly as in the
            # plain path (where the whole prompt is one contiguous sequence).
            position_ids = (torch.arange(L, device=device) + p).expand(G, L).contiguous()

            if p > 0:
                # 4-D block mask: every row attends to the whole prefix, and to
                # the suffix tokens at or before its own position.
                mask = (torch.arange(p + L, device=device) - p)[None, :] <= torch.arange(L, device=device)[:, None]
                mask = mask[None, None].expand(G, 1, L, p + L)
                saved = _expand_cache_batch(cache, G)
                hidden = decoder(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=mask,
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state
                _restore_cache_batch(cache, saved)
            else:
                # No shared prefix (the prompts disagree from the first token):
                # score the group as plain right-padded full prompts.
                mask = torch.ones(L, L, dtype=torch.bool, device=device).tril()
                mask = mask[None, None].expand(G, 1, L, L)
                hidden = decoder(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=mask,
                    use_cache=False,
                ).last_hidden_state

            last = hidden[torch.arange(G, device=device), torch.tensor(lens, device=device) - 1]
            scores.extend(_yes_no_scores(
                model, last, yes_token_id, no_token_id, temperature, legacy_bf16).cpu().tolist())
    return scores


def _build_parity_prompt_ids(
    tokenizer, history, candidate, max_prompt,
    use_chat_template=False, use_subcategory=False, use_recency=False,
    use_profile_summary=False, impression_timestamp=None, history_format="flat",
):
    """Token ids for ONE (history, candidate) pair, shrunk exactly like training.

    Training (data.py `encode_example`) renders the prompt, and if it does not
    fit `max_len - len(target_ids)` it drops the OLDEST ~1/8 of the history and
    re-renders, until it fits or the history is empty; if it still overflows
    (giant abstract) it keeps the TAIL so the candidate block, the question and
    the answer survive.

    Eval never did this: it scored the FULL history. So every impression whose
    prompt overflows 2048 tokens was scored on a prompt shape the model had
    never seen in training.

    The shrink is PER CANDIDATE (each training example has one candidate), so
    the candidates of a single impression can end up with different history
    lengths. That is reproduced here rather than approximated per impression.

    Returns:
        List[int]: the prompt token ids (no answer token).
    """
    h = list(history)
    while True:
        if use_subcategory:
            prompt = build_pointwise_prompt_subcategory(
                h, candidate, tokenizer=tokenizer, use_chat_template=use_chat_template)
        else:
            prompt = build_pointwise_prompt(
                h, candidate, tokenizer=tokenizer, use_chat_template=use_chat_template,
                use_recency=use_recency, use_profile_summary=use_profile_summary,
                impression_timestamp=impression_timestamp, history_format=history_format,
            )
        ids = tokenizer.encode(prompt, add_special_tokens=(not use_chat_template))
        if len(ids) <= max_prompt or not h:
            break
        # Too long: drop the oldest ~1/8 of the history and re-render.
        h = h[max(1, len(h) // 8):]

    if len(ids) > max_prompt:
        # Even with no history the prompt overflows: keep the TAIL so the
        # candidate + question survive (same as training).
        ids = ids[-max_prompt:]
    return ids


def _build_parity_prompt_ids_batch(
    tokenizer, history, candidates, max_prompt,
    use_chat_template=False, use_subcategory=False, use_recency=False,
    use_profile_summary=False, impression_timestamp=None, history_format="flat",
):
    """Batched version of `_build_parity_prompt_ids` over all candidates.

    The overwhelming majority of impressions fit in `max_prompt` on the first
    render, so those are encoded in ONE batched tokenizer call (the Rust fast
    tokenizer parallelises that; one call per candidate does not). Only the
    candidates that actually overflow are re-rendered through the shrink loop.

    The result is identical to calling `_build_parity_prompt_ids` per candidate:
    the first render is the same string, and the shrink loop only runs for
    candidates that did not fit.
    """
    def _render(h, cand):
        if use_subcategory:
            return build_pointwise_prompt_subcategory(
                h, cand, tokenizer=tokenizer, use_chat_template=use_chat_template)
        return build_pointwise_prompt(
            h, cand, tokenizer=tokenizer, use_chat_template=use_chat_template,
            use_recency=use_recency, use_profile_summary=use_profile_summary,
            impression_timestamp=impression_timestamp, history_format=history_format,
        )

    prompts = [_render(history, cand) for cand in candidates]
    all_ids = tokenizer(prompts, add_special_tokens=(not use_chat_template))["input_ids"]

    # Re-render only the candidates that overflowed, through the shrink loop.
    for i, ids in enumerate(all_ids):
        if len(ids) <= max_prompt:
            continue
        all_ids[i] = _build_parity_prompt_ids(
            tokenizer, history, candidates[i], max_prompt,
            use_chat_template=use_chat_template, use_subcategory=use_subcategory,
            use_recency=use_recency, use_profile_summary=use_profile_summary,
            impression_timestamp=impression_timestamp, history_format=history_format,
        )
    return all_ids


def batch_score_candidates_pointwise(
    model,
    tokenizer,
    history: List[dict],
    candidates: List[dict],
    device,
    yes_token_id: int,
    no_token_id: int,
    batch_size: int = 8,
    use_chat_template: bool = False,
    use_subcategory: bool = False,
    temperature: float = 1.0,
    use_recency: bool = False,
    use_profile_summary: bool = False,
    impression_timestamp: str = None,
    history_format: str = "flat",
    prefix_cache: bool = False,
    pack_tokens: int = 8192,
    legacy_bf16_scores: bool = False,
    prefix_cache_mode: str = "packed",
    prefix_group_size: int = 16,
    right_pad: bool = False,
    prompt_parity: bool = False,
    parity_max_len: int = 2048,
    answer_len: int = 1,
) -> List[float]:
    """
    Score multiple candidates in batches for efficiency.

    With prefix_cache, the token prefix shared by every candidate's prompt
    (system + history, ~all of the prompt) is run once and its KV cache is
    reused for the packed per-candidate suffixes.

    Returns:
        List of scores for each candidate
    """
    # Build all prompts
    if prompt_parity:
        # Train/eval parity: reproduce data.py `encode_example`'s shrink loop
        # PER CANDIDATE, so each candidate is scored on the same prompt shape
        # the model was trained on. Per candidate (not per impression), because
        # each training example has exactly one candidate.
        max_prompt = parity_max_len - answer_len
        all_prompt_ids = _build_parity_prompt_ids_batch(
            tokenizer, history, candidates, max_prompt,
            use_chat_template=use_chat_template, use_subcategory=use_subcategory,
            use_recency=use_recency, use_profile_summary=use_profile_summary,
            impression_timestamp=impression_timestamp, history_format=history_format,
        )
    else:
        if use_subcategory:
            prompts = [
                build_pointwise_prompt_subcategory(history, cand, tokenizer=tokenizer, use_chat_template=use_chat_template)
                for cand in candidates
            ]
        else:
            prompts = [
                build_pointwise_prompt(
                    history, cand, tokenizer=tokenizer, use_chat_template=use_chat_template,
                    use_recency=use_recency, use_profile_summary=use_profile_summary,
                    impression_timestamp=impression_timestamp, history_format=history_format,
                )
                for cand in candidates
            ]

        # Tokenize all prompts (batched call -> parallel Rust fast tokenizer)
        # For chat templates, add_special_tokens is already handled
        all_prompt_ids = tokenizer(prompts, add_special_tokens=(not use_chat_template))["input_ids"]

    if prefix_cache:
        if prefix_cache_mode == "hybrid":
            return _score_with_prefix_cache_hybrid(
                model, all_prompt_ids, device, yes_token_id, no_token_id,
                tokenizer.pad_token_id, prefix_group_size, temperature, legacy_bf16_scores,
            )
        return _score_with_prefix_cache(
            model, all_prompt_ids, device, yes_token_id, no_token_id,
            pack_tokens, temperature, legacy_bf16_scores,
        )

    scores = []

    # Process in batches
    for batch_start in range(0, len(all_prompt_ids), batch_size):
        batch_ids = all_prompt_ids[batch_start:batch_start + batch_size]

        # Pad to same length
        max_len = max(len(ids) for ids in batch_ids)
        padded_ids = []
        attention_masks = []

        for ids in batch_ids:
            pad_len = max_len - len(ids)
            if right_pad:
                # RIGHT padding: the linear-attention recurrence is causal, so
                # trailing pads are never consumed by the positions we read
                # back (each row's last real token). LEFT padding instead puts
                # the pads BEFORE the real tokens, where the recurrence
                # consumes them first and they shift the whole history.
                padded_ids.append(list(ids) + [tokenizer.pad_token_id] * pad_len)
                attention_masks.append([1] * len(ids) + [0] * pad_len)
            else:
                # Left padding
                padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            if right_pad:
                hidden = model.get_decoder()(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                # The answer position is each row's last REAL token.
                last_idx = torch.tensor([len(ids) - 1 for ids in batch_ids], device=device)
                hidden = hidden[torch.arange(len(batch_ids), device=device), last_idx]
            else:
                # Left-padded, so the answer position (after "Answer:") is the last
                # column; lm_head is applied there only (see _yes_no_scores).
                hidden = model.get_decoder()(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state[:, -1, :]
            batch_scores = _yes_no_scores(
                model, hidden, yes_token_id, no_token_id, temperature, legacy_bf16_scores)
            scores.extend(batch_scores.cpu().tolist())

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--use_subcategory", action="store_true", help="Use [category/subcategory] format in prompts")
    parser.add_argument("--max_history", type=int, default=0, help="Max history items (0=unlimited)")
    parser.add_argument("--history_selection", choices=["recent", "relevant", "dedup_backfill", "dedup_only", "catcap"], default="recent",
                        help="recent = last N; relevant = category-matched; dedup_backfill/dedup_only = drop repeated IDs; catcap = per-category cap (must match training)")
    parser.add_argument("--history_format", choices=["flat", "grouped_profile", "two_block", "title_only"], default="flat",
                        help="History rendering format (must match training)")
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--prompt_parity", action="store_true",
                        help="Reproduce training's prompt-shrink loop per candidate: if the rendered "
                             "prompt exceeds --parity_max_len, drop the oldest ~1/8 of the history and "
                             "re-render (exactly data.py encode_example). Off by default because it "
                             "changes the scores of every over-length impression.")
    parser.add_argument("--parity_max_len", type=int, default=2048,
                        help="Training cutoff_len the parity path reproduces (default 2048)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for scoring candidates")
    parser.add_argument("--no_prefix_cache", action="store_true",
                        help="Disable shared-prefix KV caching (score every full prompt independently)")
    parser.add_argument("--legacy_bf16_scores", action="store_true",
                        help="Old bf16 full-vocab scoring (many tied scores, lower AUC); only for comparing with earlier results")
    parser.add_argument("--pack_tokens", type=int, default=8192,
                        help="Max candidate-suffix tokens packed into one forward when prefix caching is on")
    parser.add_argument("--prefix_cache_mode", choices=["auto", "packed", "hybrid"], default="auto",
                        help="auto = pick from the backbone's layer types (hybrid for GatedDeltaNet/hybrid "
                             "backbones, packed for full-attention ones); packed = pack candidate suffixes "
                             "into one sequence (full-attention backbones); hybrid = one batch row per "
                             "candidate (GatedDeltaNet/hybrid backbones)")
    parser.add_argument("--prefix_group_size", type=int, default=16,                        help="Candidates per forward in hybrid prefix-cache mode")
    parser.add_argument("--right_pad", action="store_true",
                        help="Right-pad the plain path instead of left-padding it. Diagnostic only: the linear-"
                             "attention layers zero the pad hidden states (apply_mask_to_padding_states), so the "
                             "pads contribute nothing to the recurrence either way. Used to confirm that the "
                             "hybrid prefix path and the plain path agree.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", help="Output prediction file for MIND leaderboard")
    parser.add_argument("--output_scores_file", help="Output raw scores file for ensemble (impression_id score1 score2 ...)")
    parser.add_argument("--flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--use_chat_template", action="store_true", help="Use chat template (for instruct models)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: evaluate 500 impressions")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for logit scaling (default=1.0, no effect); mainly useful for ensemble calibration")
    parser.add_argument("--use_recency", action="store_true", help="Mark 5 most recent history items with '(recent)' tag")
    parser.add_argument("--use_profile_summary", action="store_true", help="Prepend top-3 category interest summary to prompt")
    parser.add_argument("--use_impression_timestamp", action="store_true", help="Add day/time-of-day context from impression timestamp (must match training)")
    parser.add_argument("--use_entity", action="store_true", help="Append Wikidata entity labels ('Topics: ...') to article text (must match training)")
    parser.add_argument("--cf_scores_file", default="", help="TSV file with CF scores (impression_id, news_id, cf_score) to blend with LLM scores")
    parser.add_argument("--cf_alpha", type=float, default=0.3, help="Blend weight for CF: final = (1-alpha)*LLM + alpha*CF (default 0.3)")
    args = parser.parse_args()

    # Env-var override for the parity switch, so the shell wrappers that are
    # CURRENTLY RUNNING under six test-scoring jobs do not have to be edited
    # (editing a script a running job is executing corrupts its byte offsets).
    # MINIONEREC_PROMPT_PARITY=1 turns the train/eval parity path on.
    _env_parity = os.environ.get("MINIONEREC_PROMPT_PARITY", "").strip().lower()
    if _env_parity in ("1", "true", "yes", "on"):
        args.prompt_parity = True
    _env_parity_len = os.environ.get("MINIONEREC_PARITY_MAX_LEN", "").strip()
    if _env_parity_len.isdigit():
        args.parity_max_len = int(_env_parity_len)

    # Quick mode overrides max_impressions
    if args.quick and args.max_impressions == 0:
        args.max_impressions = 500

    set_seed(args.seed)

    # Load CF scores if provided
    cf_scores: dict = {}
    if args.cf_scores_file:
        print(f"Loading CF scores from: {args.cf_scores_file}")
        with open(args.cf_scores_file, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    imp_id, news_id, score = parts
                    cf_scores[(imp_id, news_id)] = float(score)
        print(f"Loaded {len(cf_scores):,} CF scores (alpha={args.cf_alpha})")

    print(f"Loading news from: {args.news_path}")
    news = load_news(args.news_path, args.use_abstract, use_entity=args.use_entity)
    print(f"Loaded {len(news)} news articles")

    print(f"Loading model from: {args.model_path}")
    import os as _os
    from pathlib import Path as _Path
    import time as _time
    # Cold fuse mounts (RW_MOUNT on Singularity) can take a while to populate
    # deep subpaths. isdir() returning False on first access makes HF treat the
    # absolute path as a repo id ("Repo id must be in the form ..."), a flaky
    # mount-warmup race. Wait until the checkpoint dir + safetensors are visible.
    # Single-shard checkpoints carry a "model.safetensors"; sharded dirs carry a
    # "model.safetensors.index.json" + model-000NN-of-MMMM.safetensors files.
    # Accept either as the mount-ready marker so multi-shard checkpoints (e.g.
    # the Qwen3-8B large, 4 shards) are not mis-flagged as a cold-mount race.
    for _attempt in range(60):
        mp = _Path(args.model_path)
        # LoRA Trainer checkpoints carry only adapter_model.safetensors.
        if mp.is_dir() and (
            (mp / "model.safetensors").is_file()
            or (mp / "model.safetensors.index.json").is_file()
            or (mp / "adapter_model.safetensors").is_file()
        ):
            break
        print(f"  mount warm-up: {mp} not visible yet (attempt {_attempt+1}/60)...",
              flush=True)
        _time.sleep(10)
    # Treat any absolute path or existing directory as local to avoid
    # huggingface_hub validate_repo_id rejecting absolute paths
    local_only = _os.path.isabs(args.model_path) or _os.path.isdir(args.model_path)
    model_path_arg = _Path(args.model_path) if local_only else args.model_path
    # Checkpoint subdirs may not contain tokenizer files; fall back to parent dir
    tokenizer_path = model_path_arg
    if local_only and not (_Path(model_path_arg) / "tokenizer_config.json").exists():
        parent = _Path(model_path_arg).parent
        if (parent / "tokenizer_config.json").exists():
            print(f"Tokenizer files not found in checkpoint dir, loading from parent: {parent}")
            tokenizer_path = parent

    # Defensive: some checkpoint snapshots carry an empty/corrupt
    # special_tokens_map.json (0 bytes), which makes transformers
    # json.load crash at tokenizer load time even though the actual token
    # definitions live in tokenizer_config.json / added_tokens_decoder.json.
    # Replace any unparseable special_tokens_map.json with a valid empty {}
    # (semantically a no-op) so the tokenizer loads.
    stm = _Path(tokenizer_path) / "special_tokens_map.json"
    if stm.exists():
        try:
            with open(stm, "r", encoding="utf-8") as _f:
                json.load(_f)
        except Exception:
            print(f"  Corrupt special_tokens_map.json; rewriting as empty {{}}")
            with open(stm, "w", encoding="utf-8") as _f:
                _f.write("{}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=local_only)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load model with optional Flash Attention 2
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if local_only:
        model_kwargs["local_files_only"] = True
    # flash_attention_2 is the fast path for dense causal LMs (e.g. Qwen3).
    # The hybrid Qwen3.5 text backbone (GatedDeltaNet linear attention + full
    # attention) cannot dispatch through FA2, so force sdpa for those so that
    # the pure-text Qwen3_5ForCausalLM loads and scores correctly (vision tower
    # is auto-ignored — MIND needs no multimodality).
    import transformers as _tf
    # PEFT adapter checkpoints (LoRA) carry no config.json; the base model is
    # named in adapter_config.json. AutoModelForCausalLM resolves the base and
    # applies the adapter automatically when given the adapter dir — keep
    # model_path_arg as-is for the load, and only point the config sniff at
    # the base model.
    _cfg_path = model_path_arg
    if not (_Path(model_path_arg) / "config.json").is_file():
        _ac = _Path(model_path_arg) / "adapter_config.json"
        if _ac.is_file():
            import json as _json
            _base = _json.load(open(_ac)).get("base_model_name_or_path")
            if _base:
                print(f"PEFT adapter checkpoint; base model: {_base}")
                _cfg_path = _Path(_base) if _os.path.isabs(_base) else _base
    _cfg = _tf.AutoConfig.from_pretrained(_cfg_path)
    _arch = " ".join(_cfg.architectures or [])
    # Match the trainer's detection (sft_mind_pointwise_ds.py): the hybrid Qwen3.5
    # GatedDeltaNet backbone declares multimodal architectures / ForConditionalGeneration
    # but loads as pure-text Qwen3_5ForCausalLM, which FA2 cannot dispatch. Relying on
    # model_type == "qwen3_5" is unreliable, so sniff the architecture string instead.
    _is_qwen35 = "ForConditionalGeneration" in _arch or "Qwen3_5" in _arch
    if _is_qwen35:
        model_kwargs["attn_implementation"] = "sdpa"
        print("Detected Qwen3.5 backbone: using sdpa attention (vision tower not loaded)")
    elif args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2")

    model = AutoModelForCausalLM.from_pretrained(
        model_path_arg, **model_kwargs
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")
    # device_map="auto" silently falls back to CPU when the target GPU is
    # unhealthy (a node with a faulted GPU reports 0 MiB used and the model
    # lands in RAM). Scoring a 9B model on CPU takes days, so fail fast
    # instead of hanging the whole multi-GPU job on one bad device.
    if device.type == "cpu" and _os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError(
            f"Model loaded on CPU but CUDA_VISIBLE_DEVICES="
            f"{_os.environ.get('CUDA_VISIBLE_DEVICES')}. The target GPU is "
            "likely unhealthy -- aborting instead of scoring on CPU."
        )

    # Prefix caching assumes plain full-attention KV layers for the PACKED
    # path (it packs the candidate suffixes into one sequence). Hybrid
    # backbones (Qwen3.5 GatedDeltaNet) keep a causal conv state and a
    # recurrent state instead, which cannot be packed; they use the "hybrid"
    # mode, which runs one batch row per candidate and replicates the prefix
    # state across the batch.
    # It also requires SDPA: the packed path feeds a custom 4-D bool block
    # mask, which flash_attention_2 (and eager's additive-float mask path)
    # does not honor — packed candidates would attend to each other.
    _layer_types = set(getattr(model.config.get_text_config(), "layer_types", None) or ["full_attention"])
    _attn_impl = str(getattr(model.config, "_attn_implementation", "sdpa"))
    _has_linear = "linear_attention" in _layer_types
    # "auto" picks the mode the backbone can actually run: the packed path
    # needs plain full-attention KV layers, so a GatedDeltaNet backbone gets
    # the hybrid (one batch row per candidate) path. Resolving it here keeps
    # the caller from having to know the backbone's internals.
    _mode = args.prefix_cache_mode
    if _mode == "auto":
        _mode = "hybrid" if _has_linear else "packed"
        print(f"Prefix cache mode 'auto' -> {_mode} "
              f"(layer types: {sorted(_layer_types)})")
    if _has_linear and _mode != "hybrid":
        print("Prefix KV cache disabled: this backbone has linear-attention "
              "(GatedDeltaNet) layers, which the packed path cannot use "
              "(set PREFIX_CACHE_MODE=hybrid to use the per-candidate path)")
        prefix_cache = False
    else:
        prefix_cache = (not args.no_prefix_cache
                        and _attn_impl == "sdpa"
                        and (("full_attention" in _layer_types) or _mode == "hybrid"))
    if not prefix_cache and not args.no_prefix_cache and _attn_impl != "sdpa":
        print(f"Prefix KV cache disabled: attn_implementation={_attn_impl} does not honor "
              "the packed block mask (set FLASH_ATTN=0 to enable the fast path)")
    print(f"Prefix KV cache: {prefix_cache}"
          + (f" (mode={_mode}"
             + (f", pack {args.pack_tokens} tokens)" if _mode == "packed"
                else f", group {args.prefix_group_size})")
             if prefix_cache else ""))

    # Get Yes/No token IDs
    yes_tokens = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tokens = tokenizer.encode(" No", add_special_tokens=False)

    # Handle tokenizers that might split " Yes"/" No" into multiple tokens.
    # Use [-1] (the content token e.g. "Yes") rather than [0] (which could be a generic space).
    if len(yes_tokens) != 1:
        print(f"WARNING: ' Yes' tokenized into {len(yes_tokens)} tokens {yes_tokens}; using last token.")
    if len(no_tokens) != 1:
        print(f"WARNING: ' No' tokenized into {len(no_tokens)} tokens {no_tokens}; using last token.")
    yes_token_id = yes_tokens[0] if len(yes_tokens) == 1 else yes_tokens[-1]
    no_token_id = no_tokens[0] if len(no_tokens) == 1 else no_tokens[-1]

    print(f"Yes token ID: {yes_token_id} ('{tokenizer.decode([yes_token_id])}')")
    print(f"No token ID: {no_token_id} ('{tokenizer.decode([no_token_id])}')")

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    def _metric(xs, fmt="{:.4f}"):
        """Format a metric, or 'n/a' when the split has no labels.

        The MIND test split has no ground-truth labels, so every metric stays
        empty and `_avg` would report 0.0000 — which reads like a broken run
        rather than "no labels to score against". Report 'n/a' instead.
        """
        return fmt.format(_avg(xs)) if xs else "n/a"

    aucs = []
    mrrs = []
    ndcg5 = []
    ndcg10 = []
    predictions = []
    raw_scores = []

    # Stream raw scores to disk as they are produced. Long test-split scoring
    # runs can hit the scheduler wall limit, and a buffered write would then
    # lose every score. Appending + flushing keeps partial results usable.
    scores_fh = None
    if args.output_scores_file:
        import os as _os2
        _os2.makedirs(_os2.path.dirname(_os2.path.abspath(args.output_scores_file)), exist_ok=True)
        scores_fh = open(args.output_scores_file, "w", encoding="utf-8")

    # Count total lines for progress bar
    total_lines = None
    if not args.max_impressions:
        with open(args.behaviors_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped_malformed = 0

    print(f"\nEvaluating with point-wise format (Yes/No)...")
    print(f"Use abstract: {args.use_abstract}")
    print(f"Max history: {'unlimited' if args.max_history == 0 else args.max_history}")
    print(f"Batch size: {args.batch_size}")
    if args.quick:
        print(f"Quick mode: {args.max_impressions} impressions")
    print()

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Evaluating impressions", unit="impression")
        for line in f:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                skipped_malformed += 1
                continue

            impression_id, _, impression_ts, history_ids, imp_list = parsed

            if args.max_history > 0:
                if args.history_selection == "dedup_backfill":
                    from mind_utils import dedup_history_ids
                    history_ids = dedup_history_ids(history_ids, max_history=args.max_history, backfill=True)
                elif args.history_selection == "dedup_only":
                    from mind_utils import dedup_history_ids
                    history_ids = dedup_history_ids(history_ids, max_history=args.max_history, backfill=False)
                elif args.history_selection == "catcap":
                    from mind_utils import catcap_history_ids
                    history_ids = catcap_history_ids(history_ids, news, max_history=args.max_history, per_cat_cap=10)
                elif args.history_selection == "relevant":
                    # Dynamic selection mirroring data.py: keep the max_history
                    # items whose category matches the impression's candidates
                    # (recency as tiebreak), re-rendered chronologically.
                    cand_cats = set()
                    for nid, _lbl in imp_list:
                        n = news.get(nid)
                        if n:
                            cand_cats.add(n.get('category', ''))
                    if cand_cats:
                        scored = []
                        for rank, nid in enumerate(reversed(history_ids)):
                            n = news.get(nid)
                            if n is None:
                                continue
                            scored.append((1 if n.get('category', '') in cand_cats else 0, rank, nid))
                        scored.sort(key=lambda t: (-t[0], t[1]))
                        sel = {t[2] for t in scored[:args.max_history]}
                        history_ids = [nid for nid in history_ids if nid in sel]
                    else:
                        history_ids = history_ids[-args.max_history:]
                else:
                    history_ids = history_ids[-args.max_history:]

            labels = []
            candidate_objs = []
            candidate_ids = []

            for nid, label in imp_list:
                if nid not in news:
                    candidate_objs.append({'text': '[MISSING_NEWS]', 'category': ''})
                else:
                    candidate_objs.append(news[nid])

                candidate_ids.append(nid)
                labels.append(label)

            # Skip if no candidates
            if not candidate_objs:
                continue

            # Build history
            history_objs = [news[nid] for nid in history_ids if nid in news]

            # Score all candidates using batched point-wise scoring
            scores = batch_score_candidates_pointwise(
                model, tokenizer, history_objs, candidate_objs, device,
                yes_token_id, no_token_id, args.batch_size, args.use_chat_template,
                args.use_subcategory,
                temperature=args.temperature,
                use_recency=args.use_recency,
                use_profile_summary=args.use_profile_summary,
                impression_timestamp=impression_ts if args.use_impression_timestamp else None,
                history_format=args.history_format,
                prefix_cache=prefix_cache,
                pack_tokens=args.pack_tokens,
                legacy_bf16_scores=args.legacy_bf16_scores,
                prefix_cache_mode=_mode,
                prefix_group_size=args.prefix_group_size,
                right_pad=args.right_pad,
                prompt_parity=args.prompt_parity,
                parity_max_len=args.parity_max_len,
                answer_len=len(yes_tokens),
            )

            # Blend with CF scores if available
            if cf_scores:
                import numpy as _np
                lm_arr = _np.array(scores)
                cf_arr = _np.array([cf_scores.get((impression_id, nid), 0.0) for nid in candidate_ids])
                # Rank-based normalisation: convert each signal to its within-impression
                # fractional rank in [0, 1].  This is robust to scale and sparsity —
                # min-max and z-score both amplify tiny differences in near-zero CF scores
                # (e.g. [0.041, 0.0, 0.0, 0.038] → z-scores [1.94, -0.33, -0.33, 1.77])
                # which caused the catastrophic AUC regression (0.7049 → 0.6213).
                # Rank norm preserves ordinal information without stretching noise.
                def _rank_norm(x):
                    n = len(x)
                    if n <= 1:
                        return _np.zeros_like(x, dtype=float)
                    # argsort twice gives rank (0-based); divide by (n-1) → [0, 1]
                    return _np.argsort(_np.argsort(x)).astype(float) / (n - 1)
                scores = ((1 - args.cf_alpha) * _rank_norm(lm_arr) + args.cf_alpha * _rank_norm(cf_arr)).tolist()

            # Compute metrics (only if we have positive labels)
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            # Generate ranked predictions
            if args.output_file:
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_news_ids = [candidate_ids[i] for i in ranked_indices]
                predictions.append((impression_id, ranked_news_ids))

            # Save raw scores for ensemble
            if scores_fh is not None:
                scores_str = "\t".join(f"{s:.6f}" for s in scores)
                scores_fh.write(f"{impression_id}\t{scores_str}\n")
                raw_scores.append(f"{impression_id}\t{scores_str}")

            count += 1

            # Progress heartbeat (unbuffered) so AML/std_log shows forward progress.
            # tqdm suppresses its bar on a non-TTY and the parallel sh pipes through
            # sed which fully buffers, so without this a healthy run looks hung.
            if count % 25 == 0:
                print(f"[heartbeat] impressions={count} "
                      f"AUC={_metric(aucs)} MRR={_metric(mrrs)} "
                      f"nDCG@10={_metric(ndcg10)}", flush=True)
                if scores_fh is not None:
                    scores_fh.flush()

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'AUC': _metric(aucs),
                'MRR': _metric(mrrs),
                'nDCG@5': _metric(ndcg5),
                'nDCG@10': _metric(ndcg10)
            })

            if args.max_impressions and count >= args.max_impressions:
                break

        pbar.close()

    print("\nMIND Evaluation (Point-wise, Yes/No)")
    print(f"Impressions processed: {count}")
    if skipped_malformed > 0:
        print(f"Skipped malformed lines: {skipped_malformed}")

    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No metrics computed (test set has no labels)")

    # Write predictions
    if args.output_file and predictions:
        print(f"\nWriting predictions to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for impression_id, ranked_news_ids in predictions:
                f.write(f"{impression_id} {' '.join(ranked_news_ids)}\n")
        print(f"Wrote {len(predictions)} predictions")

    # Raw scores were streamed to disk during the run (see scores_fh above).
    if scores_fh is not None:
        scores_fh.close()
        print(f"\nWrote {len(raw_scores)} score lines to: {args.output_scores_file}")


if __name__ == "__main__":
    main()
