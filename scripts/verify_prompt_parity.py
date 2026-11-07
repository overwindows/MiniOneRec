"""Verify the eval-side prompt-parity path is TOKEN-EXACT with training.

Training renders the prompt in data.py `MINDPointwiseSFTDataset.encode_example`
and, if it does not fit `max_len - len(target_ids)`, drops the OLDEST ~1/8 of
the history and re-renders until it fits.

Eval now has a parity path (`_build_parity_prompt_ids` in
src/evaluate_mind_pointwise.py) that must reproduce that EXACTLY. This script
compares the two token-id sequences on real MIND_large train samples, split
into "fits" and "overflow" cases, and asserts they are identical.

If they differ, the parity A/B is measuring the wrong thing and must not run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from transformers import AutoTokenizer

from data import MINDPointwiseSFTDataset
from evaluate_mind_pointwise import _build_parity_prompt_ids, _build_parity_prompt_ids_batch

MODEL = "models/Qwen3.5-9B"
ROOT = "data/MIND_large"
MAX_LEN = 2048

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)

ds = MINDPointwiseSFTDataset(
    behaviors_path=f"{ROOT}/train/behaviors.tsv",
    news_path=f"{ROOT}/train/news.tsv",
    tokenizer=tok,
    max_len=MAX_LEN,
    sample=4000,
    seed=42,
    max_history=30,
    neg_ratio=1.0,
    hard_neg_ratio=0.5,
    use_abstract=True,
    use_chat_template=True,
    use_subcategory=False,
    use_recency=True,
    use_profile_summary=False,
    use_impression_timestamp=False,
    use_entity=False,
    history_selection="recent",
    history_format="flat",
)

answer_len = len(tok.encode(" Yes", add_special_tokens=False))
max_prompt = MAX_LEN - answer_len

n_fit = n_over = 0
mismatch_fit = []
mismatch_over = []
batch_mismatch = 0
train_shrunk = 0

# Group the dataset samples by impression, so the BATCHED parity path can be
# checked against the per-candidate one on real impressions (the unit the
# scorer actually runs in).
by_impression = {}
for i, s in enumerate(ds.samples):
    by_impression.setdefault(s["impression_id"], []).append(i)

for i in range(len(ds)):
    s = ds.samples[i]
    # What TRAINING produces: prompt ids (labels are -100 there).
    input_ids, labels, alen = ds.encode_example(
        s["history"], s["candidate"], s["label"],
        impression_timestamp=s.get("impression_timestamp", ""),
    )
    train_prompt_ids = input_ids[:len(input_ids) - alen]

    # What EVAL's parity path produces.
    eval_prompt_ids = _build_parity_prompt_ids(
        tok, s["history"], s["candidate"], max_prompt,
        use_chat_template=True, use_subcategory=False, use_recency=True,
        use_profile_summary=False, impression_timestamp=None, history_format="flat",
    )

    over = len(train_prompt_ids) >= max_prompt
    if over:
        n_over += 1
    else:
        n_fit += 1

    if train_prompt_ids != eval_prompt_ids:
        rec = (i, len(train_prompt_ids), len(eval_prompt_ids))
        (mismatch_over if over else mismatch_fit).append(rec)

    # Did training actually shrink the history for this sample?
    # Recompute the un-shrunk prompt length to detect it.
    if len(s["history"]) and max_prompt < 4096:
        h = list(s["history"])
        # first render with full history
        from mind_utils import build_pointwise_prompt
        p = build_pointwise_prompt(
            h, s["candidate"], tokenizer=tok, use_chat_template=True,
            use_recency=True, use_profile_summary=False,
            impression_timestamp=None, history_format="flat")
        full_len = len(tok.encode(p, add_special_tokens=False))
        if full_len > max_prompt:
            train_shrunk += 1

# The batched path must equal the per-candidate path, impression by impression.
for imp_id, idxs in by_impression.items():
    s0 = ds.samples[idxs[0]]
    cands = [ds.samples[i]["candidate"] for i in idxs]
    batch_ids = _build_parity_prompt_ids_batch(
        tok, s0["history"], cands, max_prompt,
        use_chat_template=True, use_subcategory=False, use_recency=True,
        use_profile_summary=False, impression_timestamp=None, history_format="flat",
    )
    for j, i in enumerate(idxs):
        s = ds.samples[i]
        per = _build_parity_prompt_ids(
            tok, s["history"], s["candidate"], max_prompt,
            use_chat_template=True, use_subcategory=False, use_recency=True,
            use_profile_summary=False, impression_timestamp=None, history_format="flat",
        )
        if batch_ids[j] != per:
            batch_mismatch += 1

print(f"samples checked        : {len(ds)}")
print(f"  prompt fits 2048     : {n_fit}")
print(f"  prompt overflowed    : {n_over}")
print(f"  history was shrunk   : {train_shrunk}")
print(f"mismatch (fits)        : {len(mismatch_fit)}")
print(f"mismatch (overflow)    : {len(mismatch_over)}")
print(f"batched vs per-candidate mismatch: {batch_mismatch}")
if mismatch_fit[:5]:
    print("  fits examples:", mismatch_fit[:5])
if mismatch_over[:5]:
    print("  over examples:", mismatch_over[:5])

bad = len(mismatch_fit) + len(mismatch_over) + batch_mismatch
if bad:
    print(f"\nVERDICT: MISMATCH on {bad} cases -- parity path is NOT token-exact")
    sys.exit(1)
print(f"\nVERDICT: TOKEN-EXACT on all {len(ds)} samples "
      f"({train_shrunk} of them required a history shrink); "
      f"batched path == per-candidate path")
