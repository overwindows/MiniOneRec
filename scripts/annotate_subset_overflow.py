"""Annotate the over-length dev subset with an exact OVERFLOW flag per impression.

WHY
---
The paired A/B measures the parity effect on the over-length subset. To turn
that into a full-dev number we need:

    full-dev delta  =  P(overflow)  x  E[delta | overflow]

P(overflow) is measured over the FULL dev split by
scripts/count_dev_overflow.py. E[delta | overflow] must be measured over the
subset impressions that ACTUALLY overflow -- the subset is deliberately
over-inclusive (it also holds impressions within 48 tokens of the limit, which
fit and are therefore no-ops under parity).

This script writes, for each subset part, a dict
    impression_id -> bool (does the prompt overflow 2047 tokens?)
computed with the SAME renderer and tokenizer the training and eval code use.
The candidate-block lengths come from the cache written by
scripts/build_overlong_subset.py.

Output: results/ranker/parity_overflow_flags.pkl
    {part: {impression_id: bool}}

Usage:
    python scripts/annotate_subset_overflow.py
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from transformers import AutoTokenizer

from mind_utils import build_pointwise_prompt, load_news

MODEL = "models/Qwen3.5-9B"
DEV = "data/MIND_large/dev"
SUBSET = "data/MIND_large/dev/chunks/parity"
MAX_LEN = 2048
ANSWER_LEN = 1
MAX_PROMPT = MAX_LEN - ANSWER_LEN
CACHE = "results/ranker/parity_cand_len.pkl"
OUT = "results/ranker/parity_overflow_flags.pkl"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    news = load_news(f"{DEV}/news.tsv", True, use_entity=False)
    print(f"news articles: {len(news):,}", flush=True)

    with open(CACHE, "rb") as f:
        cand_len, overhead = pickle.load(f)
    empty = {"text": "", "category": ""}

    flags = {}
    for part in ("00", "01", "02", "03", "04", "05"):
        src = f"{SUBSET}/p{part}.tsv"
        if not os.path.exists(src):
            print(f"part {part}: MISSING {src}", flush=True)
            continue
        pf = {}
        n_over = 0
        n = 0
        with open(src) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                imp = p[0]
                n += 1
                hist_ids = p[3].split()[-30:]
                hist = [news[x] for x in hist_ids if x in news]
                over = False
                if hist:
                    h_prompt = build_pointwise_prompt(
                        hist, empty, tokenizer=tok, use_chat_template=True,
                        use_recency=True, use_profile_summary=False,
                        impression_timestamp=None, history_format="flat")
                    h_len = len(tok.encode(h_prompt, add_special_tokens=False)) - overhead
                    for cand in p[4].split():
                        if "-" not in cand:
                            continue
                        nid = cand.rsplit("-", 1)[0]
                        cl = cand_len.get(nid)
                        if cl is None:
                            continue
                        # cand_len already includes the fixed scaffolding (it was
                        # measured with an EMPTY history), so `overhead` must
                        # NOT be added again -- that double-counted 82 tokens
                        # per impression (scripts/validate_overflow_estimator.py).
                        if h_len + cl > MAX_PROMPT:
                            over = True
                            break
                pf[imp] = over
                if over:
                    n_over += 1
        flags[part] = pf
        print(f"part {part}: {n_over:,} / {n:,} overflow "
              f"({100.0*n_over/max(n,1):.2f}%)", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(flags, f)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
