"""Validate the token-additive overflow ESTIMATOR against the EXACT prompt length.

WHY
---
scripts/count_dev_overflow.py and scripts/build_overlong_subset.py estimate a
prompt's token length as

    len(history block) + len(candidate block) + overhead

where the history block is rendered with an EMPTY candidate and the candidate
block with an EMPTY history. That is token-additive, but the chat template
re-renders the whole prompt string, so the parts are not exactly additive.
If the estimator is biased, the overflow FRACTION (which is the multiplier
used to extrapolate the A/B delta to the full dev set) is biased too.

This script measures both numbers for a random sample of dev impressions:
  * estimate = h_len + cand_len + overhead   (the estimator)
  * exact    = len(tokenizer(render(history, candidate)))  (ground truth)
and reports the signed bias and the resulting error in the overflow fraction.

Usage:
    python scripts/validate_overflow_estimator.py --sample 4000 --parts 00
"""
import argparse
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from transformers import AutoTokenizer

from mind_utils import build_pointwise_prompt, load_news

MODEL = "models/Qwen3.5-9B"
DEV = "data/MIND_large/dev"
MAX_LEN = 2048
ANSWER_LEN = 1
MAX_PROMPT = MAX_LEN - ANSWER_LEN
CACHE = "results/ranker/parity_cand_len.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    ap.add_argument("--parts", default="00")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    news = load_news(f"{DEV}/news.tsv", True, use_entity=False)
    with open(CACHE, "rb") as f:
        cand_len, overhead = pickle.load(f)
    empty = {"text": "", "category": ""}

    rng = random.Random(args.seed)
    n_est_over = 0
    n_exact_over = 0
    n = 0
    biases = []
    disagreements = 0

    for part in args.parts.split(","):
        src = f"{DEV}/chunks/sub/d00/chunk_{part.strip()}.tsv"
        lines = open(src).readlines()
        idx = rng.sample(range(len(lines)), min(args.sample, len(lines)))
        for i in idx:
            p = lines[i].rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            hist_ids = p[3].split()[-30:]
            hist = [news[x] for x in hist_ids if x in news]
            if not hist:
                continue
            cands = [c.rsplit("-", 1)[0] for c in p[4].split() if "-" in c]
            cands = [c for c in cands if c in news]
            if not cands:
                continue

            h_prompt = build_pointwise_prompt(
                hist, empty, tokenizer=tok, use_chat_template=True,
                use_recency=True, use_profile_summary=False,
                impression_timestamp=None, history_format="flat")
            h_len = len(tok.encode(h_prompt, add_special_tokens=False)) - overhead

            # WORST candidate (the longest one decides whether the impression
            # overflows, because the parity path truncates per candidate).
            worst_nid = max(cands, key=lambda c: cand_len.get(c, 0))
            cl = cand_len.get(worst_nid, 0)
            # cand_len already contains the fixed scaffolding (it was measured
            # with an EMPTY history), so it must NOT be added to `overhead`
            # again. The additive form is exactly:
            #   exact = overhead + hist_block + cand_block
            #         = h_len + cand_len
            est = h_len + cl

            full = build_pointwise_prompt(
                hist, news[worst_nid], tokenizer=tok, use_chat_template=True,
                use_recency=True, use_profile_summary=False,
                impression_timestamp=None, history_format="flat")
            exact = len(tok.encode(full, add_special_tokens=False))

            n += 1
            biases.append(exact - est)
            e_over = est > MAX_PROMPT
            x_over = exact > MAX_PROMPT
            n_est_over += e_over
            n_exact_over += x_over
            if e_over != x_over:
                disagreements += 1

    if not n:
        print("no samples")
        return
    biases.sort()
    print("=" * 62)
    print(f"samples: {n:,}")
    print(f"estimator bias (exact - est): mean {sum(biases)/n:+.2f} tokens, "
          f"median {biases[n//2]:+d}, p5 {biases[int(0.05*n)]:+d}, "
          f"p95 {biases[int(0.95*n)]:+d}")
    print(f"overflow by ESTIMATOR: {n_est_over:,} ({100.0*n_est_over/n:.2f}%)")
    print(f"overflow by EXACT    : {n_exact_over:,} ({100.0*n_exact_over/n:.2f}%)")
    print(f"estimator/exact disagreements: {disagreements:,} "
          f"({100.0*disagreements/n:.2f}%)")
    print("=" * 62)


if __name__ == "__main__":
    main()
