"""Count, over the FULL MIND_large dev split, how many impressions the
prompt-parity path would actually change.

WHY
---
The paired A/B (scripts/eval_parity_ab.sh) measures the parity effect on the
over-length SUBSET. To turn that into a number for the full dev set we need
the exact fraction of full-dev impressions whose prompt overflows 2047 tokens,
because:

    full-dev delta  =  (N_overflow / N_dev)  x  mean delta on the overflow ones

The parity path is a NO-OP for a prompt that already fits (the rendered prompt
is byte-identical), so the fitting impressions contribute exactly zero and
need no measurement -- they are the A/B's control group.

The counting is done with the SAME renderer and the SAME tokenizer the
training and eval code use, and the candidate-block lengths come from the
cache written by scripts/build_overlong_subset.py, so the estimate is exact
rather than sampled.

Usage:
    python scripts/count_dev_overflow.py
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
MAX_LEN = 2048
ANSWER_LEN = 1
MAX_PROMPT = MAX_LEN - ANSWER_LEN
CACHE = "results/ranker/parity_cand_len.pkl"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    news = load_news(f"{DEV}/news.tsv", True, use_entity=False)
    print(f"news articles: {len(news):,}", flush=True)

    with open(CACHE, "rb") as f:
        cand_len, overhead = pickle.load(f)
    print(f"candidate-length cache: {len(cand_len):,} news ids (overhead={overhead})",
          flush=True)

    empty = {"text": "", "category": ""}

    total = 0
    over = 0
    per_part = {}
    for part in ("00", "01", "02", "03", "04", "05"):
        src = f"{DEV}/chunks/sub/d00/chunk_{part}.tsv"
        if not os.path.exists(src):
            print(f"part {part}: MISSING {src}", flush=True)
            continue
        p_total = 0
        p_over = 0
        with open(src) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                total += 1
                p_total += 1
                hist_ids = p[3].split()[-30:]
                hist = [news[x] for x in hist_ids if x in news]
                if not hist:
                    continue
                h_prompt = build_pointwise_prompt(
                    hist, empty, tokenizer=tok, use_chat_template=True,
                    use_recency=True, use_profile_summary=False,
                    impression_timestamp=None, history_format="flat")
                h_len = len(tok.encode(h_prompt, add_special_tokens=False)) - overhead
                worst = 0
                for imp in p[4].split():
                    if "-" not in imp:
                        continue
                    nid = imp.rsplit("-", 1)[0]
                    cl = cand_len.get(nid)
                    if cl is None:
                        continue
                    # NOTE: cand_len already includes the fixed scaffolding
                    # (system prompt + question + assistant header), because it
                    # was measured by rendering the candidate with an EMPTY
                    # history. Adding `overhead` again double-counts it -- the
                    # measured bias was exactly -82 tokens per impression
                    # (scripts/validate_overflow_estimator.py).
                    est = h_len + cl
                    if est > worst:
                        worst = est
                if worst > MAX_PROMPT:
                    over += 1
                    p_over += 1
        per_part[part] = (p_total, p_over)
        print(f"part {part}: {p_over:,} / {p_total:,} overflow "
              f"({100.0*p_over/max(p_total,1):.2f}%)", flush=True)

    print("=" * 60)
    print(f"FULL DEV (6 sub-chunks): {over:,} / {total:,} impressions overflow "
          f"({100.0*over/max(total,1):.3f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
