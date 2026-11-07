"""Build the OVER-LENGTH dev impression subset for the prompt-parity A/B.

WHY A SUBSET
------------
The parity path (src/evaluate_mind_pointwise.py `_build_parity_prompt_ids`)
reproduces training's history-shrink loop. For an impression whose prompt
already FITS in 2048 tokens, the parity path is a no-op: the rendered prompt is
identical, so the two arms produce identical scores. The parity arms can
therefore only differ on impressions whose prompt OVERFLOWS.

So the A/B only has to score the over-length impressions. On MIND_large dev
that is a few percent of 376,471 impressions -- cheap enough to run BOTH arms
on one node, which is what makes the comparison paired (same GPUs, same node,
same wall-clock state) and therefore free of node-to-node hardware variation.

HOW THE SUBSET IS BUILT
-----------------------
The prompt length is (history block) + (candidate block) + fixed overhead.
The history block is rendered ONCE per impression; the candidate block is
rendered ONCE per news id (it does not depend on the impression). Both are
measured with the REAL renderer and the REAL tokenizer.

The filter is deliberately over-inclusive: an impression is kept when its
estimated prompt length is within MARGIN tokens of the limit, or above it. The
extra impressions are harmless -- they are identical in both arms and act as a
control group.

Usage:
    python scripts/build_overlong_subset.py --parts 00,01,02,03,04,05 \
        --outdir data/MIND_large/dev/chunks/parity
"""
import argparse
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
MARGIN = 48  # tokens of slack: keeps every borderline impression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="comma list, e.g. 00,01,02")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-history", type=int, default=30)
    ap.add_argument("--cache", default="results/ranker/parity_cand_len.pkl")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    news = load_news(f"{DEV}/news.tsv", True, use_entity=False)
    print(f"news articles: {len(news):,}", flush=True)

    empty = {"text": "", "category": ""}

    # The candidate-block length does not depend on the impression, so it is
    # computed once and cached across parts.
    if os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            cand_len, overhead = pickle.load(f)
        print(f"loaded candidate-length cache: {len(cand_len):,} news ids", flush=True)
    else:
        cand_len = {}
        for i, (nid, n) in enumerate(news.items()):
            p = build_pointwise_prompt(
                [], n, tokenizer=tok, use_chat_template=True,
                use_recency=False, use_profile_summary=False,
                impression_timestamp=None, history_format="flat")
            cand_len[nid] = len(tok.encode(p, add_special_tokens=False))
            if (i + 1) % 20000 == 0:
                print(f"  candidate lengths {i+1}/{len(news)}", flush=True)
        p_empty = build_pointwise_prompt(
            [], empty, tokenizer=tok, use_chat_template=True,
            use_recency=False, use_profile_summary=False,
            impression_timestamp=None, history_format="flat")
        overhead = len(tok.encode(p_empty, add_special_tokens=False))
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump((cand_len, overhead), f)
        print(f"cached candidate lengths -> {args.cache} (overhead={overhead})", flush=True)

    os.makedirs(args.outdir, exist_ok=True)

    for part in args.parts.split(","):
        part = part.strip()
        src = f"{DEV}/chunks/sub/d00/chunk_{part}.tsv"
        out = f"{args.outdir}/p{part}.tsv"
        kept = 0
        total = 0
        with open(src) as fin, open(out, "w") as fout:
            for line in fin:
                total += 1
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                hist_ids = p[3].split()[-args.max_history:]
                hist = [news[x] for x in hist_ids if x in news]
                if not hist:
                    continue
                # History block: rendered exactly like training/eval do.
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
                    est = h_len + cl + overhead
                    if est > worst:
                        worst = est
                if worst > MAX_PROMPT - MARGIN:
                    fout.write(line)
                    kept += 1

        print(f"part {part}: {kept:,} / {total:,} impressions kept "
              f"({100.0*kept/max(total,1):.2f}%) -> {out}", flush=True)


if __name__ == "__main__":
    main()
