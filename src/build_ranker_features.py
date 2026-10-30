#!/usr/bin/env python3
"""
Build the per-candidate feature table for the Stage-2 ranker probe (MIND_small dev).

Joins, for every (impression, candidate) pair in dev behaviors:
  - LLM score from a raw-scores dump (impression_id \t s1 \t s2 ...) — candidate
    order matches the behaviors.tsv impression list order.
  - CF score from data/MIND_small/dev/cf_scores.tsv (impression_id, news_id, cf).
  - Dense features computed from train split only (no dev leakage):
      * candidate popularity / CTR prior in train impressions
      * category / subcategory match fraction vs user history
      * history length, impression size, candidate position
      * user top-category concentration
      * hour-of-day from impression timestamp

Output: TSV with header, one row per (impression, candidate), label last.

Usage:
    python src/build_ranker_features.py \
        --behaviors data/MIND_small/dev/behaviors.tsv \
        --news      data/MIND_small/dev/news.tsv \
        --train_behaviors data/MIND_small/train/behaviors.tsv \
        --llm_scores results/scores/dpo_lora9b_dev_scores.txt \
        --cf_scores  data/MIND_small/dev/cf_scores.tsv \
        --output     results/ranker/dev_features.tsv
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mind_utils import _parse_timestamp_dt


def load_news(path):
    news = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            news[parts[0]] = {
                "category": parts[1],
                "subcategory": parts[2] if len(parts) > 2 else "",
                "title": parts[3],
            }
    return news


def load_train_stats(train_behaviors_path):
    """Popularity (impressions), clicks, and per-news category from TRAIN only."""
    pop = Counter()      # news_id -> times shown in train impressions
    clicks = Counter()   # news_id -> times clicked in train impressions
    with open(train_behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            for imp in parts[4].split():
                if "-" not in imp:
                    continue
                nid, label = imp.rsplit("-", 1)
                pop[nid] += 1
                if label == "1":
                    clicks[nid] += 1
    return pop, clicks


def load_llm_scores(path):
    """impression_id -> list of scores (candidate order = behaviors order)."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            out[parts[0]] = [float(x) for x in parts[1:]]
    return out


def load_cf_scores(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # impression_id news_id cf_score
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            out[(parts[0], parts[1])] = float(parts[2])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--news", required=True)
    ap.add_argument("--train_behaviors", required=True)
    ap.add_argument("--llm_scores", required=True)
    ap.add_argument("--cf_scores", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_history", type=int, default=30)
    args = ap.parse_args()

    news = load_news(args.news)
    pop, clicks = load_train_stats(args.train_behaviors)
    llm = load_llm_scores(args.llm_scores)
    cf = load_cf_scores(args.cf_scores)

    max_pop = max(pop.values()) if pop else 1
    total_train_shows = sum(pop.values()) or 1

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    header = [
        "impression_id", "news_id", "user_id",
        "llm_score", "cf_score",
        "cand_pop", "cand_ctr", "cand_log_pop",
        "cat_match_frac", "subcat_match_frac",
        "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
        "imp_size", "position", "position_frac",
        "hour",
        "label",
    ]

    n_rows = 0
    n_missing_llm = 0
    with open(args.behaviors, "r", encoding="utf-8") as f, \
         open(args.output, "w", encoding="utf-8") as out:
        out.write("\t".join(header) + "\n")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            impression_id, user_id, ts, hist_str, imp_str = parts[:5]
            history_ids = hist_str.split()[-args.max_history:]

            # history-derived user profile features
            hist_cats = [news[h]["category"] for h in history_ids if h in news]
            hist_subcats = [news[h]["subcategory"] for h in history_ids if h in news]
            cat_counts = Counter(hist_cats)
            top_cat_frac = (cat_counts.most_common(1)[0][1] / len(hist_cats)) if hist_cats else 0.0
            hist_cat_set = set(cat_counts)
            hist_subcat_set = set(hist_subcats)

            imp_list = [p for p in imp_str.split() if p]
            imp_size = len(imp_list)

            scores = llm.get(impression_id)
            if scores is None or len(scores) != imp_size:
                n_missing_llm += 1
                continue

            # 24h hour from the 12-hour MIND timestamp (AM/PM honored).
            _dt = _parse_timestamp_dt(ts)
            hour = _dt.hour if _dt is not None else -1

            for pos, (imp, llm_s) in enumerate(zip(imp_list, scores)):
                if "-" in imp:
                    nid, label = imp.rsplit("-", 1)
                else:
                    nid, label = imp, "0"  # unlabeled split (test): label placeholder
                n = news.get(nid)
                cat = n["category"] if n else ""
                subcat = n["subcategory"] if n else ""

                p = pop.get(nid, 0)
                c = clicks.get(nid, 0)
                ctr = (c / p) if p > 0 else 0.0

                cat_match = (sum(1 for hc in hist_cats if hc == cat) / len(hist_cats)) if hist_cats else 0.0
                subcat_match = (sum(1 for hs in hist_subcats if hs == subcat) / len(hist_subcats)) if hist_subcats else 0.0

                row = [
                    impression_id, nid, user_id,
                    f"{llm_s:.6f}", f"{cf.get((impression_id, nid), 0.0):.6f}",
                    f"{p / max_pop:.6f}", f"{ctr:.6f}",
                    f"{__import__('math').log1p(p) / __import__('math').log1p(max_pop):.6f}",
                    f"{cat_match:.6f}", f"{subcat_match:.6f}",
                    str(len(history_ids)), str(len(hist_cat_set)), f"{top_cat_frac:.6f}",
                    str(imp_size), str(pos), f"{pos / max(imp_size - 1, 1):.6f}",
                    str(hour),
                    label,
                ]
                out.write("\t".join(row) + "\n")
                n_rows += 1

    print(f"Wrote {n_rows:,} rows -> {args.output}")
    if n_missing_llm:
        print(f"WARNING: {n_missing_llm} impressions skipped (missing/mismatched LLM scores)")


if __name__ == "__main__":
    main()