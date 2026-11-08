#!/usr/bin/env python3
"""Recompute the FOUR dead category columns of the dev feature table.

WHY
---
`src/build_ranker_features.py::load_news()` used to skip any news line with
fewer than 4 columns (it wanted news_id, category, subcategory AND title).
The dev feature table was built with `data/MIND_large/news_all.tsv`, which has
only THREE columns (news_id, category, subcategory), so the loader returned
0 entries and these four features collapsed to identically zero across the
ENTIRE dev table:

    cat_match_frac, subcat_match_frac, hist_uniq_cats, hist_top_cat_frac

They are live on train and test (cat_match_frac is non-zero on 70.4% of test
rows), and v6 -- which was FITTED on the train table -- uses all four. That is
a train/eval feature mismatch. `load_news()` is now tolerant of 3-column
files, but rebuilding the whole 14M-row table would also recompute the other
20 columns from their sources, which risks perturbing them.

This script is deliberately minimal: it rewrites ONLY those four columns and
copies every other field of every other row through unchanged. A row is
therefore identical to the input except in the 4 patched fields.

The history cap must match the one used to build the table (30, the
`--max_history` default), otherwise `hist_uniq_cats` / `hist_top_cat_frac`
would disagree with the table's own `hist_len` column.

Usage:
    python src/patch_dev_cat_feats.py \
        --base results/ranker/ck45500_dev_features.tsv \
        --behaviors data/MIND_large/dev/behaviors.tsv \
        --news data/MIND_large/news_all.tsv \
        --out results/ranker/ck45500_dev_features_fixcat.tsv
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ranker_features import load_news  # noqa: E402

CAT_MATCH = "cat_match_frac"
SUBCAT_MATCH = "subcat_match_frac"
HIST_UNIQ = "hist_uniq_cats"
HIST_TOP = "hist_top_cat_frac"
PATCH_COLS = (CAT_MATCH, SUBCAT_MATCH, HIST_UNIQ, HIST_TOP)


def iter_profiles(behaviors_path, news, max_history):
    """Yield (impression_id, hist_len, hist_cats, hist_subcats, top_cat_frac, hist_cat_set).

    `hist_len` is the number of history items after the cap -- the same value
    the table's own hist_len column holds -- so we can cross-check it.
    """
    with open(behaviors_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            impression_id, hist_str = parts[0], parts[3]
            history_ids = hist_str.split()[-max_history:]
            hist_cats = [news[h]["category"] for h in history_ids if h in news]
            hist_subcats = [news[h]["subcategory"] for h in history_ids if h in news]
            cat_counts = Counter(hist_cats)
            top_cat_frac = (cat_counts.most_common(1)[0][1] / len(hist_cats)) if hist_cats else 0.0
            yield impression_id, len(history_ids), hist_cats, hist_subcats, top_cat_frac, set(cat_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--behaviors", required=True)
    ap.add_argument("--news", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_history", type=int, default=30)
    args = ap.parse_args()

    news = load_news(args.news)
    print(f"news entries: {len(news):,}", flush=True)
    if not news:
        raise SystemExit("FATAL: news file yielded 0 entries -- the loader is broken again")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    n_rows = 0
    n_imps = 0
    n_changed = 0
    n_hist_len_mismatch = 0

    profiles = iter_profiles(args.behaviors, news, args.max_history)

    with open(args.base, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        header = fin.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for c in PATCH_COLS + ("news_id", "hist_len"):
            if c not in idx:
                raise SystemExit(f"FATAL: column {c} not in {args.base} header")
        fout.write("\t".join(header) + "\n")

        cur_imp = None
        hist_len = 0
        hist_cats = []
        hist_subcats = []
        top_cat_frac = 0.0
        hist_cat_set = set()

        for line in fin:
            parts = line.rstrip("\n").split("\t")

            # advance the behavior profile whenever the impression changes
            if cur_imp is None or parts[0] != cur_imp:
                for prof in profiles:
                    if prof[0] == parts[0]:
                        cur_imp = prof[0]
                        hist_len, hist_cats, hist_subcats, top_cat_frac, hist_cat_set = prof[1:]
                        n_imps += 1
                        break
                else:
                    raise SystemExit(
                        f"FATAL: impression {parts[0]} is not in {args.behaviors} -- "
                        "the feature table and the behaviors file do not align"
                    )

            nid = parts[idx["news_id"]]
            n = news.get(nid)
            cat = n["category"] if n else ""
            subcat = n["subcategory"] if n else ""

            cat_match = (sum(1 for hc in hist_cats if hc == cat) / len(hist_cats)) if hist_cats else 0.0
            subcat_match = (sum(1 for hs in hist_subcats if hs == subcat) / len(hist_subcats)) if hist_subcats else 0.0

            new_vals = [
                f"{cat_match:.6f}",
                f"{subcat_match:.6f}",
                str(len(hist_cat_set)),
                f"{top_cat_frac:.6f}",
            ]
            if [parts[idx[c]] for c in PATCH_COLS] != new_vals:
                n_changed += 1
            for c, v in zip(PATCH_COLS, new_vals):
                parts[idx[c]] = v

            # cross-check: our capped history length must equal the table's hist_len
            if int(parts[idx["hist_len"]]) != hist_len:
                n_hist_len_mismatch += 1

            fout.write("\t".join(parts) + "\n")
            n_rows += 1
            if n_rows % 2_000_000 == 0:
                print(f"  {n_rows:,} rows", flush=True)

    print(f"Wrote {n_rows:,} rows / {n_imps:,} impressions -> {args.out}", flush=True)
    print(f"rows whose 4 patched columns changed: {n_changed:,} "
          f"({100.0 * n_changed / max(n_rows, 1):.2f}%)", flush=True)
    print(f"hist_len cross-check mismatches: {n_hist_len_mismatch:,}", flush=True)
    if n_hist_len_mismatch:
        raise SystemExit("FATAL: hist_len mismatch -- the history cap does not match the table's")
    if n_changed == 0:
        print("WARNING: nothing changed -- was the table already correct?", flush=True)


if __name__ == "__main__":
    main()
