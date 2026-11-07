#!/usr/bin/env python3
"""Append dev-derived popularity + category + subcategory popularity to a
test feature table.

At inference time the whole of dev precedes test, so the "as-of" state is the
FULL dev state: the normalisation constant is the max count over all of dev.
This matches the causal training features (which normalise by the max count
observed so far) because at test time the as-of max IS the full-dev max.

Appends 7 columns:
    dev_pop, dev_ctr, dev_log_pop   -- per-news popularity from dev
    cat_pop, cat_ctr                -- per-CATEGORY popularity from dev
    sub_pop, sub_ctr                -- per-SUBCATEGORY popularity from dev

Category popularity is the only popularity signal with ~100% test coverage
(83% of test rows reference a news that appears in neither train nor dev),
so it is the feature that actually transfers. Subcategory popularity is
finer-grained (191 subcats) with 99.4% test coverage and adds a further
+0.0096 val AUC over category alone.

Usage:
    python src/append_devpop_cat_test.py \
        --features results/ranker/lg9b_test_features.tsv \
        --dev_behaviors data/MIND_large/dev/behaviors.tsv \
        --news data/MIND_large/test/news.tsv \
        --output results/ranker/lg9b_test_features_devpopcat.tsv
"""
import argparse
import math
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dev_behaviors", required=True)
    ap.add_argument("--news", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("Loading news categories ...", flush=True)
    news_cat = {}
    news_sub = {}
    with open(args.news) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                news_cat[p[0]] = p[1]
            if len(p) >= 3:
                news_sub[p[0]] = p[2]

    print("Computing dev popularity ...", flush=True)
    pop = Counter()
    clicks = Counter()
    cat_pop = Counter()
    cat_clicks = Counter()
    sub_pop = Counter()
    sub_clicks = Counter()
    with open(args.dev_behaviors) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            for tk in parts[4].split():
                if "-" not in tk:
                    continue
                nid, lab = tk.rsplit("-", 1)
                pop[nid] += 1
                cat = news_cat.get(nid, "")
                cat_pop[cat] += 1
                sub = news_sub.get(nid, "")
                sub_pop[sub] += 1
                if lab == "1":
                    clicks[nid] += 1
                    cat_clicks[cat] += 1
                    sub_clicks[sub] += 1
    max_pop = max(pop.values()) if pop else 1
    log_max = math.log1p(max_pop)
    max_cat_pop = max(cat_pop.values()) if cat_pop else 1
    max_sub_pop = max(sub_pop.values()) if sub_pop else 1
    print(f"  {len(pop):,} unique dev candidates, max_pop={max_pop}", flush=True)
    print(f"  {len(cat_pop):,} categories, max_cat_pop={max_cat_pop}", flush=True)
    print(f"  {len(sub_pop):,} subcategories, max_sub_pop={max_sub_pop}", flush=True)

    with open(args.features) as f, open(args.output, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
        out.write("\t".join(header + ["dev_pop", "dev_ctr", "dev_log_pop",
                                      "cat_pop", "cat_ctr",
                                      "sub_pop", "sub_ctr"]) + "\n")
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            nid = p[i_nid]
            dp = pop.get(nid, 0)
            dc = clicks.get(nid, 0)
            cat = news_cat.get(nid, "")
            cp = cat_pop.get(cat, 0)
            cc = cat_clicks.get(cat, 0)
            sub = news_sub.get(nid, "")
            sp = sub_pop.get(sub, 0)
            sc = sub_clicks.get(sub, 0)
            p.append(f"{dp / max_pop:.6f}")
            p.append(f"{(dc / dp) if dp > 0 else 0.0:.6f}")
            p.append(f"{math.log1p(dp) / log_max:.6f}")
            p.append(f"{cp / max_cat_pop:.6f}")
            p.append(f"{(cc / cp) if cp > 0 else 0.0:.6f}")
            p.append(f"{sp / max_sub_pop:.6f}")
            p.append(f"{(sc / sp) if sp > 0 else 0.0:.6f}")
            out.write("\t".join(p) + "\n")
            n += 1
            if n % 10_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
    print(f"Wrote {n:,} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
