#!/usr/bin/env python3
"""Append dev-derived popularity columns to a feature table (test inference).

Computes pop/ctr/log_pop from the FULL dev behaviors (all 376,471 impressions —
at inference time there is no honesty constraint; the val-slice honesty only
applied while selecting the ranker) and appends 3 columns to a copy of the
input table. The ranker of record (ranker_v1_lg9b_dev80_devpop.txt) expects
BASE_FEATS (14) + 3 devpop columns = 17 features.

Usage:
    python src/append_devpop_test.py \
        --features results/ranker/lg9b_test_features.tsv \
        --dev_behaviors data/MIND_large/dev/behaviors.tsv \
        --output results/ranker/lg9b_test_features_devpop.tsv
"""
import argparse
import math
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--dev_behaviors", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("Computing dev popularity ...", flush=True)
    pop = Counter()
    clicks = Counter()
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
                if lab == "1":
                    clicks[nid] += 1
    max_pop = max(pop.values()) if pop else 1
    log_max = math.log1p(max_pop)
    print(f"  {len(pop):,} unique dev candidates, max_pop={max_pop}", flush=True)

    with open(args.features) as f, open(args.output, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
        out.write("\t".join(header + ["dev_pop", "dev_ctr", "dev_log_pop"]) + "\n")
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            nid = p[i_nid]
            dp = pop.get(nid, 0)
            dc = clicks.get(nid, 0)
            p.append(f"{dp / max_pop:.6f}")
            p.append(f"{(dc / dp) if dp > 0 else 0.0:.6f}")
            p.append(f"{math.log1p(dp) / log_max:.6f}")
            out.write("\t".join(p) + "\n")
            n += 1
            if n % 10_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
    print(f"Wrote {n:,} rows -> {args.output}", flush=True)


if __name__ == "__main__":
    main()