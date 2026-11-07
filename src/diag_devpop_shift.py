#!/usr/bin/env python3
"""Do the devpop features have the SAME distribution at train time and test time?

WHY
---
The ranker's devpop features are built two different ways:

  TRAIN (dev rows): CAUSAL -- each row sees only PRIOR dev impressions.
                    So dev_pop is small for early rows and grows; dev_ctr is
                    clicks_so_far / pop_so_far, which is 0/1 = 0 for a news
                    seen once and never clicked.
  TEST  (test rows): FULL DEV -- all of dev precedes test, so dev_pop is the
                    complete dev count and dev_ctr is a stable estimate.

The POP scale is handled by the as-of normalisation (divide by the max count
seen SO FAR; at test that max IS the full-dev max). But the CTR is NOT
normalised: a training row can only ever see a noisy early CTR, while a test
row always sees the converged CTR. If those distributions differ, the model
learns a CTR mapping that does not hold at test time.

This script prints the two distributions side by side so the mismatch (if
any) is visible, and reports how many training rows have a degenerate CTR
(pop_so_far == 1).

Usage: python src/diag_devpop_shift.py
"""
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")

DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"


def main():
    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
    z2 = np.load(CACHE2)
    S = {k: z2[k] for k in z2.files}

    # ---- TRAIN-time (causal) features ----
    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    tr_pop = C["causal_pop"] / mp
    tr_ctr = np.where(C["causal_pop"] > 0,
                      C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0)
    tr_cat_pop = C["causal_cat_pop"] / mc
    tr_cat_ctr = np.where(C["causal_cat_pop"] > 0,
                          C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    tr_sub_pop = S["sub_pop"] / ms
    tr_sub_ctr = np.where(S["sub_pop"] > 0,
                          S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)

    # ---- TEST-time (full dev) features ----
    news_cat = {}
    news_sub = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                news_cat[p[0]] = p[1]
            if len(p) >= 3:
                news_sub[p[0]] = p[2]
    pop = Counter()
    clicks = Counter()
    cat_pop = Counter()
    cat_clicks = Counter()
    sub_pop = Counter()
    sub_clicks = Counter()
    with open(DEV_BEHAVIORS) as f:
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
    max_cat = max(cat_pop.values()) if cat_pop else 1
    max_sub = max(sub_pop.values()) if sub_pop else 1

    # A test row's devpop depends on its candidate's news. Weight each news by
    # how often it appears as a test candidate, so the distribution reflects
    # what the model actually sees at inference.
    te_pop = []
    te_ctr = []
    te_cat_pop = []
    te_cat_ctr = []
    te_sub_pop = []
    te_sub_ctr = []
    with open("data/MIND_large/test/behaviors.tsv") as f:
        for i, line in enumerate(f):
            if i >= 200000:
                break
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            for nid in p[4].split():
                dp = pop.get(nid, 0)
                te_pop.append(dp / max_pop)
                te_ctr.append((clicks[nid] / dp) if dp > 0 else 0.0)
                cat = news_cat.get(nid, "")
                cp = cat_pop.get(cat, 0)
                te_cat_pop.append(cp / max_cat)
                te_cat_ctr.append((cat_clicks[cat] / cp) if cp > 0 else 0.0)
                sub = news_sub.get(nid, "")
                sp = sub_pop.get(sub, 0)
                te_sub_pop.append(sp / max_sub)
                te_sub_ctr.append((sub_clicks[sub] / sp) if sp > 0 else 0.0)
    te_pop = np.array(te_pop)
    te_ctr = np.array(te_ctr)
    te_cat_pop = np.array(te_cat_pop)
    te_cat_ctr = np.array(te_cat_ctr)
    te_sub_pop = np.array(te_sub_pop)
    te_sub_ctr = np.array(te_sub_ctr)

    def show(name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        print(f"\n{name}")
        print(f"  TRAIN(causal) n={len(a):>10,}  mean={a.mean():.4f}  "
              f"p25={np.percentile(a,25):.4f}  med={np.median(a):.4f}  "
              f"p75={np.percentile(a,75):.4f}  p99={np.percentile(a,99):.4f}  "
              f"frac==0: {np.mean(a==0):.4f}")
        print(f"  TEST (full)  n={len(b):>10,}  mean={b.mean():.4f}  "
              f"p25={np.percentile(b,25):.4f}  med={np.median(b):.4f}  "
              f"p75={np.percentile(b,75):.4f}  p99={np.percentile(b,99):.4f}  "
              f"frac==0: {np.mean(b==0):.4f}")

    show("dev_pop (normalised)", tr_pop, te_pop)
    show("dev_ctr", tr_ctr, te_ctr)
    show("cat_pop (normalised)", tr_cat_pop, te_cat_pop)
    show("cat_ctr", tr_cat_ctr, te_cat_ctr)
    show("sub_pop (normalised)", tr_sub_pop, te_sub_pop)
    show("sub_ctr", tr_sub_ctr, te_sub_ctr)

    # how degenerate is the training CTR?
    cp = C["causal_pop"]
    print(f"\n  training rows with causal_pop == 0: {np.mean(cp == 0):.4f}")
    print(f"  training rows with causal_pop == 1: {np.mean(cp == 1):.4f}")
    print(f"  training rows with causal_pop >= 10: {np.mean(cp >= 10):.4f}")
    print(f"  training rows with causal_pop >= 100: {np.mean(cp >= 100):.4f}")
    print(f"  test rows with dev_pop == 0: {np.mean(te_pop == 0):.4f}")


if __name__ == "__main__":
    main()
