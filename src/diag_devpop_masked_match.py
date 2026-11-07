#!/usr/bin/env python3
"""Does the MASKED training devpop distribution match the TEST devpop distribution?

WHY
---
diag_devpop_shift.py showed the RAW distributions differ enormously:
train dev_pop mean=0.2488 vs test mean=0.0095. Test-regime masking fixes the
SPARSITY (84.5% of test rows have dev_pop==0), but the masking only zeroes
rows out -- it does not change the VALUES of the rows that survive.

The values differ for a structural reason:
  TRAIN: dev_pop = causal_pop / asof_max_pop   (denominator = max count seen
         SO FAR, which grows through dev)
  TEST:  dev_pop = dev_count / full_dev_max    (denominator = max over ALL dev)

So the same news popularity maps to a different feature value at train and at
test. This script measures the residual mismatch on the NONZERO rows -- the
rows that survive masking -- which is what the model actually learns from.

Usage: python src/diag_devpop_masked_match.py
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

COV_DEVPOP = 0.155


def show(name, a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    print(f"\n{name}  (nonzero rows only)")
    print(f"  TRAIN(masked) n={len(a):>10,}  mean={a.mean():.4f}  "
          f"p25={np.percentile(a,25):.4f}  med={np.median(a):.4f}  "
          f"p75={np.percentile(a,75):.4f}  p99={np.percentile(a,99):.4f}")
    print(f"  TEST          n={len(b):>10,}  mean={b.mean():.4f}  "
          f"p25={np.percentile(b,25):.4f}  med={np.median(b):.4f}  "
          f"p75={np.percentile(b,75):.4f}  p99={np.percentile(b,99):.4f}")
    # how much of the test mass falls outside the train range?
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    print(f"  train p1..p99 = [{lo:.4f}, {hi:.4f}];  "
          f"test mass outside = {np.mean((b < lo) | (b > hi)):.4f}")


def main():
    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
    z2 = np.load(CACHE2)
    S = {k: z2[k] for k in z2.files}

    # ---- row -> impression index ----
    rows_per_imp = []
    for _Xb, y in stream_groups(DEV_FEATS):
        rows_per_imp.append(len(y))
    rows_per_imp = np.array(rows_per_imp)
    n_imp = len(rows_per_imp)
    imp_of_row = np.repeat(np.arange(n_imp), rows_per_imp)
    cut = int(n_imp * 0.8)
    tr = imp_of_row < cut
    va = ~tr

    # ---- TRAIN-side features, as the trainer builds them ----
    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    tr_pop_asof = C["causal_pop"] / mp
    tr_pop_const = C["causal_pop"] / np.maximum(C["max_pop"], 1.0)
    tr_ctr = np.where(C["causal_pop"] > 0,
                      C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0)
    tr_cat_pop = C["causal_cat_pop"] / mc
    tr_cat_ctr = np.where(C["causal_cat_pop"] > 0,
                          C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    tr_sub_pop = S["sub_pop"] / ms
    tr_sub_ctr = np.where(S["sub_pop"] > 0,
                          S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)

    # ---- TEST-side features, as the appender builds them ----
    news_cat, news_sub = {}, {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                news_cat[p[0]] = p[1]
            if len(p) >= 3:
                news_sub[p[0]] = p[2]
    pop, clicks = Counter(), Counter()
    cat_pop, cat_clicks = Counter(), Counter()
    sub_pop, sub_clicks = Counter(), Counter()
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
    max_pop = max(pop.values())
    max_cat = max(cat_pop.values())
    max_sub = max(sub_pop.values())

    te_pop, te_ctr, te_cat_pop, te_cat_ctr, te_sub_pop, te_sub_ctr = [], [], [], [], [], []
    with open("data/MIND_large/test/behaviors.tsv") as f:
        for i, line in enumerate(f):
            if i >= 300000:
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

    # ---- apply the SAME masking the trainer uses, on the TRAIN rows ----
    rng = np.random.default_rng(13)
    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])
    dev_hit = C["causal_pop"] > 0
    wmap = Counter()
    for n, h in zip(nids, dev_hit):
        if h:
            wmap[n] += 1
    want = int(COV_DEVPOP * len(nids))
    keys = list(wmap.keys())
    keep, cov = set(), 0
    for j in rng.permutation(len(keys)):
        n = keys[j]
        keep.add(n)
        cov += wmap[n]
        if cov >= want:
            break
    m_dev = np.array([n in keep for n in nids], dtype=bool) & dev_hit

    print(f"train rows: {tr.sum():,}  val rows: {va.sum():,}  "
          f"masked devpop coverage: {m_dev.mean():.4f} (target {COV_DEVPOP})")

    # the model trains on masked rows; compare the surviving rows' values
    for lbl, m in [("TRAIN", tr), ("VAL", va)]:
        mm = m & m_dev
        print(f"\n########## {lbl} block ##########")
        show("dev_pop  (as-of norm, what the trainer uses)",
             tr_pop_asof[mm], te_pop[te_pop > 0])
        show("dev_pop  (constant full-dev norm, alternative)",
             tr_pop_const[mm], te_pop[te_pop > 0])
        show("dev_ctr", tr_ctr[mm], te_ctr[te_ctr > 0])
        show("cat_pop", tr_cat_pop[mm], te_cat_pop[te_cat_pop > 0])
        show("cat_ctr", tr_cat_ctr[mm], te_cat_ctr[te_cat_ctr > 0])
        show("sub_pop", tr_sub_pop[mm], te_sub_pop[te_sub_pop > 0])
        show("sub_ctr", tr_sub_ctr[mm], te_sub_ctr[te_sub_ctr > 0])


if __name__ == "__main__":
    main()
