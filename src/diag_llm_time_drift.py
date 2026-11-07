#!/usr/bin/env python3
"""Does the LLM degrade over time? (dev -> test drift diagnostic)

WHY
---
The previous test submission scored 0.6986 while its dev number was ~0.707.
Part of that gap could be TIME DRIFT: dev covers 11/10-11/15 and test covers
11/16-11/22, so if the LLM's ranking quality decays as the news distribution
moves, the dev number is optimistic for test.

This splits dev into 5 consecutive time blocks (by impression order, which is
chronological in MIND) and reports the LLM's per-impression AUC in each block.
A flat profile means the dev number transfers; a declining profile means the
dev number is optimistic and the test score will be lower for reasons no
amount of ranker work can fix.

Also reports the same for the ranker, so we can see whether the RANKER's
gain over the LLM is stable over time (that is the thing we control).

Usage: python src/diag_llm_time_drift.py [--blocks 5]
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"


def load_llm(path):
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def imp_ids(path):
    out = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_imp = header.index("impression_id")
        cur = None
        n = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[i_imp] != cur:
                if cur is not None:
                    out.append((cur, n))
                cur = p[i_imp]
                n = 0
            n += 1
        out.append((cur, n))
    return out


def per_imp(y_t, s, gg):
    order = np.argsort(gg, kind="stable")
    g_s, p_s, y_s = gg[order], s[order], y_t[order]
    ug = np.unique(g_s)
    starts = np.searchsorted(g_s, ug, side="left")
    ends = np.searchsorted(g_s, ug, side="right")
    ranks = np.empty(len(p_s))
    for s0, e0 in zip(starts, ends):
        o = np.argsort(p_s[s0:e0], kind="stable")
        ranks[s0:e0] = o.argsort() + 1
    pos = y_s == 1
    n_pos = np.bincount(g_s[pos], minlength=len(ug)).astype(float)
    n_neg = np.bincount(g_s[~pos], minlength=len(ug)).astype(float)
    rsum = np.bincount(g_s[pos], weights=ranks[pos], minlength=len(ug))
    valid = (n_pos > 0) & (n_neg > 0)
    auc = (rsum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) / (n_pos[valid] * n_neg[valid])
    return auc, ug[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5)
    args = ap.parse_args()

    print("Loading dev features ...", flush=True)
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(DEV_FEATS)):
        Xs.append(Xb)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    del Xs, ys
    ids = imp_ids(DEV_FEATS)
    llm9 = load_llm(LG9B_DEV)
    off = 0
    for imp, n in ids:
        s = llm9.get(imp)
        if s and len(s) == n:
            X[off:off + n, 0] = s
        off += n
    n_imp = g.max() + 1
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

    # LLM-only AUC per impression
    auc_llm, valid_imps = per_imp(y, X[:, 0], g)
    print(f"  LLM-only per-impression AUC (all dev): {auc_llm.mean():.4f} "
          f"(n={len(auc_llm):,})", flush=True)

    # split the VALID impressions into consecutive time blocks
    n_blocks = args.blocks
    edges = np.linspace(0, len(valid_imps), n_blocks + 1).astype(int)
    print("\n=== LLM-only AUC by dev time block (block 5 is closest to test) ===")
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        seg = auc_llm[lo:hi]
        print(f"  block {b+1}/{n_blocks}  imps[{valid_imps[lo]:,}..{valid_imps[hi-1]:,}]  "
              f"n={len(seg):>7,}  AUC={seg.mean():.4f}", flush=True)

    # also: the dev-80% -> dev-20% split the ranker experiments use
    cut = int(n_imp * 0.8)
    is_late = valid_imps >= cut
    print(f"\n  dev-80% (train block) LLM AUC: {auc_llm[~is_late].mean():.4f} "
          f"(n={(~is_late).sum():,})")
    print(f"  dev-20% (val block)   LLM AUC: {auc_llm[is_late].mean():.4f} "
          f"(n={is_late.sum():,})")
    print(f"  drift (val - train): {auc_llm[is_late].mean() - auc_llm[~is_late].mean():+.4f}")


if __name__ == "__main__":
    main()
