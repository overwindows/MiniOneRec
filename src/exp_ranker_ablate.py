#!/usr/bin/env python3
"""Where does the stack's +0.059 gain actually live, and does it survive the
test regime?

WHY
---
The stack ranker claims +0.0590 per-impression AUC over the LLM on dev.
Two things must be true before that number can be trusted at test time:

  (1) The gain must come from features that EXIST at test time. 83.3% of
      test rows reference a news in neither train nor dev, so every
      popularity feature is ZERO for them. If the gain lives in the
      popularity features, it cannot transfer.

  (2) The gain must not come from a feature that is a no-op for the metric.
      `imp_size` is CONSTANT within an impression, so it cannot change the
      within-impression ranking by itself -- any per-impression AUC gain it
      produces must come through interactions (it acts as a gate on which
      other feature to trust). 27.6% of the model's gain sits on it, so we
      need to know what it is gating.

This script measures, on the SAME arm-G recipe (joint mask + WIT + binary
/2000 rounds):

  A. FEATURE-GROUP ABLATION -- retrain with one group removed at a time and
     report the per-impression AUC on the val block. The drop tells us how
     much each group contributes.

  B. GAIN BY MASK CELL -- evaluate the trained model separately on the rows
     whose news landed in each of the 4 joint-mask cells (both / train_only /
     dev_only / neither). The "neither" cell is the 83.3% test regime: no
     popularity features at all. If the gain survives there, it is real.

Usage: python src/exp_ranker_ablate.py [--rounds 2000] [--threads 12]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features_v2.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310
COV_CF = 0.113

# The 25 production features, in the order the trainer builds them.
ALL_FEATS = ["llm_score", "cf_score", "cand_pop", "cand_ctr", "cand_log_pop",
             "cat_match_frac", "subcat_match_frac", "hist_len",
             "hist_uniq_cats", "hist_top_cat_frac", "imp_size", "position",
             "position_frac", "hour", "dev_pop", "dev_ctr", "dev_log_pop",
             "cat_pop", "cat_ctr", "sub_pop", "sub_ctr", "llm_rank", "llm_z",
             "cf_rank", "cf_z"]

GROUPS = {
    "llm": ["llm_score"],
    "wit": ["llm_rank", "llm_z", "cf_rank", "cf_z"],
    "imp_size": ["imp_size"],
    "position": ["position", "position_frac"],
    "trainpop": ["cand_pop", "cand_ctr", "cand_log_pop"],
    "devpop": ["dev_pop", "dev_ctr", "dev_log_pop"],
    "catpop": ["cat_pop", "cat_ctr", "sub_pop", "sub_ctr"],
    "cf": ["cf_score"],
    "profile": ["cat_match_frac", "subcat_match_frac", "hist_len",
                "hist_uniq_cats", "hist_top_cat_frac"],
    "hour": ["hour"],
}


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
    # NOTE: g_s holds ORIGINAL group ids, which are not contiguous when the
    # caller passes a subset of impressions. minlength must therefore be the
    # max id + 1, not len(ug).
    n_grp = int(g_s.max()) + 1
    n_pos = np.bincount(g_s[pos], minlength=n_grp).astype(float)
    n_neg = np.bincount(g_s[~pos], minlength=n_grp).astype(float)
    rsum = np.bincount(g_s[pos], weights=ranks[pos], minlength=n_grp)
    valid = (n_pos > 0) & (n_neg > 0)
    return float(((rsum[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2) /
                  (n_pos[valid] * n_neg[valid])).mean()), int(valid.sum())


def mask_rows_to_coverage(hit, target, rng):
    total = len(hit)
    want = int(target * total)
    idx = np.where(hit)[0]
    rng.shuffle(idx)
    m = np.zeros(total, dtype=bool)
    m[idx[:want]] = True
    return m


def joint_mask(nids, train_hit, dev_hit, rng):
    """Assign each news to one of 4 cells matching the test row distribution.

    Returns (m_train, m_dev, cell_of_row) where cell_of_row is 0=both,
    1=train_only, 2=dev_only, 3=neither.
    """
    total = len(nids)
    wmap = Counter(nids)
    has_train, has_dev = {}, {}
    for n, t, d in zip(nids, train_hit, dev_hit):
        if t:
            has_train[n] = True
        if d:
            has_dev[n] = True
    quotas = {"both": int(TEST_BOTH * total),
              "tr_only": int(TEST_TR_ONLY * total),
              "dv_only": int(TEST_DV_ONLY * total),
              "neither": int(TEST_NEITHER * total)}
    remaining = dict(quotas)
    keys = list(wmap.keys())
    order = rng.permutation(len(keys))
    assign = {}
    for j in order:
        n = keys[j]
        t = has_train.get(n, False)
        d = has_dev.get(n, False)
        if t and d:
            feasible = ["both", "tr_only", "dv_only", "neither"]
        elif t:
            feasible = ["tr_only", "neither"]
        elif d:
            feasible = ["dv_only", "neither"]
        else:
            feasible = ["neither"]
        best = max(feasible, key=lambda c: (remaining[c], rng.random()))
        assign[n] = best
        remaining[best] -= wmap[n]
    m_train = np.zeros(total, dtype=bool)
    m_dev = np.zeros(total, dtype=bool)
    cell = np.empty(total, dtype=np.int8)
    cid = {"both": 0, "tr_only": 1, "dv_only": 2, "neither": 3}
    for i, n in enumerate(nids):
        c = assign[n]
        if c in ("both", "tr_only"):
            m_train[i] = True
        if c in ("both", "dv_only"):
            m_dev[i] = True
        cell[i] = cid[c]
    return m_train, m_dev, cell


def _avg_rank_grouped(v, g):
    n = len(v)
    o = np.lexsort((v, g))
    gs, vs = g[o], v[o]
    new_g = np.empty(n, dtype=bool)
    new_g[0] = True
    new_g[1:] = gs[1:] != gs[:-1]
    grp_start = np.flatnonzero(new_g)
    grp_end = np.append(grp_start[1:], n)
    grp_size = grp_end - grp_start
    tie = np.empty(n, dtype=bool)
    tie[0] = True
    tie[1:] = (gs[1:] != gs[:-1]) | (vs[1:] != vs[:-1])
    run_start = np.flatnonzero(tie)
    run_end = np.append(run_start[1:], n)
    run_len = run_end - run_start
    gidx = np.searchsorted(grp_start, run_start, side="right") - 1
    avg0 = (run_start - grp_start[gidx]) + (run_len - 1) / 2.0
    denom = np.maximum(grp_size[gidx] - 1, 1)
    out = np.empty(n, dtype=np.float32)
    out[o] = np.repeat(avg0 / denom, run_len)
    return out


def _zscore_grouped(v, g):
    cnt = np.bincount(g).astype(np.float64)
    s1 = np.bincount(g, weights=v.astype(np.float64))
    s2 = np.bincount(g, weights=(v.astype(np.float64) ** 2))
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return ((v - mean[g]) / (np.sqrt(var[g]) + 1e-6)).astype(np.float32)


def within_imp_transforms(X, g):
    out = np.zeros((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = _avg_rank_grouped(X[:, 0], g)
    out[:, 1] = _zscore_grouped(X[:, 0], g)
    out[:, 2] = _avg_rank_grouped(X[:, 1], g)
    out[:, 3] = _zscore_grouped(X[:, 1], g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--only-b", action="store_true",
                    help="skip the group-ablation loop, run only part B")
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
    cut = int(n_imp * 0.8)
    va = g >= cut
    tr = g < cut
    yv, gv = y[va], g[va]
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps, cut={cut}", flush=True)

    z = np.load(CACHE)
    C = {k: z[k] for k in z.files}
    z2 = np.load(CACHE2)
    S = {k: z2[k] for k in z2.files}

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    ms = np.maximum(S["sub_asof"], 1.0)
    DEVPOP = np.column_stack([
        C["causal_pop"] / mp,
        np.where(C["causal_pop"] > 0,
                 C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
        np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    CAT = np.column_stack([
        C["causal_cat_pop"] / mc,
        np.where(C["causal_cat_pop"] > 0,
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    ]).astype(np.float32)
    SUBCAT = np.column_stack([
        S["sub_pop"] / ms,
        np.where(S["sub_pop"] > 0, S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)
    ]).astype(np.float32)

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["causal_pop"] > 0

    rng_j = np.random.default_rng(4000)
    m_tr_j, m_dv_j, _ = joint_mask(nids, train_hit, dev_hit, rng_j)
    m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
    rng_e = np.random.default_rng(2000)
    m_tr_e, m_dv_e, cell_e = joint_mask(nids, train_hit, dev_hit, rng_e)
    m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

    Xtr = X.copy()
    Xtr[~m_tr_j, 2:5] = 0.0
    Xtr[~m_cf_j, 1] = 0.0
    dptr = DEVPOP.copy()
    dptr[~m_dv_j] = 0.0
    Xf = np.hstack([Xtr, dptr, CAT, SUBCAT, within_imp_transforms(Xtr, g)])
    del Xtr

    Xev = X.copy()
    Xev[~m_tr_e, 2:5] = 0.0
    Xev[~m_cf_e, 1] = 0.0
    dpe = DEVPOP.copy()
    dpe[~m_dv_e] = 0.0
    Xe = np.hstack([Xev, dpe, CAT, SUBCAT, within_imp_transforms(Xev, g)])
    del Xev

    assert Xf.shape[1] == len(ALL_FEATS), (Xf.shape, len(ALL_FEATS))
    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)

    def fit_eval(cols, name):
        """Train on the given feature columns, return (auc, gain, preds)."""
        idx = [ALL_FEATS.index(c) for c in cols]
        dtr = lgb.Dataset(Xf[tr][:, idx], label=y[tr])
        bst = lgb.train(params, dtr, num_boost_round=args.rounds)
        Xva = Xe[va][:, idx]
        p = bst.predict(Xva)
        a, n = per_imp(yv, p, gv)
        al, _ = per_imp(yv, Xva[:, 0], gv)
        print(f"  {name:<28} AUC={a:.4f}  gain={a-al:+.4f}  (n={n})", flush=True)
        return a, a - al, p

    print("\n=== A. FEATURE-GROUP ABLATION (drop one group, retrain) ===",
          flush=True)
    a_full, g_full, p_full = fit_eval(ALL_FEATS, "FULL (all 25)")
    results = {"FULL": (a_full, g_full)}
    if not args.only_b:
        for gname, cols in GROUPS.items():
            keep = [c for c in ALL_FEATS if c not in cols]
            a, g, _ = fit_eval(keep, f"-{gname}")
            results[f"-{gname}"] = (a, g)

        print("\n=== ablation summary (drop = how much the group was worth) ===")
        print(f"{'group':<14} {'AUC':>9} {'gain':>9} {'drop':>9}")
        for gname in GROUPS:
            a, g = results[f"-{gname}"]
            print(f"{gname:<14} {a:>9.4f} {g:>+9.4f} {g_full-g:>+9.4f}")

    # ---- B. gain by mask cell ----
    print("\n=== B. GAIN BY MASK CELL (eval mask, seed 2000) ===")
    print(f"{'cell':<12} {'rows':>12} {'share':>7} {'llm AUC':>9} "
          f"{'stack AUC':>10} {'gain':>9}")
    names = ["both", "train_only", "dev_only", "neither"]
    Xva_full = Xe[va]
    llm_p = Xva_full[:, 0]
    for c in range(4):
        sel = cell_e[va] == c
        if not sel.any():
            continue
        ab, nb_ = per_imp(yv[sel], p_full[sel], gv[sel])
        al, _ = per_imp(yv[sel], llm_p[sel], gv[sel])
        print(f"{names[c]:<12} {sel.sum():>12,} {sel.mean():>6.1%} "
              f"{al:>9.4f} {ab:>10.4f} {ab-al:>+9.4f}  (n_imp={nb_:,})")

    # The decisive number: the "neither" cell is the 83.3% test regime.
    sel = cell_e[va] == 3
    ab, nb_ = per_imp(yv[sel], p_full[sel], gv[sel])
    al, _ = per_imp(yv[sel], llm_p[sel], gv[sel])
    print(f"\n  TEST-REGIME proxy (neither cell, {sel.mean():.1%} of val rows):")
    print(f"    llm={al:.4f}  stack={ab:.4f}  gain={ab-al:+.4f}")
    if ab - al > 0.02:
        print("    => the gain SURVIVES with NO popularity features: it is real.")
    else:
        print("    => the gain COLLAPSES without popularity features: it will "
              "NOT transfer to the 83.3% cold test rows.")

    with open("logs/exp_ranker_ablate.json", "w") as f:
        json.dump({k: list(v) for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
