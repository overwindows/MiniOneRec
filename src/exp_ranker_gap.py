#!/usr/bin/env python3
"""Does the stack ranker's +0.059 dev gain survive a TEMPORAL GAP?

WHY
---
We have three independent proofs that the stack's dev gain is not a leak:
  (1) leak test (exp_ranker_leak.py): leave-one-impression-out full-dev
      popularity costs only -0.0002 AUC;
  (2) time-block stability: +0.0587 / +0.0599 / +0.0597 / +0.0578 across
      four dev time blocks;
  (3) mask-cell analysis (exp_ranker_ablate.py part B): the "neither" cell
      (83.4% of val rows = the test regime) still shows +0.0568.

But all three share ONE assumption: the popularity features for the eval
rows are built from a window that OVERLAPS the training window. At test
time the popularity window (all of dev) and the eval window (test) are
DISJOINT. If the gain comes from the popularity features "remembering" the
training window's clicks in a way that only works when the windows overlap,
it will not transfer.

THE TEST-EQUIVALENT
-------------------
    train on dev[0:60%], evaluate on dev[60:100%]
    popularity for the eval rows = FIXED at the end-of-window state
                                   (exactly what a test row sees: the
                                    full-dev state, frozen, never updated
                                    by test clicks)

The "LIVE" variant instead lets the popularity walk continue through the
eval window (each eval row sees the clicks of the eval impressions before
it). LIVE - FIXED = the leak cost across the gap.

We also slice the eval rows by USER OVERLAP: does the eval impression's
user appear in the training window? 31% of test users appear in dev, so
user-level correlation is a REAL transfer channel at test time -- but only
at the test rate. If the dev gain is concentrated in overlapping users and
the dev overlap rate is much higher than 31%, the dev number is inflated.

Usage: python src/exp_ranker_gap.py [--rounds 2000] [--threads 32]
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
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"

TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310
COV_CF = 0.113

ALL_FEATS = ["llm_score", "cf_score", "cand_pop", "cand_ctr", "cand_log_pop",
             "cat_match_frac", "subcat_match_frac", "hist_len",
             "hist_uniq_cats", "hist_top_cat_frac", "imp_size", "position",
             "position_frac", "hour", "dev_pop", "dev_ctr", "dev_log_pop",
             "cat_pop", "cat_ctr", "sub_pop", "sub_ctr", "llm_rank", "llm_z",
             "cf_rank", "cf_z"]


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
    """Mean per-impression AUC. gg may hold non-contiguous group ids."""
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
    """Assign each news to one of 4 cells matching the test row distribution."""
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


def build_causal_window(n_rows, cut_imp):
    """Walk dev; the popularity state FREEZES after impression `cut_imp`.

    Rows of impressions < cut_imp see the causal (past-only) state.
    Rows of impressions >= cut_imp see the FROZEN end-of-window state --
    exactly what a test row sees (the full-dev state, never updated by
    test clicks).

    Returns dict with pop/clicks/cat_pop/cat_clicks/asof_max_pop/asof_max_cat.
    """
    news_cat = {}
    news_sub = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                news_cat[p[0]] = p[1]
                news_sub[p[0]] = p[2]

    run_pop, run_clicks = Counter(), Counter()
    run_cat_pop, run_cat_clicks = Counter(), Counter()
    run_sub_pop, run_sub_clicks = Counter(), Counter()
    out = {k: np.zeros(n_rows, dtype=np.float32)
           for k in ("pop", "clicks", "cat_pop", "cat_clicks",
                     "asof_pop", "asof_cat", "sub_pop", "sub_clk",
                     "sub_asof")}
    row = 0
    max_pop = max_cat = max_sub = 0
    imp = 0
    with open(DEV_BEHAVIORS) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            cands = []
            for tk in parts[4].split():
                if "-" not in tk:
                    continue
                nid, lab = tk.rsplit("-", 1)
                cands.append((nid, lab))
            for nid, lab in cands:
                cat = news_cat.get(nid, "")
                sub = news_sub.get(nid, "")
                out["pop"][row] = run_pop.get(nid, 0)
                out["clicks"][row] = run_clicks.get(nid, 0)
                out["cat_pop"][row] = run_cat_pop.get(cat, 0)
                out["cat_clicks"][row] = run_cat_clicks.get(cat, 0)
                out["asof_pop"][row] = max_pop
                out["asof_cat"][row] = max_cat
                out["sub_pop"][row] = run_sub_pop.get(sub, 0)
                out["sub_clk"][row] = run_sub_clicks.get(sub, 0)
                out["sub_asof"][row] = max_sub
                row += 1
            # only fold this impression into the state if it is INSIDE the
            # training window; after the cut the state is frozen.
            if imp < cut_imp:
                for nid, lab in cands:
                    run_pop[nid] += 1
                    if run_pop[nid] > max_pop:
                        max_pop = run_pop[nid]
                    cat = news_cat.get(nid, "")
                    run_cat_pop[cat] += 1
                    if run_cat_pop[cat] > max_cat:
                        max_cat = run_cat_pop[cat]
                    sub = news_sub.get(nid, "")
                    run_sub_pop[sub] += 1
                    if run_sub_pop[sub] > max_sub:
                        max_sub = run_sub_pop[sub]
                    if lab == "1":
                        run_clicks[nid] += 1
                        run_cat_clicks[cat] += 1
                        run_sub_clicks[sub] += 1
            imp += 1
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"
    out["max_pop"] = np.float32(max_pop)
    out["max_cat"] = np.float32(max_cat)
    out["sub_max"] = np.float32(max_sub)
    return out


def read_users(path):
    """One user id per impression, in behaviors order."""
    users = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                users.append(p[1])
    return users


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
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

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    users = read_users(DEV_BEHAVIORS)
    assert len(users) == n_imp, (len(users), n_imp)

    z = np.load(CACHE)
    C_full = {"pop": z["causal_pop"], "clicks": z["causal_clicks"],
              "cat_pop": z["causal_cat_pop"], "cat_clicks": z["causal_cat_clicks"],
              "asof_pop": z["asof_max_pop"], "asof_cat": z["asof_max_cat"]}
    z2 = np.load(CACHE2)
    C_full["sub_pop"] = z2["sub_pop"]
    C_full["sub_clk"] = z2["sub_clk"]
    C_full["sub_asof"] = z2["sub_asof"]

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0

    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)

    def blocks(C, m_dev, m_train, m_cf, Xbase):
        """Assemble the 25-column feature matrix from a causal dict."""
        Xm = Xbase.copy()
        Xm[~m_train, 2:5] = 0.0
        Xm[~m_cf, 1] = 0.0
        mp = np.maximum(C["asof_pop"], 1.0)
        mc = np.maximum(C["asof_cat"], 1.0)
        dp = np.column_stack([
            C["pop"] / mp,
            np.where(C["pop"] > 0, C["clicks"] / np.maximum(C["pop"], 1), 0.0),
            np.log1p(C["pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
        dp[~m_dev] = 0.0
        cp = np.column_stack([
            C["cat_pop"] / mc,
            np.where(C["cat_pop"] > 0,
                     C["cat_clicks"] / np.maximum(C["cat_pop"], 1), 0.0)
        ]).astype(np.float32)
        ms = np.maximum(C["sub_asof"], 1.0)
        sp = np.column_stack([
            C["sub_pop"] / ms,
            np.where(C["sub_pop"] > 0,
                     C["sub_clk"] / np.maximum(C["sub_pop"], 1), 0.0)
        ]).astype(np.float32)
        return np.hstack([Xm, dp, cp, sp, within_imp_transforms(Xm, g)])

    def run_variant(name, cut_frac, eval_live):
        """cut_frac: train on dev[0:cut_frac]. eval_live: let the popularity
        walk continue through the eval window (leaky) or freeze it."""
        cut = int(n_imp * cut_frac)
        print(f"\n=== {name}: train dev[0:{cut_frac:.0%}] "
              f"({cut:,} imps), eval dev[{cut_frac:.0%}:1] "
              f"({n_imp-cut:,} imps), eval_pop={'LIVE' if eval_live else 'FIXED'} ===",
              flush=True)

        print("  building windowed causal features ...", flush=True)
        Cw = build_causal_window(X.shape[0], cut)

        # masks: train rows use the window's dev-pop hit; eval rows use the
        # same hit definition so the joint distribution matches test.
        dev_hit = Cw["pop"] > 0
        rng_j = np.random.default_rng(4000)
        m_tr_j, m_dv_j, _ = joint_mask(nids, train_hit, dev_hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
        rng_e = np.random.default_rng(2000)
        m_tr_e, m_dv_e, cell_e = joint_mask(nids, train_hit, dev_hit, rng_e)
        m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

        Xf = blocks(Cw, m_dv_j, m_tr_j, m_cf_j, X)
        if eval_live:
            # eval rows get the LIVE causal features (walk continues)
            Ce = C_full
        else:
            Ce = Cw
        Xe = blocks(Ce, m_dv_e, m_tr_e, m_cf_e, X)
        assert Xf.shape[1] == len(ALL_FEATS), (Xf.shape, len(ALL_FEATS))

        tr = g < cut
        va = g >= cut
        yv, gv = y[va], g[va]
        dtr = lgb.Dataset(Xf[tr], label=y[tr])
        bst = lgb.train(params, dtr, num_boost_round=args.rounds)
        Xva = Xe[va]
        p = bst.predict(Xva)
        a, n = per_imp(yv, p, gv)
        al, _ = per_imp(yv, Xva[:, 0], gv)
        print(f"  llm only   AUC={al:.4f}")
        print(f"  stack      AUC={a:.4f}   gain={a-al:+.4f}  (n_imp={n:,})",
              flush=True)

        # ---- user-overlap slice ----
        tr_users = set(users[:cut])
        ov = np.array([u in tr_users for u in users[cut:]])
        imp_ov = ov[gv - cut]
        print(f"  user overlap: {ov.mean():.1%} of eval impressions have a "
              f"user seen in the train window")
        for name2, sel in (("overlap", imp_ov), ("no-overlap", ~imp_ov)):
            if not sel.any():
                continue
            ab, nb = per_imp(yv[sel], p[sel], gv[sel])
            alb, _ = per_imp(yv[sel], Xva[sel, 0], gv[sel])
            print(f"    {name2:<11} n_imp={nb:>7,}  llm={alb:.4f}  "
                  f"stack={ab:.4f}  gain={ab-alb:+.4f}")

        # ---- test-regime cell slice ----
        names = ["both", "train_only", "dev_only", "neither"]
        print(f"  {'cell':<12} {'share':>7} {'llm':>8} {'stack':>8} {'gain':>8}")
        for c in range(4):
            sel = cell_e[va] == c
            if not sel.any():
                continue
            ab, _ = per_imp(yv[sel], p[sel], gv[sel])
            alb, _ = per_imp(yv[sel], Xva[sel, 0], gv[sel])
            print(f"  {names[c]:<12} {sel.mean():>6.1%} {alb:>8.4f} "
                  f"{ab:>8.4f} {ab-alb:>+8.4f}")

        del Xf, Xe
        return a - al, ov.mean()

    results = {}
    # A: the test-equivalent -- 40% temporal gap, popularity frozen.
    gA, ovA = run_variant("A GAP-40 FIXED (TEST semantics)", 0.60, False)
    results["A_gap40_fixed"] = {"gain": gA, "user_overlap": ovA}
    # B: same split but the popularity walk continues through the eval window.
    gB, _ = run_variant("B GAP-40 LIVE (leaky)", 0.60, True)
    results["B_gap40_live"] = {"gain": gB}
    # C: a 20% gap, closer to the dev-80/20 setup.
    gC, ovC = run_variant("C GAP-20 FIXED", 0.80, False)
    results["C_gap20_fixed"] = {"gain": gC, "user_overlap": ovC}

    print("\n=== VERDICT ===")
    print(f"  A  gap-40 FIXED (test semantics) gain = {gA:+.4f}")
    print(f"  B  gap-40 LIVE  (leaky)          gain = {gB:+.4f}")
    print(f"  C  gap-20 FIXED                  gain = {gC:+.4f}")
    print(f"  leak cost across the gap (B - A)  = {gB-gA:+.4f}")
    if gA > 0.02:
        print("  => the gain SURVIVES a 40% temporal gap with FROZEN "
              "popularity: it is real and should transfer to test.")
    else:
        print("  => the gain COLLAPSES across a temporal gap: the dev number "
              "is a temporal artifact and will NOT transfer.")
    print(f"  dev user-overlap rate = {ovA:.1%} (test users in dev = 31.0%)")

    with open("logs/exp_ranker_gap.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
