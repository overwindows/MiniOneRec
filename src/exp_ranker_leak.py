#!/usr/bin/env python3
"""Leak test: is the val AUC inflated by the val period's own clicks?

WHY
---
The stack ranker claims +0.0575 per-impression AUC over the LLM on dev.
The previous (l13) ranker claimed +0.0105 on dev and transferred as ~0.000
at test. Before submitting we must know whether the stack's dev number is
honest.

THE ASYMMETRY
-------------
The causal popularity features are built by walking dev in order, so a val
row's dev_pop / dev_ctr include the clicks of the val impressions that came
BEFORE it. A news that is popular during the val period gets its dev_pop
boosted for later val rows -- and that boost correlates with the val row's
own label.

At TEST time this cannot happen: the test feature table's dev_pop is built
from dev/behaviors.tsv ONLY (append_devpop_cat_test.py --dev_behaviors), so
a test row never sees another test row's click.

THE EXACT TEST-EQUIVALENT
-------------------------
A test row's dev_pop = popularity over ALL of dev, and the test impression
is not in dev, so there is nothing to subtract. The dev row that matches
this exactly is the LEAVE-ONE-IMPRESSION-OUT full-dev popularity:

    loo_pop(nid)    = full_dev_pop(nid)    - 1   (this impression showed nid)
    loo_clicks(nid) = full_dev_clicks(nid) - label (this impression clicked nid)
    asof_max        = full-dev max          (a constant, as at test time)

If the val gain survives under LOO-full-dev, the dev number is honest and
the stack will transfer. If it collapses, the dev number is a leak.

Usage: python src/exp_ranker_leak.py [--rounds 2000] [--threads 32]
"""
import argparse
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

TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310
COV_CF = 0.113


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
    for i, n in enumerate(nids):
        c = assign[n]
        if c in ("both", "tr_only"):
            m_train[i] = True
        if c in ("both", "dv_only"):
            m_dev[i] = True
    return m_train, m_dev


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


def build_causal(n_rows):
    """LIVE causal popularity (as the trainer builds it) + the full-dev totals.

    The full-dev totals are what the TEST table uses, so they give the
    leave-one-impression-out variant for free.
    """
    news_cat = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                news_cat[p[0]] = p[1]

    run_pop, run_clicks = Counter(), Counter()
    run_cat_pop, run_cat_clicks = Counter(), Counter()
    live = {k: np.zeros(n_rows, dtype=np.float32)
            for k in ("pop", "clicks", "cat_pop", "cat_clicks",
                      "asof_pop", "asof_cat")}
    # per-row: this impression's own contribution, for the LOO subtraction
    own_pop = np.zeros(n_rows, dtype=np.float32)
    own_clicks = np.zeros(n_rows, dtype=np.float32)
    own_cat_pop = np.zeros(n_rows, dtype=np.float32)
    own_cat_clicks = np.zeros(n_rows, dtype=np.float32)

    row = 0
    max_pop = max_cat = 0
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
                live["pop"][row] = run_pop.get(nid, 0)
                live["clicks"][row] = run_clicks.get(nid, 0)
                live["cat_pop"][row] = run_cat_pop.get(cat, 0)
                live["cat_clicks"][row] = run_cat_clicks.get(cat, 0)
                live["asof_pop"][row] = max_pop
                live["asof_cat"][row] = max_cat
                # this impression's own contribution (always 1 impression)
                own_pop[row] = 1.0
                own_clicks[row] = 1.0 if lab == "1" else 0.0
                own_cat_pop[row] = 1.0
                own_cat_clicks[row] = 1.0 if lab == "1" else 0.0
                row += 1
            for nid, lab in cands:
                run_pop[nid] += 1
                if run_pop[nid] > max_pop:
                    max_pop = run_pop[nid]
                cat = news_cat.get(nid, "")
                run_cat_pop[cat] += 1
                if run_cat_pop[cat] > max_cat:
                    max_cat = run_cat_pop[cat]
                if lab == "1":
                    run_clicks[nid] += 1
                    run_cat_clicks[cat] += 1
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"

    # full-dev totals == what the TEST table's dev_pop is built from
    full = {
        "pop": np.array([run_pop.get(n, 0) for n in _nids], dtype=np.float32),
        "clicks": np.array([run_clicks.get(n, 0) for n in _nids], dtype=np.float32),
        "cat_pop": np.array([run_cat_pop.get(c, 0) for c in _cats],
                            dtype=np.float32),
        "cat_clicks": np.array([run_cat_clicks.get(c, 0) for c in _cats],
                               dtype=np.float32),
        "asof_pop": np.float32(max_pop),
        "asof_cat": np.float32(max_cat),
    }
    return live, full, own_pop, own_clicks, own_cat_pop, own_cat_clicks


_nids = []
_cats = []


def main():
    global _nids, _cats
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
    cut = int(n_imp * 0.8)
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps, cut={cut}", flush=True)

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    _nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            _nids.append(line.rstrip("\n").split("\t")[i_nid])
    news_cat = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                news_cat[p[0]] = p[1]
    _cats = [news_cat.get(n, "") for n in _nids]

    print("Building causal features (live + full-dev totals) ...", flush=True)
    live, full, own_pop, own_clicks, own_cat_pop, own_cat_clicks = \
        build_causal(X.shape[0])

    va = g >= cut
    tr = g < cut
    yv, gv = y[va], g[va]

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0

    # ---- LOO full-dev: exactly the test semantics ----
    loo = {
        "pop": np.maximum(full["pop"] - own_pop, 0.0),
        "clicks": np.maximum(full["clicks"] - own_clicks, 0.0),
        "cat_pop": np.maximum(full["cat_pop"] - own_cat_pop, 0.0),
        "cat_clicks": np.maximum(full["cat_clicks"] - own_cat_clicks, 0.0),
        "asof_pop": np.full(X.shape[0], full["asof_pop"], dtype=np.float32),
        "asof_cat": np.full(X.shape[0], full["asof_cat"], dtype=np.float32),
    }

    def make_blocks(C):
        mp = np.maximum(C["asof_pop"], 1.0)
        mc = np.maximum(C["asof_cat"], 1.0)
        dp = np.column_stack([
            C["pop"] / mp,
            np.where(C["pop"] > 0, C["clicks"] / np.maximum(C["pop"], 1), 0.0),
            np.log1p(C["pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
        cp = np.column_stack([
            C["cat_pop"] / mc,
            np.where(C["cat_pop"] > 0,
                     C["cat_clicks"] / np.maximum(C["cat_pop"], 1), 0.0)
        ]).astype(np.float32)
        return dp, cp

    DP_L, CP_L = make_blocks(live)
    DP_F, CP_F = make_blocks(loo)

    def run_variant(name, DP, CP, hit, train_DP=None, train_CP=None):
        """Train on dev-80% with (train_DP or DP); evaluate on dev-20% with DP."""
        tDP = DP if train_DP is None else train_DP
        tCP = CP if train_CP is None else train_CP
        rng_j = np.random.default_rng(4000)
        m_tr_j, m_dv_j = joint_mask(_nids, train_hit, hit, rng_j)
        m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
        rng_e = np.random.default_rng(2000)
        m_tr_e, m_dv_e = joint_mask(_nids, train_hit, hit, rng_e)
        m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

        Xtr = X.copy()
        Xtr[~m_tr_j, 2:5] = 0.0
        Xtr[~m_cf_j, 1] = 0.0
        dptr = tDP.copy()
        dptr[~m_dv_j] = 0.0
        Xf = np.hstack([Xtr, dptr, tCP, within_imp_transforms(Xtr, g)])
        del Xtr

        Xev = X.copy()
        Xev[~m_tr_e, 2:5] = 0.0
        Xev[~m_cf_e, 1] = 0.0
        dpe = DP.copy()
        dpe[~m_dv_e] = 0.0
        Xe = np.hstack([Xev, dpe, CP, within_imp_transforms(Xev, g)])
        del Xev

        params = dict(objective="binary", metric="auc", learning_rate=0.05,
                      num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      num_threads=args.threads, verbose=-1)
        dtr = lgb.Dataset(Xf[tr], label=y[tr])
        bst = lgb.train(params, dtr, num_boost_round=args.rounds)
        Xva = Xe[va]
        p = bst.predict(Xva)
        llm_p = Xva[:, 0]
        a, n = per_imp(yv, p, gv)
        al, _ = per_imp(yv, llm_p, gv)
        print(f"\n=== {name} ===")
        print(f"  llm_score only   AUC={al:.4f}")
        print(f"  stack            AUC={a:.4f}   gain={a-al:+.4f}  (n={n})")
        return a, al

    hit_live = live["pop"] > 0
    hit_loo = loo["pop"] > 0
    print(f"\ndev-pop hit rate: live={hit_live.mean():.1%}  "
          f"loo-full-dev={hit_loo.mean():.1%}", flush=True)

    # A: the trainer's own setup (live causal features everywhere)
    a_live, al = run_variant(
        "A LIVE causal features (what the trainer uses)",
        DP_L, CP_L, hit_live)

    # B: TEST-EQUIVALENT. Train on live causal features (as production does),
    #    but evaluate the val rows with LOO-full-dev popularity -- exactly
    #    what a test row sees.
    a_loo, _ = run_variant(
        "B LOO-full-dev popularity on the val rows (TEST semantics)",
        DP_F, CP_F, hit_loo, train_DP=DP_L, train_CP=CP_L)

    print(f"\n=== VERDICT ===")
    print(f"  live   gain over llm: {a_live-al:+.4f}")
    print(f"  LOO    gain over llm: {a_loo-al:+.4f}")
    print(f"  leak cost (live - LOO): {a_live-a_loo:+.4f}")
    if a_loo - al > 0.02:
        print("  => the gain SURVIVES the leak test: it is real and should "
              "transfer to test.")
    else:
        print("  => the gain COLLAPSES under test semantics: the dev number "
              "is largely a leak and will NOT transfer.")


if __name__ == "__main__":
    main()
