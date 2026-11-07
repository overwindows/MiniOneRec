#!/usr/bin/env python3
"""User-level dev-history features: a NEW signal that transfers to test.

WHY
---
The feature-group ablation showed the stack ranker is ~98% popularity/profile
driven: dropping `llm_score` costs only 0.0013 AUC. So the LLM is already
carrying almost no weight, and the only way to move the number is to add
information the stack does not have.

There is one such channel that we have never used: THE USER'S OWN HISTORY IN
THE PREVIOUS PERIOD.

    43.78% of TEST impressions belong to a user who also appears in dev.
    46.41% of dev[80:100%] impressions belong to a user in dev[0:80%].

Those two rates are nearly identical, so a feature built this way is
measured under almost exactly the test conditions -- it is NOT inflated by
the dev split.

For such an impression we know, from the previous period, which news the
user actually CLICKED. That gives:
    uhit        : 1 if this candidate is a news the user clicked before
    uhit_cat    : fraction of the user's previous clicks in this category
    uhit_sub    : fraction of the user's previous clicks in this subcategory
    uhit_n      : how many previous clicks the user has (log1p)
    uhit_present: 1 if the user has any previous click at all

At test time the "previous period" is ALL of dev; at validation time it is
dev[0:80%]. The construction is identical.

Usage: python src/exp_ranker_userhist.py [--rounds 2000] [--threads 32]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

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

UH_FEATS = ["uhit", "uhit_cat", "uhit_sub", "uhit_n", "uhit_present"]


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


def build_userhist(n_rows, cut_imp):
    """User -> clicked news / category counts, from PRIOR impressions only.

    CAUSAL WALK (not a frozen state):
      - a TRAINING row at impression i sees the user's clicks from
        impressions < i. Its own impression's clicks are NOT in the state,
        so the row's own label cannot leak.
      - an EVAL row (impression >= cut_imp) sees the user's clicks from
        ALL training-window impressions (the state is frozen at the cut),
        which is exactly what a test row sees (all of dev precedes test).

    A leave-one-impression-out state would be WRONG here: for a training
    row, the LOO state still contains the OTHER candidates of the SAME
    impression, so "uhit=1" would mean "some sibling candidate was
    clicked" -- a within-impression label leak that does not exist at test
    time (a test impression's clicks are unknown).

    Returns a (n_rows, 5) float32 array aligned with the dev feature table.
    """
    news_cat, news_sub = {}, {}
    for path in (NEWS, "data/MIND_large/dev/news.tsv"):
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    news_cat.setdefault(p[0], p[1])
                    news_sub.setdefault(p[0], p[2])

    u_clicked = defaultdict(set)
    u_cats = defaultdict(Counter)
    u_subs = defaultdict(Counter)
    u_n = Counter()

    out = np.zeros((n_rows, 5), dtype=np.float32)
    row = 0
    imp = 0
    with open(DEV_BEHAVIORS) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            uid = parts[1]
            cands = []
            for tk in parts[4].split():
                if "-" not in tk:
                    continue
                nid, lab = tk.rsplit("-", 1)
                cands.append((nid, lab))

            # the state used for THIS impression
            clicked = u_clicked.get(uid)
            if clicked:
                cc = u_cats.get(uid, Counter())
                cs = u_subs.get(uid, Counter())
                n = u_n.get(uid, 0)
                tot_c = sum(cc.values()) or 1
                tot_s = sum(cs.values()) or 1
                for nid, lab in cands:
                    out[row, 0] = 1.0 if nid in clicked else 0.0
                    out[row, 1] = cc.get(news_cat.get(nid, ""), 0) / tot_c
                    out[row, 2] = cs.get(news_sub.get(nid, ""), 0) / tot_s
                    out[row, 3] = np.log1p(n)
                    out[row, 4] = 1.0
                    row += 1
            else:
                for nid, lab in cands:
                    out[row, 4] = 0.0
                    row += 1

            # fold this impression's clicks into the state -- but ONLY if the
            # impression is inside the training window. After the cut the
            # state is frozen (a test row never sees test clicks).
            if imp < cut_imp:
                for nid, lab in cands:
                    if lab != "1":
                        continue
                    u_clicked[uid].add(nid)
                    u_n[uid] += 1
                    c = news_cat.get(nid, "")
                    s = news_sub.get(nid, "")
                    u_cats[uid][c] += 1
                    u_subs[uid][s] += 1
            imp += 1
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"
    return out


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
    cut = int(n_imp * 0.8)
    print(f"  {X.shape[0]:,} rows, {n_imp:,} imps, cut={cut}", flush=True)

    with open(DEV_FEATS) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(DEV_FEATS) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    print("Building user-history features (from dev[0:80%]) ...", flush=True)
    UH = build_userhist(X.shape[0], cut)
    va = g >= cut
    print(f"  val rows with a user history: {UH[va, 4].mean():.2%}")
    print(f"  val rows where the candidate was a PREVIOUS CLICK: "
          f"{UH[va, 0].mean():.3%}", flush=True)

    z = np.load(CACHE)
    C = {"pop": z["causal_pop"], "clicks": z["causal_clicks"],
         "cat_pop": z["causal_cat_pop"], "cat_clicks": z["causal_cat_clicks"],
         "asof_pop": z["asof_max_pop"], "asof_cat": z["asof_max_cat"]}
    z2 = np.load(CACHE2)
    C["sub_pop"] = z2["sub_pop"]
    C["sub_clk"] = z2["sub_clk"]
    C["sub_asof"] = z2["sub_asof"]

    train_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    dev_hit = C["pop"] > 0

    rng_j = np.random.default_rng(4000)
    m_tr_j, m_dv_j, _ = joint_mask(nids, train_hit, dev_hit, rng_j)
    m_cf_j = mask_rows_to_coverage(cf_hit, COV_CF, rng_j)
    rng_e = np.random.default_rng(2000)
    m_tr_e, m_dv_e, cell_e = joint_mask(nids, train_hit, dev_hit, rng_e)
    m_cf_e = mask_rows_to_coverage(cf_hit, COV_CF, rng_e)

    def blocks(m_dev, m_train, m_cf):
        Xm = X.copy()
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

    Xf = blocks(m_dv_j, m_tr_j, m_cf_j)
    Xe = blocks(m_dv_e, m_tr_e, m_cf_e)
    assert Xf.shape[1] == len(ALL_FEATS), (Xf.shape, len(ALL_FEATS))

    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)

    tr = g < cut
    yv, gv = y[va], g[va]

    def fit_eval(Xtr, Xev, name):
        dtr = lgb.Dataset(Xtr[tr], label=y[tr])
        bst = lgb.train(params, dtr, num_boost_round=args.rounds)
        Xva = Xev[va]
        p = bst.predict(Xva)
        a, n = per_imp(yv, p, gv)
        al, _ = per_imp(yv, Xva[:, 0], gv)
        # no-popularity slice (the test regime)
        nopop = (Xva[:, 2] == 0) & (Xva[:, 1] == 0) & (Xva[:, 14] == 0)
        an, _ = per_imp(yv[nopop], p[nopop], gv[nopop])
        aln, _ = per_imp(yv[nopop], Xva[nopop, 0], gv[nopop])
        print(f"  {name:<34} val={a:.4f} gain={a-al:+.4f}  "
              f"nopop={an:.4f} gain={an-aln:+.4f} (n={n:,})", flush=True)
        return a, a - al, an - aln, p

    print("\n=== user-history features ===", flush=True)
    a0, g0, n0, p0 = fit_eval(Xf, Xe, "BASE (25 feats)")
    Xf_u = np.hstack([Xf, UH])
    Xe_u = np.hstack([Xe, UH])
    a1, g1, n1, p1 = fit_eval(Xf_u, Xe_u, "BASE + userhist (30 feats)")

    # user-history ONLY, to see how much signal it carries alone
    Xf_uo = np.hstack([Xf[:, [0]], UH])
    Xe_uo = np.hstack([Xe[:, [0]], UH])
    fit_eval(Xf_uo, Xe_uo, "llm + userhist only (6 feats)")

    print("\n=== gain by mask cell (with userhist) ===")
    names = ["both", "train_only", "dev_only", "neither"]
    Xva = Xe_u[va]
    for c in range(4):
        sel = cell_e[va] == c
        if not sel.any():
            continue
        ab, _ = per_imp(yv[sel], p1[sel], gv[sel])
        alb, _ = per_imp(yv[sel], Xva[sel, 0], gv[sel])
        print(f"  {names[c]:<12} {sel.mean():>6.1%} llm={alb:.4f} "
              f"stack={ab:.4f} gain={ab-alb:+.4f}")

    print(f"\n=== VERDICT ===")
    print(f"  base      val={a0:.4f} gain={g0:+.4f}  nopop gain={n0:+.4f}")
    print(f"  +userhist val={a1:.4f} gain={g1:+.4f}  nopop gain={n1:+.4f}")
    print(f"  delta     val={a1-a0:+.4f}  nopop={n1-n0:+.4f}")
    if a1 - a0 > 0.002:
        print("  => user-history features ADD real signal: build them for test.")
    else:
        print("  => user-history features add nothing: the LLM already "
              "captures this signal.")

    with open("logs/exp_ranker_userhist.json", "w") as f:
        json.dump({"base": [a0, g0, n0], "userhist": [a1, g1, n1]}, f, indent=2)


if __name__ == "__main__":
    main()
