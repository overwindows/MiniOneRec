#!/usr/bin/env python3
"""Final production ranker: causal devpop + category popularity.

Recipe (selected by src/exp_ranker_causal.py + exp_ranker_causal2.py +
exp_ranker_final.py + exp_ranker_subcat.py under the honest test-regime
masking):

    BASE_FEATS (14) + causal devpop (3) + cat popularity (2) [+ optional]

Why causal devpop: the naive devpop for a training row included that row's
own impression, so a clicked news had its own click inflating the feature.
At test time the test impression is not in dev, so the feature is a clean
prior and the model over-trusts it. The causal (past-only) construction
matches test-time semantics exactly and is leak-free by construction.

Why category popularity: 83.3% of test rows reference a news that appears in
NEITHER train NOR dev, so news-level popularity is missing for the large
majority of test rows. Category popularity has ~100% test coverage and is
the only popularity signal that transfers.

Why TEST-REGIME MASKING: the dev split has far denser popularity coverage
than the test split (dev news pop 100% vs test 15.5%). A model trained on
the dense dev regime over-trusts the popularity features and collapses on
the sparse test regime. Each feature family is therefore masked down to its
MEASURED TEST COVERAGE on the training rows, so the model learns to work
in the sparsity it will actually face. This is what turns val 0.7751
(unmasked, optimistic) into val 0.7635 (masked, honest).

Why AS-OF normalisation: causal devpop is read from the impressions that
came BEFORE the row, so early dev rows see tiny counts while late dev rows
(and every test row) see large ones. Dividing by the max count observed SO
FAR makes the feature scale-free, and at test time the as-of max IS the
full-dev max, so the formula is identical at inference.

Training data: ALL dev impressions (376,471). The causal features are built
by walking dev in order, so each row sees only prior impressions -- exactly
what a test row sees (all of dev precedes test).

Usage:
    python src/train_ranker_causal_final.py \
        --out results/ranker/ranker_v2_lg9b_causal_devpop_cat.txt
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

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"
# separate path for the trainer's own subcat cache, so it never overwrites the
# short-named cache written by exp_ranker_subcat.py
CACHE2_TRAINER = "results/ranker/causal_subcat_trainer.npz"

BASE_FEATS = [
    "llm_score", "cf_score",
    "cand_pop", "cand_ctr", "cand_log_pop",
    "cat_match_frac", "subcat_match_frac",
    "hist_len", "hist_uniq_cats", "hist_top_cat_frac",
    "imp_size", "position", "position_frac",
    "hour",
]

# Measured TEST coverage of each feature family (fraction of test rows that
# have a non-zero value). The training rows are masked down to these rates.
COV_TRAINPOP = 0.128   # cand_pop > 0
COV_CF = 0.113         # cf_score != 0
COV_DEVPOP = 0.155     # dev-derived news pop > 0

# Measured TEST JOINT distribution of (train_pop>0, dev_pop>0) over all
# 93,115,001 test rows. The two popularity families are NOT independent at
# test time: a news seen in train is much more likely to also be seen in dev.
# Masking them independently reproduces the marginals but not the joint, so
# the model never sees the "both present" case at the test rate (11.6%).
TEST_BOTH = 0.11599
TEST_TR_ONLY = 0.01163
TEST_DV_ONLY = 0.03928
TEST_NEITHER = 0.83310

SEED = 13


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


def mask_news_to_coverage(nids, hit, target, rng):
    """Keep a random subset of the news that HAVE a value, so that the
    fraction of rows with a value equals `target` (the test coverage)."""
    total = len(nids)
    want = int(target * total)
    wmap = Counter()
    for n, h in zip(nids, hit):
        if h:
            wmap[n] += 1
    keys = list(wmap.keys())
    order = rng.permutation(len(keys))
    keep = set()
    cov = 0
    for j in order:
        n = keys[j]
        keep.add(n)
        cov += wmap[n]
        if cov >= want:
            break
    return np.array([n in keep for n in nids], dtype=bool)


def mask_rows_to_coverage(hit, target, rng):
    total = len(hit)
    want = int(target * total)
    idx = np.where(hit)[0]
    rng.shuffle(idx)
    m = np.zeros(total, dtype=bool)
    m[idx[:want]] = True
    return m


def joint_mask(nids, train_hit, dev_hit, rng):
    """Assign each news to one of 4 cells so the ROW joint distribution of
    (train_pop>0, dev_pop>0) matches the measured test distribution.

    Masking the two popularity families INDEPENDENTLY reproduces their
    marginals but not their joint: at test time 11.6% of rows have BOTH,
    while independent masking of the dev rows (94.6% have both) produces a
    very different joint. This function reproduces the test joint exactly.
    """
    total = len(nids)
    wmap = Counter(nids)
    has_train = {}
    has_dev = {}
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
    """Average (fractional) rank of v within each group g, normalized to 0..1.

    Plain `argsort().argsort()` breaks ties by POSITION, which injects pure
    noise: after test-regime masking, cf_score is 0 for 88.7% of rows, so
    most candidates in an impression tie at zero and would get an arbitrary
    0..m-1 ordering. Average rank gives every tied candidate the same value.
    """
    n = len(v)
    o = np.lexsort((v, g))
    gs = g[o]
    vs = v[o]

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
    avg_norm = avg0 / denom

    out = np.empty(n, dtype=np.float32)
    out[o] = np.repeat(avg_norm, run_len)
    return out


def _zscore_grouped(v, g):
    """Per-group z-score of v (population std), vectorized."""
    cnt = np.bincount(g).astype(np.float64)
    s1 = np.bincount(g, weights=v.astype(np.float64))
    s2 = np.bincount(g, weights=(v.astype(np.float64) ** 2))
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0.0)
    return ((v - mean[g]) / (np.sqrt(var[g]) + 1e-6)).astype(np.float32)


def within_imp_transforms(X, g):
    """Per-impression average-rank + z-score of llm_score and cf_score.

    Returns 4 columns: [llm_rank, llm_z, cf_rank, cf_z].

    MUST be computed from the MASKED matrix: at test time cf_score is
    genuinely absent for 88.7% of rows, so the within-impression z-score is
    taken over a mostly-zero column. Computing it from the unmasked matrix
    would train the model on a distribution it never sees at test.
    """
    out = np.zeros((X.shape[0], 4), dtype=np.float32)
    out[:, 0] = _avg_rank_grouped(X[:, 0], g)
    out[:, 1] = _zscore_grouped(X[:, 0], g)
    out[:, 2] = _avg_rank_grouped(X[:, 1], g)
    out[:, 3] = _zscore_grouped(X[:, 1], g)
    return out


POP_WIT_FEATS = ["cat_pop_rank", "cat_pop_z", "cat_ctr_rank", "cat_ctr_z",
                 "sub_pop_rank", "sub_pop_z", "sub_ctr_rank", "sub_ctr_z"]


def pop_wit_transforms(catpop, subpop, g):
    """Per-impression average-rank + z-score of the popularity features.

    Returns 8 columns in POP_WIT_FEATS order.

    WHY
    ---
    A tree splits on ABSOLUTE values, so it can learn "cat_pop > 0.3 is
    good" but it CANNOT learn "this candidate is the most popular category
    WITHIN this impression". The metric is per-impression AUC, so only the
    ORDER WITHIN an impression matters -- and the within-impression rank is
    exactly that quantity.

    cat_pop / sub_pop have 100% test coverage (vs 15.5% for news-level
    devpop), so these transforms are available on every row.

    Measured on the honest held-out dev split (3 seeds, joint test-regime
    mask, 2000 rounds): +0.0092 +/- 0.0001 over the llm/cf-only WIT arm.
    """
    out = np.zeros((catpop.shape[0], 8), dtype=np.float32)
    for j, v in enumerate([catpop[:, 0], catpop[:, 1],
                           subpop[:, 0], subpop[:, 1]]):
        out[:, 2 * j] = _avg_rank_grouped(v, g)
        out[:, 2 * j + 1] = _zscore_grouped(v, g)
    return out


def build_causal(n_rows, use_subcat):
    """Walk dev in order; each row sees only PRIOR impressions' popularity."""
    news_cat = {}
    news_sub = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                news_cat[p[0]] = p[1]
                news_sub[p[0]] = p[2]

    run_pop = Counter()
    run_clicks = Counter()
    run_cat_pop = Counter()
    run_cat_clicks = Counter()
    run_sub_pop = Counter()
    run_sub_clicks = Counter()

    causal_pop = np.zeros(n_rows, dtype=np.float32)
    causal_clicks = np.zeros(n_rows, dtype=np.float32)
    causal_cat_pop = np.zeros(n_rows, dtype=np.float32)
    causal_cat_clicks = np.zeros(n_rows, dtype=np.float32)
    asof_max_pop = np.zeros(n_rows, dtype=np.float32)
    asof_max_cat = np.zeros(n_rows, dtype=np.float32)
    causal_sub_pop = np.zeros(n_rows, dtype=np.float32)
    causal_sub_clicks = np.zeros(n_rows, dtype=np.float32)
    asof_max_sub = np.zeros(n_rows, dtype=np.float32)

    row = 0
    max_pop = 0
    max_cat = 0
    max_sub = 0
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
            # record the PRIOR state for this impression's candidates
            for nid, lab in cands:
                causal_pop[row] = run_pop.get(nid, 0)
                causal_clicks[row] = run_clicks.get(nid, 0)
                cat = news_cat.get(nid, "")
                causal_cat_pop[row] = run_cat_pop.get(cat, 0)
                causal_cat_clicks[row] = run_cat_clicks.get(cat, 0)
                asof_max_pop[row] = max_pop
                asof_max_cat[row] = max_cat
                if use_subcat:
                    sub = news_sub.get(nid, "")
                    causal_sub_pop[row] = run_sub_pop.get(sub, 0)
                    causal_sub_clicks[row] = run_sub_clicks.get(sub, 0)
                    asof_max_sub[row] = max_sub
                row += 1
            # then fold this impression into the running state
            for nid, lab in cands:
                run_pop[nid] += 1
                if run_pop[nid] > max_pop:
                    max_pop = run_pop[nid]
                cat = news_cat.get(nid, "")
                run_cat_pop[cat] += 1
                if run_cat_pop[cat] > max_cat:
                    max_cat = run_cat_pop[cat]
                if use_subcat:
                    sub = news_sub.get(nid, "")
                    run_sub_pop[sub] += 1
                    if run_sub_pop[sub] > max_sub:
                        max_sub = run_sub_pop[sub]
                if lab == "1":
                    run_clicks[nid] += 1
                    run_cat_clicks[cat] += 1
                    if use_subcat:
                        run_sub_clicks[sub] += 1
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"
    out = dict(causal_pop=causal_pop, causal_clicks=causal_clicks,
               causal_cat_pop=causal_cat_pop, causal_cat_clicks=causal_cat_clicks,
               asof_max_pop=asof_max_pop, asof_max_cat=asof_max_cat,
               max_pop=np.float32(max_pop), max_cat=np.float32(max_cat))
    if use_subcat:
        out.update(causal_sub_pop=causal_sub_pop,
                   causal_sub_clicks=causal_sub_clicks,
                   asof_max_sub=asof_max_sub, max_sub=np.float32(max_sub))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ranker/ranker_v2_lg9b_causal_devpop_cat.txt")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--subcat", action="store_true",
                    help="also add subcategory popularity (2 features)")
    ap.add_argument("--no-mask", action="store_true",
                    help="disable test-regime masking (diagnostic only)")
    ap.add_argument("--joint-mask", action="store_true",
                    help="mask train_pop/dev_pop to the measured TEST JOINT "
                         "distribution instead of their independent marginals")
    ap.add_argument("--wit", action="store_true",
                    help="add within-impression rank/z-score of llm_score and "
                         "cf_score (4 features)")
    ap.add_argument("--pop-wit", action="store_true",
                    help="add within-impression rank/z-score of cat_pop, "
                         "cat_ctr, sub_pop, sub_ctr (8 features). Requires "
                         "--subcat. Measured +0.0092 +/- 0.0001 on the honest "
                         "held-out dev split over the llm/cf-only WIT arm.")
    ap.add_argument("--dev-feats", default=DEV_FEATS,
                    help="dev feature table to train on. The default table was "
                         "built with a timestamp parser that ignored AM/PM, so "
                         "its `hour` column spans 1..12 while the TEST table "
                         "spans 0..23. Use the _v2 table to fix that.")
    ap.add_argument("--llm-dev", default=LG9B_DEV,
                    help="dev LLM score file (impression_id + per-candidate "
                         "scores). Point this at a different checkpoint's dev "
                         "scores to retrain the stack on a different LLM.")
    args = ap.parse_args()
    dev_feats = args.dev_feats
    llm_dev = args.llm_dev

    print(f"Loading dev features ({dev_feats}) ...", flush=True)
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(dev_feats)):
        Xs.append(Xb)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    del Xs, ys, gs
    ids = imp_ids(dev_feats)
    llm9 = load_llm(llm_dev)
    off = 0
    n_miss = 0
    for imp, n in ids:
        s = llm9.get(imp)
        if s and len(s) == n:
            X[off:off + n, 0] = s
        else:
            n_miss += 1
        off += n
    print(f"  {X.shape[0]:,} rows, {len(ids):,} imps, missing llm: {n_miss}", flush=True)

    print("Loading causal popularity arrays ...", flush=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        C = {k: z[k] for k in z.files}
    else:
        C = build_causal(X.shape[0], args.subcat)
        np.savez(CACHE, **C)
    if args.subcat:
        if os.path.exists(CACHE2_TRAINER):
            z2 = np.load(CACHE2_TRAINER)
            S = {k: z2[k] for k in z2.files}
        elif os.path.exists(CACHE2):
            z2 = np.load(CACHE2)
            S = {k: z2[k] for k in z2.files}
            # the cache may use the short names written by exp_ranker_subcat.py
            if "causal_sub_pop" not in S:
                S = {"causal_sub_pop": S["sub_pop"],
                     "causal_sub_clicks": S["sub_clk"],
                     "asof_max_sub": S["sub_asof"],
                     "max_sub": S["sub_max"]}
        else:
            S = build_causal(X.shape[0], True)
            np.savez(CACHE2_TRAINER, **S)
    else:
        S = None

    # ---- test-regime masking ----
    with open(dev_feats) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(dev_feats) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    rng = np.random.default_rng(SEED)
    trainpop_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    devpop_hit = C["causal_pop"] > 0
    if args.joint_mask:
        # Reproduce the measured TEST joint distribution of
        # (train_pop>0, dev_pop>0) instead of the two marginals.
        m_trainpop, m_devpop = joint_mask(nids, trainpop_hit, devpop_hit, rng)
        m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        both = (m_trainpop & m_devpop).mean()
        print(f"  joint mask: trainpop={m_trainpop.mean():.1%}  "
              f"devpop={m_devpop.mean():.1%}  both={both:.1%}  "
              f"(test: 12.8% / 15.5% / 11.6%)", flush=True)
    else:
        m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
        m_trainpop &= trainpop_hit
        m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
        m_devpop &= devpop_hit
        print(f"  masks: trainpop={m_trainpop.mean():.1%}  cf={m_cf.mean():.1%}  "
              f"devpop={m_devpop.mean():.1%}", flush=True)

    Xm = X.copy()
    if not args.no_mask:
        Xm[~m_trainpop, 2:5] = 0.0
        Xm[~m_cf, 1] = 0.0

    # ---- as-of normalisation (production-consistent: at test time the as-of
    # max IS the full-dev max) ----
    mp = np.maximum(C["asof_max_pop"], 1.0)
    mc = np.maximum(C["asof_max_cat"], 1.0)
    devpop = np.column_stack([
        C["causal_pop"] / mp,
        np.where(C["causal_pop"] > 0,
                 C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
        np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
    if not args.no_mask:
        devpop[~m_devpop] = 0.0
    catpop = np.column_stack([
        C["causal_cat_pop"] / mc,
        np.where(C["causal_cat_pop"] > 0,
                 C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)
    ]).astype(np.float32)

    blocks = [Xm, devpop, catpop]
    feats = list(BASE_FEATS) + ["dev_pop", "dev_ctr", "dev_log_pop",
                                "cat_pop", "cat_ctr"]
    if args.subcat:
        ms = np.maximum(S["asof_max_sub"], 1.0)
        subpop = np.column_stack([
            S["causal_sub_pop"] / ms,
            np.where(S["causal_sub_pop"] > 0,
                     S["causal_sub_clicks"] / np.maximum(S["causal_sub_pop"], 1), 0.0)
        ]).astype(np.float32)
        blocks.append(subpop)
        feats += ["sub_pop", "sub_ctr"]
    if args.wit:
        # WIT from the MASKED matrix: at test time cf_score is genuinely
        # absent for 88.7% of rows, so the within-impression z-score is
        # taken over a mostly-zero column. Computing it from the unmasked
        # matrix would train the model on a distribution it never sees.
        blocks.append(within_imp_transforms(Xm, g))
        feats += ["llm_rank", "llm_z", "cf_rank", "cf_z"]
    if args.pop_wit:
        # Within-impression transforms of the popularity features. These are
        # NOT masked: cat_pop / sub_pop have 100% test coverage, so their
        # within-impression rank is well defined on every row.
        if not args.subcat:
            raise SystemExit("--pop-wit requires --subcat (it needs sub_pop)")
        blocks.append(pop_wit_transforms(catpop, subpop, g))
        feats += POP_WIT_FEATS

    X2 = np.hstack(blocks)
    del blocks, devpop, catpop
    print(f"  feature matrix: {X2.shape}", flush=True)

    params = dict(objective="binary", metric="auc", learning_rate=0.05,
                  num_leaves=63, min_child_samples=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=args.threads, verbose=-1)
    dtr = lgb.Dataset(X2, label=y, feature_name=feats)
    bst = lgb.train(params, dtr, num_boost_round=args.rounds)

    # in-sample sanity only (NOT a validation number)
    p = bst.predict(X2)
    print(f"  in-sample mean pred: {p.mean():.4f} (sanity)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bst.save_model(args.out)
    # LightGBM persists generic Column_N names, so the real feature list is
    # written to a sidecar that apply_ranker_devpop.py reads.
    with open(args.out + ".feats.json", "w") as f:
        json.dump({"features": feats}, f, indent=1)
    print(f"saved {args.out}", flush=True)
    print(f"saved {args.out}.feats.json", flush=True)


if __name__ == "__main__":
    main()
