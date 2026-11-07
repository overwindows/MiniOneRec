#!/usr/bin/env python3
"""Train the stack ranker on the MIND_large TRAIN split (dev = eval only).

WHY
---
The previous stack rankers (v3/v4/v5) were trained on dev labels, which
violates the evaluation protocol: dev is touched exactly once, as the final
held-out eval. This trainer uses ONLY train-split labels, with the same
promoted feature set as train_ranker_causal_final.py --pop-wit:

    BASE_FEATS + devpop(3) + catpop(2) + subpop(2)
    + WIT(llm/cf rank+z, 4) + POP_WIT(cat/sub pop/ctr rank+z, 8)

Key differences vs the dev trainer:
  1. Labels come from the train split (large_train_features.tsv).
  2. Causal popularity walks TRAIN behaviors in order (each row sees only
     PRIOR train impressions), so the popularity features are strictly
     as-of within the train split.
  3. The devpop family (news popularity from the DEV split) is NOT
     available here -- at train time the dev split "hasn't happened yet".
     The test-regime masking already zeroes devpop for 84.5% of rows, and
     the model treats it as a sparse bonus signal. The dev-side feature
     table still carries it; the model simply sees it zero on most
     training rows (same as test).
  4. The LLM's train scores are memorization-inflated (probe: train AUC
     0.7692 vs dev 0.7134). To stop the ranker from learning "llm_score
     is always right", llm_score is NOISED-DOWN by masking: a random
     subset of rows has llm_score zeroed so the effective llm coverage
     matches the DEV distribution of informative rows. The WIT (rank/z)
     is computed from the masked matrix, exactly like cf.

Usage:
    python src/train_ranker_train_split.py \
        --llm-train results/ranker/lg9b_ckpt45500_train_scores.tsv \
        --out results/ranker/ranker_v6_train_popwit.txt \
        --joint-mask --wit --subcat --pop-wit --rounds 2000
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402
from train_ranker_causal_final import (  # noqa: E402
    BASE_FEATS, POP_WIT_FEATS, COV_CF,
    TEST_BOTH, TEST_TR_ONLY, TEST_DV_ONLY, TEST_NEITHER,
    _avg_rank_grouped, _zscore_grouped, within_imp_transforms,
    pop_wit_transforms, joint_mask, mask_rows_to_coverage,
)

import lightgbm as lgb  # noqa: E402

TRAIN_FEATS = "results/ranker/large_train_features.tsv"
TRAIN_BEHAVIORS = "data/MIND_large/train/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_train_arrays.npz"

# The LLM saw the train split during fine-tuning, so its train scores are
# memorization-inflated (probe: 0.7692 train vs 0.7134 dev). Masking
# llm_score down to this coverage both (a) limits how much the ranker can
# lean on it and (b) mirrors the test regime where llm_score is the only
# dense signal. 0.50 keeps half the rows informative.
COV_LLM = 0.50
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


def build_causal_train(n_rows, use_subcat):
    """Walk TRAIN in order; each row sees only PRIOR train impressions."""
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
    with open(TRAIN_BEHAVIORS) as f:
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
    ap.add_argument("--out", default="results/ranker/ranker_v6_train_popwit.txt")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--subcat", action="store_true")
    ap.add_argument("--wit", action="store_true")
    ap.add_argument("--pop-wit", action="store_true")
    ap.add_argument("--joint-mask", action="store_true")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--cov-llm", type=float, default=COV_LLM,
                    help="llm_score masking coverage: the train scores are "
                         "memorization-inflated, so a fraction is zeroed to "
                         "stop the ranker over-trusting llm_score")
    ap.add_argument("--train-feats", default=TRAIN_FEATS)
    ap.add_argument("--llm-train", required=True,
                    help="train LLM score file (impression_id + per-candidate "
                         "scores from the SAME checkpoint used at test time)")
    args = ap.parse_args()

    print(f"Loading train features ({args.train_feats}) ...", flush=True)
    Xs, ys, gs = [], [], []
    for gi, (Xb, y) in enumerate(stream_groups(args.train_feats)):
        Xs.append(Xb)
        ys.append(y)
        gs.append(np.full(len(y), gi, dtype=np.int64))
        if gi % 500000 == 0 and gi:
            print(f"  {gi:,} imps ...", flush=True)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    del Xs, ys
    ids = imp_ids(args.train_feats)
    llm = load_llm(args.llm_train)
    off = 0
    n_miss = 0
    for imp, n in ids:
        s = llm.get(imp)
        if s and len(s) == n:
            X[off:off + n, 0] = s
        else:
            n_miss += 1
        off += n
    print(f"  {X.shape[0]:,} rows, {len(ids):,} imps, missing llm: {n_miss}",
          flush=True)

    print("Loading causal popularity arrays (train walk) ...", flush=True)
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        C = {k: z[k] for k in z.files}
    else:
        C = build_causal_train(X.shape[0], args.subcat)
        np.savez(CACHE, **C)

    with open(args.train_feats) as f:
        header = f.readline().rstrip("\n").split("\t")
        i_nid = header.index("news_id")
    nids = []
    with open(args.train_feats) as f:
        f.readline()
        for line in f:
            nids.append(line.rstrip("\n").split("\t")[i_nid])

    rng = np.random.default_rng(SEED)
    trainpop_hit = X[:, 2] > 0
    cf_hit = X[:, 1] != 0
    # devpop does not exist at train time: the causal train walk's news-level
    # popularity plays that role for the joint mask's second family.
    devpop_hit = C["causal_pop"] > 0
    if args.joint_mask:
        m_trainpop, m_devpop = joint_mask(nids, trainpop_hit, devpop_hit, rng)
        m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        both = (m_trainpop & m_devpop).mean()
        print(f"  joint mask: trainpop={m_trainpop.mean():.1%}  "
              f"devpop={m_devpop.mean():.1%}  both={both:.1%}  "
              f"(test: 12.8% / 15.5% / 11.6%)", flush=True)
    else:
        m_trainpop = mask_rows_to_coverage(trainpop_hit, 0.128, rng)
        m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
        m_devpop = mask_rows_to_coverage(devpop_hit, 0.155, rng)
        print(f"  masks: trainpop={m_trainpop.mean():.1%}  "
              f"cf={m_cf.mean():.1%}  devpop={m_devpop.mean():.1%}", flush=True)

    # llm_score masking: the train scores are memorization-inflated. Zeroing
    # a fraction of them (a) caps how much the ranker can lean on llm_score
    # and (b) makes the WIT (rank/z) computed from a partially-masked
    # column, mirroring how cf WIT is already handled.
    m_llm = mask_rows_to_coverage(np.ones(len(y), dtype=bool), args.cov_llm, rng)
    print(f"  llm mask: keep={m_llm.mean():.1%}", flush=True)

    Xm = X.copy()
    if not args.no_mask:
        Xm[~m_trainpop, 2:5] = 0.0
        Xm[~m_cf, 1] = 0.0
        Xm[~m_llm, 0] = 0.0

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
        ms = np.maximum(C["asof_max_sub"], 1.0)
        subpop = np.column_stack([
            C["causal_sub_pop"] / ms,
            np.where(C["causal_sub_pop"] > 0,
                     C["causal_sub_clicks"] / np.maximum(C["causal_sub_pop"], 1), 0.0)
        ]).astype(np.float32)
        blocks.append(subpop)
        feats += ["sub_pop", "sub_ctr"]
    if args.wit:
        blocks.append(within_imp_transforms(Xm, g))
        feats += ["llm_rank", "llm_z", "cf_rank", "cf_z"]
    if args.pop_wit:
        if not args.subcat:
            raise SystemExit("--pop-wit requires --subcat")
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

    p = bst.predict(X2)
    print(f"  in-sample mean pred: {p.mean():.4f} (sanity, NOT a result)",
          flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bst.save_model(args.out)
    with open(args.out + ".feats.json", "w") as f:
        json.dump({"features": feats}, f, indent=1)
    print(f"saved {args.out}", flush=True)
    print(f"saved {args.out}.feats.json", flush=True)


if __name__ == "__main__":
    main()