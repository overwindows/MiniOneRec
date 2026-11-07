#!/usr/bin/env python3
"""Title-text similarity features: candidate vs user history.

Every feature in the current ranker is either (a) the LLM score, (b) a
popularity count, or (c) a category/history statistic. None of them compare
the candidate's TEXT with the user's history TEXT.

Title similarity is a completely independent signal that:
  * is available for 100% of test rows (no popularity dependency),
  * is computable identically at train and test time,
  * is strongest exactly where the popularity features are dead (the 83% of
    test rows whose news appears in neither train nor dev).

Features computed per (impression, candidate):
  title_max_cos   - max cosine(TFIDF(cand.title), TFIDF(hist.title))
  title_mean_cos  - mean cosine over the history
  title_top3_cos  - mean of the 3 highest cosines
  title_jaccard   - Jaccard overlap of the title word sets
  title_ovl_hist  - fraction of history items sharing >=1 word with the title

Usage: python src/exp_ranker_textsim.py
"""
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from train_ranker_v2 import stream_groups  # noqa: E402

import lightgbm as lgb  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

DEV_FEATS = "results/ranker/large_dev_features.tsv"
LG9B_DEV = "results/ranker/lg9b_ep082_dev_scores.tsv"
DEV_BEHAVIORS = "data/MIND_large/dev/behaviors.tsv"
NEWS = "data/MIND_large/test/news.tsv"
CACHE = "results/ranker/causal_dev_arrays.npz"
CACHE2 = "results/ranker/causal_subcat_arrays.npz"
TEXTCACHE = "results/ranker/textsim_dev.npz"

COV_TRAINPOP = 0.128
COV_CF = 0.113
COV_DEVPOP = 0.155

SEED = 13
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05,
              num_leaves=63, min_child_samples=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=16, verbose=-1)

TOKEN = re.compile(r"[a-z0-9]+")


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


def per_imp(y_t, s, gg, mask=None):
    yy, ss, ggg = (y_t, s, gg) if mask is None else (y_t[mask], s[mask], gg[mask])
    if mask is not None:
        # Remap the group ids to 0..k-1. After masking, ggg still holds the
        # ORIGINAL impression ids (e.g. 301176..376470), so
        # np.bincount(g_s, minlength=len(ug)) would allocate len(ug) buckets
        # while g_s indexes far beyond them -> broadcast error.
        _, ggg = np.unique(ggg, return_inverse=True)
    order = np.argsort(ggg, kind="stable")
    g_s, p_s, y_s = ggg[order], ss[order], yy[order]
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


def mask_news_to_coverage(nids, hit, target, rng):
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


def build_textsim(n_rows):
    """Per (impression, candidate) title-vs-history similarity features."""
    print("Loading news titles ...", flush=True)
    title = {}
    with open(NEWS) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                title[p[0]] = p[3]
    # also pull dev-only news titles (771 dev news are absent from test/news.tsv)
    with open("data/MIND_large/dev/news.tsv") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                title.setdefault(p[0], p[3])

    print("Fitting TF-IDF ...", flush=True)
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+",
                           min_df=2, max_features=200000, sublinear_tf=True)
    all_titles = list(title.values())
    vec.fit(all_titles)
    print(f"  vocab={len(vec.vocabulary_):,}", flush=True)

    # token sets for the cheap lexical features
    toks = {n: set(TOKEN.findall(t.lower())) for n, t in title.items()}

    feats = np.zeros((n_rows, 5), dtype=np.float32)
    row = 0
    n_imp = 0
    with open(DEV_BEHAVIORS) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            hist = parts[3].split()[-30:]
            cands = []
            for tk in parts[4].split():
                if "-" not in tk:
                    continue
                nid, lab = tk.rsplit("-", 1)
                cands.append(nid)

            h_titles = [title[h] for h in hist if h in title]
            c_titles = [title[c] for c in cands if c in title]
            if not h_titles or not c_titles:
                # no history or no known titles -> all-zero features
                row += len(cands)
                n_imp += 1
                continue

            H = vec.transform(h_titles)
            C = vec.transform(c_titles)
            # cosine similarity matrix (candidates x history)
            sim = (C @ H.T).toarray()
            sim = np.nan_to_num(sim)
            for i, cid in enumerate(cands):
                s = sim[i]
                feats[row, 0] = s.max()
                feats[row, 1] = s.mean()
                k = min(3, len(s))
                feats[row, 2] = np.sort(s)[-k:].mean()
                ct = toks.get(cid, set())
                if ct:
                    hs = [toks.get(h, set()) for h in hist if h in toks]
                    if hs:
                        jac = [len(ct & h) / max(len(ct | h), 1) for h in hs]
                        feats[row, 3] = max(jac)
                        feats[row, 4] = sum(1 for j in jac if j > 0) / len(jac)
                row += 1
            n_imp += 1
            if n_imp % 50000 == 0:
                print(f"  {n_imp:,} imps, {row:,} rows", flush=True)
    assert row == n_rows, f"row mismatch {row} vs {n_rows}"
    return feats


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
print(f"  {X.shape[0]:,} rows, {n_imp:,} imps", flush=True)

if os.path.exists(TEXTCACHE):
    print(f"Loading cached text features from {TEXTCACHE} ...", flush=True)
    zt = np.load(TEXTCACHE)
    TS = zt["textsim"]
else:
    print("Building title-similarity features ...", flush=True)
    TS = build_textsim(X.shape[0])
    np.savez(TEXTCACHE, textsim=TS)
    print(f"  cached -> {TEXTCACHE}", flush=True)

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

rng = np.random.default_rng(SEED)
trainpop_hit = X[:, 2] > 0
m_trainpop = mask_news_to_coverage(nids, trainpop_hit, COV_TRAINPOP, rng)
m_trainpop &= trainpop_hit
cf_hit = X[:, 1] != 0
m_cf = mask_rows_to_coverage(cf_hit, COV_CF, rng)
devpop_hit = C["causal_pop"] > 0
m_devpop = mask_news_to_coverage(nids, devpop_hit, COV_DEVPOP, rng)
m_devpop &= devpop_hit
print(f"  masks: trainpop={m_trainpop.mean():.1%}  cf={m_cf.mean():.1%}  "
      f"devpop={m_devpop.mean():.1%}", flush=True)

Xm = X.copy()
Xm[~m_trainpop, 2:5] = 0.0
Xm[~m_cf, 1] = 0.0

mp = np.maximum(C["asof_max_pop"], 1.0)
mc = np.maximum(C["asof_max_cat"], 1.0)
ms = np.maximum(S["sub_asof"], 1.0)
DEVPOP = np.column_stack([
    C["causal_pop"] / mp,
    np.where(C["causal_pop"] > 0, C["causal_clicks"] / np.maximum(C["causal_pop"], 1), 0.0),
    np.log1p(C["causal_pop"]) / np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
DEVPOP_M = DEVPOP.copy()
DEVPOP_M[~m_devpop] = 0.0
CAT = np.column_stack([
    C["causal_cat_pop"] / mc,
    np.where(C["causal_cat_pop"] > 0,
             C["causal_cat_clicks"] / np.maximum(C["causal_cat_pop"], 1), 0.0)]).astype(np.float32)
SUBCAT = np.column_stack([
    S["sub_pop"] / ms,
    np.where(S["sub_pop"] > 0, S["sub_clk"] / np.maximum(S["sub_pop"], 1), 0.0)]).astype(np.float32)

va = g >= cut
tr = g < cut
yv, gv = y[va], g[va]
no_pop = (~m_devpop) & (~m_trainpop)


def fit_eval(Xf, name, mask=None):
    dtr = lgb.Dataset(Xf[tr], label=y[tr])
    bst = lgb.train(PARAMS, dtr, num_boost_round=300)
    p = bst.predict(Xf[va])
    a, n = per_imp(yv, p, gv)
    extra = ""
    if mask is not None:
        am, nm = per_imp(yv, p, gv, mask[va])
        extra = f"  nopop={am:.4f} (n={nm})"
    print(f"{name:>52}  val={a:.4f} (n={n}){extra}", flush=True)
    return a


print("\n=== title-similarity features ===", flush=True)
BASE = np.hstack([Xm, DEVPOP_M, CAT, SUBCAT])
fit_eval(BASE, "base + devpop + cat + subcat")
fit_eval(np.hstack([BASE, TS]), "base + devpop + cat + subcat + textsim")
fit_eval(np.hstack([Xm, TS]), "base + textsim")
fit_eval(np.hstack([Xm, DEVPOP_M, SUBCAT, TS]), "base + devpop + subcat + textsim")

print("\n=== on the no-popularity rows ===", flush=True)
fit_eval(BASE, "base + devpop + cat + subcat", no_pop)
fit_eval(np.hstack([BASE, TS]), "base + devpop + cat + subcat + textsim", no_pop)
fit_eval(np.hstack([Xm, TS]), "base + textsim", no_pop)
