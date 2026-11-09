"""SCREEN: can an MRR/nDCG-oriented objective beat binary on AUC, MRR, nDCG?

WHY THIS SCREEN EXISTS
-----------------------
The leaderboard shows #1 (dingjian, AUC 0.7530) with MRR 0.4070, which is
+3.4 sd ABOVE the 50-team MRR~AUC regression line (AUC explains 94.6% of MRR
variance; residual sd 0.00449). So the #1 team's TOP-1 efficiency is
genuinely better than its AUC implies -- the observation is real, not an
artefact of reading the table.

Question: can we buy that top-1 efficiency with the objective, and what does
it cost in AUC?

THE TRAP THIS SCREEN MUST AVOID
--------------------------------
screen1 reported objective=lambdarank as +0.0029/+0.0031 (t=+6.3/+6.7) --
but that was the MEMORISATION LEAK (the train llm column is inflated:
within-impression AUC 0.7738 train vs 0.7188 dev). See
src/screen_rank_objective.py, which re-ran the same arms on a proxy llm
column degraded to EVAL quality and got the SIGN FLIP:
    lambdarank trunc30  -0.002015 (t=-5.52)
    lambdarank trunc100 -0.001559 (t=-4.15)
So lambdarank is DEAD *as an AUC lever*.

[CODE REVIEW 2026-11-08 -- ANNOTATION, NOT A DELETION. The historical text
above is kept verbatim for the record, but it must be read with two caveats:
 (i)  The PREVIOUS RUN OF THIS SCREEN fitted on the memorised column, so the
      lambdarank deltas it produced (quoted elsewhere) are INVALID for the
      FIT slice. See the CODE-REVIEW NOTE below.
 (ii) src/screen_rank_objective.py -- the source of the -0.002015/-0.001559
      sign flip quoted above -- has the SAME fit-slice leak: its `build()`
      also never references `proxy`, so it fitted on the memorised column and
      only swapped the proxy in for the EVAL slice. Its deltas are therefore
      "fit on memorised, evaluated on proxy", not "fit and evaluated on
      proxy". The comparison BETWEEN its arms is still like-for-like (all arms
      shared the same fit-time column), but its absolute lambdarank verdict
      is not confound-free in the fit slice either. The re-run of this screen
      in THIS revision is the first with a proxy fit slice.]

BUT: that screen measured ONLY AUC. It never measured MRR or nDCG. And
lambdarank is precisely an objective that reweights the loss towards the TOP
of each group (NDCG@truncation). So the honest state of knowledge is:
    "lambdarank hurts AUC"  ==  MEASURED
    "lambdarank hurts MRR"  ==  NOT MEASURED
This screen closes that gap.

DESIGN (identical to screen_rank_objective.py, so the two are comparable)
------------------------------------------------------------------------
  * 250,000 train impressions; fit [0,190k), validate [190k,250k);
    60,000 validation impressions; dev NEVER touched.
  * The llm column is degraded to EVAL quality with Gaussian noise whose
    sigma is CALIBRATED AT RUNTIME against dev's within-impression llm AUC
    (0.718848). Every arm is fitted AND evaluated on that same proxy
    column (the fit slice included) -- the fix in this revision closes the
    fit-on-memorised leak that the previous revision had -- so an arm
    cannot win by leaning harder on a memorised feature. (The llm
    zero-mask still applies at fit time as before; only the QUALITY of the
    llm column seen at fit time changed, raw memorised -> proxy.)
  * The script aborts if the calibration gap exceeds 0.005.

CODE-REVIEW NOTE (2026-11-08) -- THE FIT SLICE USED TO LEAK
-----------------------------------------------------------
Code review on 2026-11-08 found that the PREVIOUS revision of this screen
fitted every arm on the RAW (memorisation-inflated) train llm column and
only swapped in the proxy for the EVAL slice: `build()` never referenced
`proxy`, so `Xm[:, 0]` kept the memorised column for the fit rows. The
docstring claim "an arm cannot win by leaning harder on a memorised feature"
was therefore FALSE as written -- the arms were fitted on the leak.

THIS REVISION FIXES IT: `build()` now takes a `proxy_col` argument and writes
the proxy into the llm column BEFORE the within-impression transforms are
computed, so the fit slice sees the proxy too (and `llm_rank`/`llm_z` are
derived from the proxy rather than from the memorised column).

CONSEQUENCE FOR PREVIOUSLY QUOTED NUMBERS: the earlier run of THIS screen
fitted on the memorised column, so its lambdarank deltas are INVALID for the
FIT slice and must be treated as such wherever they are quoted. Those numbers
are NOT deleted -- they are annotated here and in the closing notes -- but
they do not support any conclusion about lambdarank.

METRICS (all per-impression, then averaged -- the official MIND protocol)
    AUC       : roc_auc_score per impression (ties -> 0.5)
    MRR       : official MIND mrr_score per impression
    nDCG@5/10 : official MIND ndcg_score per impression
Note the MRR column here uses the repo's `mind_utils.mrr_score` tie-break,
which breaks ties in REVERSE original order (`np.argsort(y_score)[::-1]`).
That differs from BOTH the production ranker path
(src/apply_ranker_devpop.py: `np.argsort(-s, kind="stable")`) and the official
evaluator (1/rank, which is tie-free). With LightGBM's piecewise-constant
outputs (~24% of candidates tied) the screen and production disagree on
~97% of impressions (mean |dMRR| 0.145). So the ABSOLUTE MRR values printed
by this screen are NOT directly comparable to the leaderboard's MRR column.
Only the PAIRED DELTAS between arms within this screen are meaningful, since
every arm shares the same tie-break.

ARMS
    A/binary              v6's exact recipe (the incumbent)
    B/lambdarank_t30      NDCG@30 truncation
    C/lambdarank_t100     NDCG@100 truncation
    D/lambdarank_t5       TOP-5 focused truncation  <-- new
    E/lambdarank_t1       TOP-1 focused truncation  <-- new (closest to MRR)
    F/rank_xendcg        rank_xendcg objective (default truncation)  <-- new

Every arm gets the SAME features, the SAME llm zero-mask, and the SAME proxy
column at fit time AND at eval time. The objective is the only thing that
differs.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "/tmp/res")
from train_ranker_causal_final import (BASE_FEATS, POP_WIT_FEATS,
    joint_mask, mask_rows_to_coverage, within_imp_transforms, pop_wit_transforms)
from imp_auc import per_imp_auc
from mind_utils import mrr_score, ndcg_score
import lightgbm as lgb

N_IMPS  = int(os.environ.get("N_IMPS", 250000))
CUT     = int(os.environ.get("CUT", 190000))
THREADS = int(os.environ.get("THREADS", 16))
ROUNDS  = int(os.environ.get("ROUNDS", 400))
SEED    = 13
DEV_LLM_AUC = 0.718848   # dev (cap-30) within-impression LLM AUC -- the target
TRAIN_FEATS = "results/ranker/large_train_features.tsv"
LLM_TRAIN   = "results/ranker/lg9b_ckpt45500_train_scores.tsv"
CACHE       = "results/ranker/causal_train_arrays.npz"

t0 = time.time()
X_list = []; y_list = []; imp_ids = []; nids_all = []; sizes = []
cur = None; rows = []; labs = []; nids = []
with open(TRAIN_FEATS) as f:
    header = f.readline().rstrip("\n").split("\t")
    idx = {c: i for i, c in enumerate(header)}
    cols = [idx[c] for c in BASE_FEATS]
    i_lab = idx["label"]; i_imp = idx["impression_id"]; i_nid = idx["news_id"]
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[i_imp] != cur:
            if cur is not None:
                X_list.append(np.array(rows, dtype=np.float32))
                y_list.append(np.array(labs, dtype=np.int8))
                imp_ids.append(cur); nids_all.extend(nids); sizes.append(len(rows))
                if len(imp_ids) >= N_IMPS: break
            cur = p[i_imp]; rows = []; labs = []; nids = []
        rows.append([p[c] for c in cols]); labs.append(int(p[i_lab])); nids.append(p[i_nid])
    if cur is not None and len(imp_ids) < N_IMPS:
        X_list.append(np.array(rows, dtype=np.float32))
        y_list.append(np.array(labs, dtype=np.int8))
        imp_ids.append(cur); nids_all.extend(nids); sizes.append(len(rows))
X = np.vstack(X_list); y = np.concatenate(y_list)
sizes = np.array(sizes, dtype=np.int64)
g = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
nrows = len(y); del X_list, y_list

llm = {}
with open(LLM_TRAIN) as f:
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2: llm[p[0]] = [float(v) for v in p[1:]]
off = 0; nmiss = 0
for imp, m in zip(imp_ids, sizes):
    s = llm.get(imp)
    if s and len(s) == m: X[off:off+m, 0] = s
    else: nmiss += 1
    off += m
llm_raw = X[:, 0].copy()
SD = float(llm_raw.std())
print(f"loaded {len(imp_ids):,} imps, {nrows:,} rows, missing llm {nmiss}, "
      f"sd={SD:.4f}, {time.time()-t0:.0f}s", flush=True)

C = np.load(CACHE)
def sl(k): return C[k][:nrows]
mp = np.maximum(sl("asof_max_pop"), 1.0)
mc = np.maximum(sl("asof_max_cat"), 1.0)
ms = np.maximum(sl("asof_max_sub"), 1.0)
DEVPOP = np.column_stack([sl("causal_pop")/mp,
    np.where(sl("causal_pop") > 0, sl("causal_clicks")/np.maximum(sl("causal_pop"), 1), 0.0),
    np.log1p(sl("causal_pop"))/np.maximum(np.log1p(mp), 1e-6)]).astype(np.float32)
CATPOP = np.column_stack([sl("causal_cat_pop")/mc,
    np.where(sl("causal_cat_pop") > 0, sl("causal_cat_clicks")/np.maximum(sl("causal_cat_pop"), 1), 0.0)]).astype(np.float32)
SUBPOP = np.column_stack([sl("causal_sub_pop")/ms,
    np.where(sl("causal_sub_pop") > 0, sl("causal_sub_clicks")/np.maximum(sl("causal_sub_pop"), 1), 0.0)]).astype(np.float32)

tr = g < CUT
va = ~tr
gtr = np.bincount(g[tr]).astype(np.int32)

# ---- runtime calibration of the noise sigma -------------------------------
_gv = g[va]; _yv = y[va]
_a_raw = per_imp_auc(_yv, llm_raw[va], _gv)
print(f"\ncalibration: raw train llm within-imp AUC = {_a_raw.mean():.6f} "
      f"(target {DEV_LLM_AUC:.6f})", flush=True)
_z = np.random.default_rng(SEED + 7).standard_normal(nrows)
_best = None
for _frac in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
    _v = llm_raw + _z * (_frac * SD)
    _a = per_imp_auc(_yv, _v[va], _gv).mean()
    _gap = _a - DEV_LLM_AUC
    print(f"  sigma = {_frac:.2f} sd -> within-imp AUC {_a:.6f}  gap {_gap:+.6f}", flush=True)
    if _best is None or abs(_gap) < abs(_best[2]):
        _best = (_frac, _a, _gap)
SIG_FRAC, _a_sel, _gap_sel = _best
print(f"  CHOSEN sigma = {SIG_FRAC:.2f} sd (within-imp AUC {_a_sel:.6f}, "
      f"gap {_gap_sel:+.6f})", flush=True)
if abs(_gap_sel) > 0.005:
    raise SystemExit(f"FATAL: could not calibrate sigma to dev quality "
                     f"(best gap {_gap_sel:+.6f}) -- the screen would be invalid")
proxy = (llm_raw + _z * (SIG_FRAC * SD)).astype(np.float32)
_a_proxy = per_imp_auc(_yv, proxy[va], _gv).mean()
print(f"  proxy column verified: within-imp AUC {_a_proxy:.6f} "
      f"(dev {DEV_LLM_AUC:.6f}, gap {_a_proxy - DEV_LLM_AUC:+.6f})", flush=True)


def build(mask_llm, proxy_col=None):
    """v6's exact feature construction. `mask_llm` toggles the llm zero-mask.

    `proxy_col` (the degraded-to-eval-quality llm column) is written into the
    llm column BEFORE the within-impression transforms are computed, so that
    the FIT rows see the proxy rather than the memorised raw column, and
    `llm_rank`/`llm_z` are derived from the proxy rather than from the
    memorised column. Without this the arms were fitted on the leak.

    Note the llm zero-mask still applies afterwards, exactly as before: the
    fit rows see the MASKED proxy (50% of rows zeroed), while the eval rows
    get the UNMASKED proxy via `Xv[:, 0] = proxy[va]` in fit_eval. That
    asymmetry is pre-existing and unchanged; only the QUALITY of the llm
    column seen at fit time changed (raw memorised -> proxy).
    """
    rng = np.random.default_rng(SEED)
    trainpop_hit = X[:, 2] > 0; cf_hit = X[:, 1] != 0; devpop_hit = sl("causal_pop") > 0
    m_trainpop, m_devpop = joint_mask(nids_all, trainpop_hit, devpop_hit, rng)
    m_cf = mask_rows_to_coverage(cf_hit, 0.113, rng)
    Xm = X.copy()
    if proxy_col is not None:
        Xm[:, 0] = proxy_col
    Xm[~m_trainpop, 2:5] = 0.0; Xm[~m_cf, 1] = 0.0
    if mask_llm:
        m_llm = mask_rows_to_coverage(np.ones(len(y), dtype=bool), 0.50, rng)
        Xm[~m_llm, 0] = 0.0
    dp = DEVPOP.copy(); dp[~m_devpop] = 0.0
    wit = within_imp_transforms(Xm, g)
    pw = pop_wit_transforms(CATPOP, SUBPOP, g)
    feats = (list(BASE_FEATS) + ["dev_pop", "dev_ctr", "dev_log_pop", "cat_pop",
              "cat_ctr", "sub_pop", "sub_ctr", "llm_rank", "llm_z", "cf_rank",
              "cf_z"] + POP_WIT_FEATS)
    return np.hstack([Xm, dp, CATPOP, SUBPOP, wit, pw]), feats


WIT_COLS = None


def eval_all_metrics(ys, ss, gg):
    """Per-impression AUC / MRR / nDCG@5 / nDCG@10, then averaged."""
    auc = per_imp_auc(ys, ss, gg)
    # group slices for the python-loop metrics
    order = np.argsort(gg, kind="stable")
    ys_s = ys[order]; ss_s = ss[order]; gg_s = gg[order]
    bounds = np.flatnonzero(np.r_[True, gg_s[1:] != gg_s[:-1], True])
    mrrs = np.empty(len(bounds) - 1); n5 = np.empty_like(mrrs); n10 = np.empty_like(mrrs)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i+1]
        yt = ys_s[a:b]; sc = ss_s[a:b]
        mrrs[i] = mrr_score(yt, sc)
        n5[i] = ndcg_score(yt, sc, 5)
        n10[i] = ndcg_score(yt, sc, 10)
    return auc, mrrs, n5, n10


def fit_eval(name, objective, trunc):
    t1 = time.time()
    # `proxy` is passed so that the FIT slice also sees the degraded llm
    # column, not just the eval slice (the fit-on-memorised leak).
    Xf, feats = build(True, proxy)
    global WIT_COLS
    if WIT_COLS is None:
        WIT_COLS = [feats.index(x) for x in ["llm_rank", "llm_z", "cf_rank", "cf_z"]]
    P = dict(learning_rate=0.05, num_leaves=63, min_child_samples=200,
             feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
             lambda_l2=1.0, num_threads=THREADS, verbose=-1)
    if objective == "binary":
        P.update(objective="binary", metric="auc")
        dtr = lgb.Dataset(Xf[tr], label=y[tr])
    elif objective == "xendcg":
        # NOTE: LightGBM has NO `xendcg_truncation_level` parameter. Passing it
        # was silently ignored (LightGBM 4.7 ignores unknown params), so this
        # arm was never "XENDCG@30" -- it was rank_xendcg with the DEFAULT
        # truncation. `objective="xendcg"` is an alias for "rank_xendcg".
        P.update(objective="xendcg", metric="ndcg", label_gain=[0, 1])
        dtr = lgb.Dataset(Xf[tr], label=y[tr], group=gtr)
    else:
        P.update(objective="lambdarank", metric="ndcg", label_gain=[0, 1],
                 lambdarank_truncation_level=trunc)
        dtr = lgb.Dataset(Xf[tr], label=y[tr], group=gtr)
    bst = lgb.train(P, dtr, num_boost_round=ROUNDS)
    del dtr
    Xv = Xf[va].copy(); gv = g[va]
    Xv[:, 0] = proxy[va]
    w = within_imp_transforms(Xv, gv)
    for j, p_ in enumerate(WIT_COLS):
        Xv[:, p_] = w[:, j]
    pr = bst.predict(Xv)
    a, m, d5, d10 = eval_all_metrics(y[va], pr, gv)
    del Xf, bst, pr, Xv
    print(f"  arm {name:<20} AUC {a.mean():.6f}  MRR {m.mean():.6f}  "
          f"nDCG@5 {d5.mean():.6f}  nDCG@10 {d10.mean():.6f}  "
          f"n={len(a):,}  ({time.time()-t1:.0f}s)", flush=True)
    return dict(auc=a, mrr=m, n5=d5, n10=d10)


ARMS = [
    ("A/binary",         "binary",     30),
    ("B/lambdarank_t30", "lambdarank", 30),
    ("C/lambdarank_t100","lambdarank", 100),
    ("D/lambdarank_t5",  "lambdarank", 5),
    ("E/lambdarank_t1",  "lambdarank", 1),
    ("F/rank_xendcg",     "xendcg",     30),
]
res = {}
for name, obj, trunc in ARMS:
    try:
        res[name] = fit_eval(name, obj, trunc)
    except Exception as e:
        print(f"  arm {name:<20} FAILED: {type(e).__name__}: {e}", flush=True)

ref = res.get("A/binary")
if ref is None:
    raise SystemExit("FATAL: the binary baseline failed; no comparison possible")

print("\n=== paired deltas vs A/binary (train-internal, PROXY fit AND eval column, dev untouched) ===",
      flush=True)
for k, v in res.items():
    if k == "A/binary": continue
    line = f"  {k:<20}"
    for mk in ["auc", "mrr", "n5", "n10"]:
        d = v[mk] - ref[mk]
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("nan")
        verdict = "BETTER" if (d.mean() > 0 and t > 2) else "WORSE" if (d.mean() < 0 and t < -2) else "flat"
        line += f"  {mk}: {d.mean():+.6f} (t={t:+.2f}, {verdict})"
    print(line, flush=True)

print("\n  reference: screen1 (MEMORISED eval column) reported lambdarank at "
      "+0.002893 (t=+6.31) / +0.003057 (t=+6.67) on AUC -- that was the leak.",
      flush=True)
print("  The AUC column above is the confound-free version (proxy column on the "
      "FIT slice as well as the eval slice -- the fit-on-memorised leak found by "
      "code review on 2026-11-08 is closed in this revision); "
      "the MRR / nDCG columns are NEW -- no earlier screen measured them.",
      flush=True)
print("  CAVEAT on the MRR column: it uses mind_utils.mrr_score's tie-break "
      "(argsort(sc)[::-1]), which differs from the production ranker's stable "
      "tie-break and from the official evaluator's tie-free 1/rank, so the "
      "ABSOLUTE MRR values are NOT comparable to the leaderboard's MRR column. "
      "Only the PAIRED DELTAS between arms above are meaningful.",
      flush=True)
