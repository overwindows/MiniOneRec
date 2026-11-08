"""TRAIN-ONLY screen, CONFOUND-FREE: the LightGBM OBJECTIVE lever.

WHY A SCREEN AT ALL, AND WHY NOT THE OBVIOUS ONE.
screen1 measured `objective=lambdarank` vs `objective=binary` on a train-internal
split and got +0.0029 / +0.0031 (t=+6.3 / +6.7). But the train llm column is
MEMORISATION-INFLATED: within-impression LLM AUC is 0.7741 on train vs
0.7188 on dev, i.e. +0.0553. The leak sits in BOTH the fit slice and the
validation slice, so a train-internal screen can only trust a lever that does
NOT change how much the ranker leans on llm_score. Changing the objective
does exactly that (lambdarank optimises a within-group ranking loss instead
of a pointwise logloss, so it reweights the features), so screen1's number
cannot be accepted as-is.

THE FIX (same design as screen8). Degrade the train llm column to EVAL
quality with Gaussian noise, then evaluate EVERY arm on that same degraded
column. The noise sigma is CALIBRATED AT RUNTIME by sweeping it and picking
the value whose within-impression AUC on the validation slice matches dev's
0.718848.

CALIBRATION (runtime, on the TRAIN SCORE FILE's column -- measured
2026-11-08, validation slice of this exact screen):
    train score file, raw            within-imp AUC 0.773781   (memorised)
    sigma = 0.60 sd                   within-imp AUC 0.739938   (+0.0211 -- TOO GOOD)
    sigma = 0.80 sd                   within-imp AUC 0.720447   (+0.0016 == dev)  <-- chosen
    sigma = 1.00 sd                   within-imp AUC 0.701026   (too destroyed)
NOTE: the earlier /tmp/res/sigcal.py calibration said 0.60 sd, but it
swept the FEATURE TABLE's llm_score column, which has a different scale
(raw within-imp AUC 0.749492, not 0.773781). The screen8 proxy column was
therefore NOT eval-quality either. The runtime calibration below removes
that dependency -- and it matters: at the mis-calibrated sigma=0.60 the
lambdarank arms looked merely FLAT (-0.0006..-0.0008), while at the correct
sigma=0.80 they are clearly WORSE (-0.0016..-0.0032).

RESULT (250,000 train impressions, fit [0,190k), validate [190k,250k),
60,000 validation impressions, 400 rounds, dev untouched):
    A/binary (v6 recipe)              0.738807
    B/lambdarank trunc 30              0.736792   delta -0.002015  t=-5.52  WORSE
    C/lambdarank trunc 100             0.737248   delta -0.001559  t=-4.15  WORSE
    D/lambdarank trunc 30, no mask     0.735595   delta -0.003212  t=-7.99  WORSE
The memorised-column numbers were +0.002893 (t=+6.31) and +0.003057
(t=+6.67). THE SIGN FLIPS: lambdarank is DEAD.

ARMS (identical features, identical llm masking, identical evaluation):
  A/binary       v6's exact recipe
  B/lambdarank   same features, objective=lambdarank, truncation 30
  C/lambdarank   same features, objective=lambdarank, truncation 100
  D/lambdarank   same features, objective=lambdarank, truncation 30, no
                 zero-masking of llm_score (the mask is a separate lever)
Both the fit slice and the evaluation slice see the SAME masked/zeroed
column, so the objective is the only thing that differs between A and B/C.

Dev is NOT touched.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "/tmp/res")
from train_ranker_causal_final import (BASE_FEATS, POP_WIT_FEATS,
    joint_mask, mask_rows_to_coverage, within_imp_transforms, pop_wit_transforms)
from imp_auc import per_imp_auc
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

t0=time.time()
X_list=[]; y_list=[]; imp_ids=[]; nids_all=[]; sizes=[]
cur=None; rows=[]; labs=[]; nids=[]
with open(TRAIN_FEATS) as f:
    header=f.readline().rstrip("\n").split("\t")
    idx={c:i for i,c in enumerate(header)}
    cols=[idx[c] for c in BASE_FEATS]
    i_lab=idx["label"]; i_imp=idx["impression_id"]; i_nid=idx["news_id"]
    for line in f:
        p=line.rstrip("\n").split("\t")
        if p[i_imp]!=cur:
            if cur is not None:
                X_list.append(np.array(rows,dtype=np.float32)); y_list.append(np.array(labs,dtype=np.int8))
                imp_ids.append(cur); nids_all.extend(nids); sizes.append(len(rows))
                if len(imp_ids)>=N_IMPS: break
            cur=p[i_imp]; rows=[]; labs=[]; nids=[]
        rows.append([p[c] for c in cols]); labs.append(int(p[i_lab])); nids.append(p[i_nid])
    if cur is not None and len(imp_ids)<N_IMPS:
        X_list.append(np.array(rows,dtype=np.float32)); y_list.append(np.array(labs,dtype=np.int8))
        imp_ids.append(cur); nids_all.extend(nids); sizes.append(len(rows))
X=np.vstack(X_list); y=np.concatenate(y_list)
sizes=np.array(sizes, dtype=np.int64); g=np.repeat(np.arange(len(sizes),dtype=np.int64), sizes)
nrows=len(y); del X_list,y_list

llm={}
with open(LLM_TRAIN) as f:
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)>=2: llm[p[0]]=[float(v) for v in p[1:]]
off=0; nmiss=0
for imp,m in zip(imp_ids,sizes):
    s=llm.get(imp)
    if s and len(s)==m: X[off:off+m,0]=s
    else: nmiss+=1
    off+=m
llm_raw=X[:,0].copy()
SD=float(llm_raw.std())
print(f"loaded {len(imp_ids):,} imps, {nrows:,} rows, missing llm {nmiss}, "
      f"sd={SD:.4f}, {time.time()-t0:.0f}s", flush=True)

C=np.load(CACHE)
def sl(k): return C[k][:nrows]
mp=np.maximum(sl("asof_max_pop"),1.0); mc=np.maximum(sl("asof_max_cat"),1.0); ms=np.maximum(sl("asof_max_sub"),1.0)
DEVPOP=np.column_stack([sl("causal_pop")/mp,
    np.where(sl("causal_pop")>0, sl("causal_clicks")/np.maximum(sl("causal_pop"),1),0.0),
    np.log1p(sl("causal_pop"))/np.maximum(np.log1p(mp),1e-6)]).astype(np.float32)
CATPOP=np.column_stack([sl("causal_cat_pop")/mc,
    np.where(sl("causal_cat_pop")>0, sl("causal_cat_clicks")/np.maximum(sl("causal_cat_pop"),1),0.0)]).astype(np.float32)
SUBPOP=np.column_stack([sl("causal_sub_pop")/ms,
    np.where(sl("causal_sub_pop")>0, sl("causal_sub_clicks")/np.maximum(sl("causal_sub_pop"),1),0.0)]).astype(np.float32)

tr = g < CUT
va = ~tr
gtr = np.bincount(g[tr]).astype(np.int32)

# ---- runtime calibration of the noise sigma -------------------------------
# The proxy column must have the SAME within-impression ordering quality as
# the dev llm column, otherwise a lever that only leans harder on llm_score
# can still show a fake gain. Sweep sigma and pick the value whose
# within-impression AUC on the VALIDATION slice matches dev's.
_gv = g[va]; _yv = y[va]
_a_raw = per_imp_auc(_yv, llm_raw[va], _gv)
print(f"\ncalibration: raw train llm within-imp AUC = {_a_raw.mean():.6f} "
      f"(target {DEV_LLM_AUC:.6f})", flush=True)
# ONE fixed set of standard normals, SCALED by each sigma, so the calibrated
# sigma and the final proxy column use the SAME noise realisation.
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

def build(mask_llm):
    """v6's exact feature construction. `mask_llm` toggles the llm zero-mask."""
    rng=np.random.default_rng(SEED)
    trainpop_hit=X[:,2]>0; cf_hit=X[:,1]!=0; devpop_hit=sl("causal_pop")>0
    m_trainpop,m_devpop=joint_mask(nids_all,trainpop_hit,devpop_hit,rng)
    m_cf=mask_rows_to_coverage(cf_hit,0.113,rng)
    Xm=X.copy()
    Xm[~m_trainpop,2:5]=0.0; Xm[~m_cf,1]=0.0
    if mask_llm:
        m_llm=mask_rows_to_coverage(np.ones(len(y),dtype=bool),0.50,rng)
        Xm[~m_llm,0]=0.0
    dp=DEVPOP.copy(); dp[~m_devpop]=0.0
    wit=within_imp_transforms(Xm,g)
    pw=pop_wit_transforms(CATPOP,SUBPOP,g)
    feats=list(BASE_FEATS)+["dev_pop","dev_ctr","dev_log_pop","cat_pop","cat_ctr","sub_pop","sub_ctr",
                            "llm_rank","llm_z","cf_rank","cf_z"]+POP_WIT_FEATS
    return np.hstack([Xm,dp,CATPOP,SUBPOP,wit,pw]), feats

WIT_COLS = None  # filled after feats are known

def fit_eval(name, objective, trunc, mask_llm):
    t1=time.time()
    Xf, feats = build(mask_llm)
    global WIT_COLS
    if WIT_COLS is None:
        WIT_COLS = [feats.index(x) for x in ["llm_rank","llm_z","cf_rank","cf_z"]]
    P=dict(learning_rate=0.05,num_leaves=63,min_child_samples=200,feature_fraction=0.9,
           bagging_fraction=0.8,bagging_freq=1,lambda_l2=1.0,num_threads=THREADS,verbose=-1)
    if objective=="binary":
        P.update(objective="binary", metric="auc")
        dtr=lgb.Dataset(Xf[tr],label=y[tr])
    else:
        P.update(objective="lambdarank", metric="ndcg", label_gain=[0,1],
                 lambdarank_truncation_level=trunc)
        dtr=lgb.Dataset(Xf[tr],label=y[tr],group=gtr)
    bst=lgb.train(P,dtr,num_boost_round=ROUNDS)
    del dtr
    # EVALUATION: every arm gets the SAME proxy column (eval-quality ordering)
    Xv=Xf[va].copy(); gv=g[va]
    Xv[:,0]=proxy[va]
    w=within_imp_transforms(Xv,gv)
    for j,p_ in enumerate(WIT_COLS):
        Xv[:,p_]=w[:,j]
    pr=bst.predict(Xv); a=per_imp_auc(y[va],pr,gv)
    del Xf,bst,pr,Xv
    print(f"  arm {name:<22} TRAIN-internal val within-imp AUC (proxy eval column) = "
          f"{a.mean():.6f}  n={len(a):,}  ({time.time()-t1:.0f}s)", flush=True)
    return a

ARMS = [
    ("A/binary",            "binary",     30,  True),
    ("B/lambdarank_t30",    "lambdarank", 30,  True),
    ("C/lambdarank_t100",   "lambdarank", 100, True),
    ("D/lambdarank_t30_nomask","lambdarank",30, False),
]
res={}
for name, obj, trunc, mask in ARMS:
    res[name]=fit_eval(name,obj,trunc,mask)

ref=res["A/binary"]
print("\n=== paired deltas vs A/binary (TRAIN-internal, PROXY eval column, dev untouched) ===", flush=True)
for k,v in res.items():
    d=v-ref; se=d.std(ddof=1)/np.sqrt(len(d)); t=d.mean()/se if se>0 else float("nan")
    verdict = "BETTER" if (d.mean()>0 and t>2) else "WORSE" if (d.mean()<0 and t<-2) else "NO SIG. CHANGE"
    print(f"  {k:<24} delta = {d.mean():+.6f}  SE = {se:.6f}  t = {t:+.2f}   {verdict}", flush=True)

# The screen1 numbers, for reference only (same arms, MEMORISED eval column):
print("\n  reference: screen1 (memorised eval column) gave "
      "A/lambdarank +0.002893 t=+6.31 and screen2 C/lambdarank +0.003057 t=+6.67.")
print("  If the proxy-evaluated deltas collapse, screen1's gain was memorisation.")
