"""Probe: WHICH logit row carries the relevance signal, and does the
pairwise trainer ever actually form a within-impression pair?

Two suspected defects in src/sft_mind_pointwise_pairwise.py (A7.3):

  (1) SCORE ROW. The trainer does
          valid = labels != -100 ; pos = valid[i].nonzero()[0]   # answer row
          scores.append(logits[i, pos, YES_ID])
      In causal LM, logits[..., t, :] predicts token t+1. The answer token
      sits at L-1, so the row that PREDICTS it is L-2. Eval
      (src/evaluate_mind_pointwise.py) scores the last PROMPT hidden state,
      which is the same L-2 semantics. The trainer therefore ranks on a
      different row than the metric is computed on.

  (2) PAIRING. The trainer pairs a positive with a negative from the SAME
      impression, but only if both land in the SAME micro-batch. Samples are
      shuffled, so that almost never happens (measured: 0.25% of micro-batches
      under the production length-grouped sampler).

This probe measures (1) on a real trained checkpoint. (2) is measured
separately by src/probe_pairwise_pairing.py.

SAMPLING NOTE: MINDPointwiseSFTDataset shuffles ALL samples and then
truncates to `sample_limit`, which destroys impression grouping (it yields
~1 sample per impression). To measure PER-IMPRESSION AUC we must load the
full dataset (sample=-1) and then select WHOLE impressions.

Usage:
    python src/probe_pairwise_score_row.py <ckpt> <n_impressions> [split]
"""
import sys, os, json, collections
sys.path.insert(0, '/import/ml-sc-scratch5/chenw/MiniOneRec')
os.chdir('/import/ml-sc-scratch5/chenw/MiniOneRec')
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import MINDPointwiseSFTDataset
from sklearn.metrics import roc_auc_score

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'output_dir/small_runs/hp_neg1/final_checkpoint'
N_IMPS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
SPLIT = sys.argv[3] if len(sys.argv) > 3 else 'dev'

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id; tok.padding_side = 'left'
YES = tok.encode(" Yes", add_special_tokens=False)[0]
NO = tok.encode(" No", add_special_tokens=False)[0]
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16,
                                             local_files_only=True).cuda().eval()

# sample=-1 => keep every sample, so impressions stay intact.
ds = MINDPointwiseSFTDataset(
    behaviors_path=f'data/MIND_small/{SPLIT}/behaviors.tsv',
    news_path=f'data/MIND_small/{SPLIT}/news.tsv',
    tokenizer=tok, max_len=2048, sample=-1, seed=42, max_history=30,
    neg_ratio=2.0, hard_neg_ratio=0.5, use_abstract=True,
    use_chat_template=True, return_meta=True)

# Select the first N_IMPS WHOLE impressions (in file order) and their samples.
by_imp = collections.OrderedDict()
for idx, s in enumerate(ds.samples):
    by_imp.setdefault(s['impression_id'], []).append(idx)
keep_imps = list(by_imp.keys())[:N_IMPS]
keep_idx = [i for imp in keep_imps for i in by_imp[imp]]
print(f"selected {len(keep_imps)} impressions / {len(keep_idx)} candidates")

recs = []
with torch.no_grad():
    for i in keep_idx:
        s = ds[i]
        ids = s['input_ids'].unsqueeze(0).cuda()
        logits = model(ids).logits[0].float()
        vpos = int(np.nonzero(s['labels'].numpy() != -100)[0][0])
        recs.append({
            'imp': s['impression_id'], 'label': int(s['label']),
            # (a) what the pairwise trainer reads: the row AT the answer token
            'at':   float(logits[vpos,   YES] - logits[vpos,   NO]),
            # (b) the row that PREDICTS the answer token (vpos-1) -- the same
            #     position eval scores (last prompt hidden state).
            'pred': float(logits[vpos-1, YES] - logits[vpos-1, NO]),
        })

def per_imp_auc(key):
    g = collections.defaultdict(list)
    for r in recs: g[r['imp']].append(r)
    aucs = []
    for imp, rs in g.items():
        y = [r['label'] for r in rs]
        if 0 < sum(y) < len(y):
            aucs.append(roc_auc_score(y, [r[key] for r in rs]))
    return float(np.mean(aucs)), len(aucs)

out = {}
for key in ('at', 'pred'):
    a, n = per_imp_auc(key)
    out[key] = {'per_impression_auc': round(a, 6), 'n_impressions': n}
    print(f"score row '{key}': per-impression AUC = {a:.6f}  (n={n} impressions)")
at = np.array([r['at'] for r in recs]); pr = np.array([r['pred'] for r in recs])
out['corr_at_pred'] = round(float(np.corrcoef(at, pr)[0, 1]), 6)
print(f"corr(at, pred) = {out['corr_at_pred']}")
print("RESULT_JSON " + json.dumps(out))
