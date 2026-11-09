"""Decisive probe for the A7.3 pairwise loss (src/sft_mind_pointwise_pairwise.py).

THREE MEASUREMENTS, all on the real MIND_small train split:

 (1) SCORE ROW / LABEL LEAK. The trainer scores a candidate with
         pos  = (labels != -100).nonzero()[0]      # == L-1, the ANSWER token row
         score = logits[i, pos, YES_ID]            # a SINGLE logit
     In causal LM, row t predicts token t+1, so row L-1 predicts a token that
     does not exist; the CE loss never trains it (the answer at L-1 is predicted
     by row L-2). But the hidden state at L-1 is computed FROM the answer
     token's embedding -- and during training the answer token IS the label.
     So the trainer's score may be reading the label off the input.
     We measure per-impression AUC of:
         trainer : logits[L-1, YES]                       (what the trainer uses)
         leakfree: logits[L-2, YES] - logits[L-2, NO]    (eval semantics)
     If `trainer` scores ~1.0 per impression it is a label leak, and the BT
     term is a no-op that is trivially satisfied.

 (2) PAIRING RATE. The BT term only forms a within-impression pair when a
     positive and a negative of the SAME impression land in the SAME
     micro-batch. Measured under the production sampler and under an
     impression-packed sampler.

 (3) BT MARGIN. The magnitude of (score_pos - score_neg) under each pairing,
     which is what actually drives the gradient.

Usage:
    python src/probe_pairwise_defects.py <ckpt> <n_candidates> [split]
"""
import sys, os, json, collections, random
sys.path.insert(0, '/import/ml-sc-scratch5/chenw/MiniOneRec')
os.chdir('/import/ml-sc-scratch5/chenw/MiniOneRec')
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import MINDPointwiseSFTDataset
from sklearn.metrics import roc_auc_score

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'output_dir/small_runs/hp_neg1/final_checkpoint'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SPLIT = sys.argv[3] if len(sys.argv) > 3 else 'train'

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True, local_files_only=True)
tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id; tok.padding_side = 'left'
YES = tok.encode(" Yes", add_special_tokens=False)[0]
NO = tok.encode(" No", add_special_tokens=False)[0]
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16,
                                             local_files_only=True).cuda().eval()

ds = MINDPointwiseSFTDataset(
    behaviors_path=f'data/MIND_small/{SPLIT}/behaviors.tsv',
    news_path=f'data/MIND_small/{SPLIT}/news.tsv',
    tokenizer=tok, max_len=2048, sample=-1, seed=42, max_history=30,
    neg_ratio=2.0, hard_neg_ratio=0.5, use_abstract=True,
    use_chat_template=True, return_meta=True)
print(f"loaded {len(ds.samples)} samples / {len(set(s['impression_id'] for s in ds.samples))} impressions")

# ---------- (1) SCORE ROW ----------
# The dataset SHUFFLES all samples, so the first N samples are N different
# impressions -- every impression would have a single candidate and no
# per-impression AUC would be computable. Select WHOLE impressions instead.
by_imp = collections.OrderedDict()
for idx, s in enumerate(ds.samples):
    by_imp.setdefault(s['impression_id'], []).append(idx)
keep_imps = list(by_imp.keys())[:N]
keep_idx = [i for imp in keep_imps for i in by_imp[imp]]
print(f"measuring {len(keep_imps)} whole impressions / {len(keep_idx)} candidates")
recs = []
with torch.no_grad():
    for i in keep_idx:
        s = ds[i]
        ids = s['input_ids'].unsqueeze(0).cuda()
        lg = model(ids).logits[0].float()
        L = ids.shape[1]
        vpos = int(np.nonzero(s['labels'].numpy() != -100)[0][0])
        assert vpos == L - 1
        recs.append({'imp': s['impression_id'], 'label': int(s['label']),
                     'trainer': float(lg[vpos, YES]),
                     'leakfree': float(lg[vpos - 1, YES] - lg[vpos - 1, NO]),
                     'yes_minus_no_at': float(lg[vpos, YES] - lg[vpos, NO])})

def per_imp_auc(key):
    g = collections.defaultdict(list)
    for r in recs: g[r['imp']].append(r)
    aucs = [roc_auc_score([r['label'] for r in rs], [r[key] for r in rs])
            for rs in g.values() if 0 < sum(r['label'] for r in rs) < len(rs)]
    return float(np.mean(aucs)), len(aucs)

print("\n(1) PER-IMPRESSION AUC of the candidate score definitions")
res = {}
for key in ('trainer', 'leakfree', 'yes_minus_no_at'):
    a, n = per_imp_auc(key)
    res[key] = round(a, 6)
    print(f"    {key:18s} per-impression AUC = {a:.6f}  (n={n})")
print("    NOTE: 'trainer' ~1.0 would mean the score reads the LABEL, not the")
print("          relevance of the candidate.")

# ---------- (2) PAIRING RATE ----------
print("\n(2) WITHIN-IMPRESSION PAIRING RATE")
S = ds.samples
lab = [(s['impression_id'], s['label']) for s in S]
def est_len(s):
    cand = s.get("candidate", {}) or {}; hist = s.get("history", []) or []
    return (len(cand.get("text", "")) + len(" ".join(h.get("text", "") for h in hist))) // 4
lengths = [est_len(s) for s in S]

def pair_rate(seq, mb, nb=30000):
    hit = 0; pairs = 0
    n = min(nb, len(seq) // mb)
    for b in range(n):
        batch = seq[b * mb:(b + 1) * mb]
        by = collections.defaultdict(lambda: ([], []))
        for j, (i, l) in enumerate(batch): by[i][l].append(j)
        c = sum(min(len(P), len(N)) for _, (P, N) in by.items())
        if c: hit += 1
        pairs += c
    return hit / n, pairs / n

# production: length-grouped sampler (MINIONEREC_LENGTH_GROUPED=1)
for mb in (4, 8):
    rng = random.Random(42); idx = list(range(len(S))); rng.shuffle(idx)
    mega = max(50 * mb, mb * 8)
    grouped = []
    for i in range(0, len(idx), mega):
        grouped.extend(sorted(idx[i:i + mega], key=lambda j: lengths[j]))
    h, p = pair_rate([lab[j] for j in grouped], mb)
    res[f'prod_pair_rate_mb{mb}'] = round(h, 6)
    print(f"    production length-grouped  mb={mb}: batches with a pair {h:.4%}, mean pairs/batch {p:.3f}")

# fix: impression-packed buckets of a fixed size
def pack_impressions(mb, seed=42):
    """Greedily pack WHOLE impressions into fixed-size buckets."""
    by_imp = collections.OrderedDict()
    for j, s in enumerate(S): by_imp.setdefault(s['impression_id'], []).append(j)
    imps = list(by_imp.keys()); random.Random(seed).shuffle(imps)
    buckets = []; cur = []
    for i in imps:
        g = by_imp[i]
        if len(g) > mb:                     # split an oversized impression
            for k in range(0, len(g), mb):
                part = g[k:k + mb]
                if cur: buckets.append(cur); cur = []
                buckets.append(part)
            continue
        if len(cur) + len(g) > mb:
            buckets.append(cur); cur = []
        cur.extend(g)
    if cur: buckets.append(cur)
    return buckets

for mb in (4, 8):
    buckets = pack_impressions(mb)
    seq = [lab[j] for b in buckets for j in b]
    h, p = pair_rate(seq, mb)
    res[f'fix_pair_rate_mb{mb}'] = round(h, 6)
    print(f"    impression-packed          mb={mb}: batches with a pair {h:.4%}, mean pairs/batch {p:.3f}")

# ---------- (3) BT MARGIN ----------
print("\n(3) BT MARGIN  (score_pos - score_neg); the gradient driver")
sc = {r['imp']: [] for r in recs}
for r in recs: sc[r['imp']].append(r)
margins_tr = []; margins_lf = []
for imp, rs in sc.items():
    P = [r for r in rs if r['label'] == 1]; Nn = [r for r in rs if r['label'] == 0]
    if P and Nn:
        margins_tr.append(np.mean([r['trainer'] for r in P]) - np.mean([r['trainer'] for r in Nn]))
        margins_lf.append(np.mean([r['leakfree'] for r in P]) - np.mean([r['leakfree'] for r in Nn]))
print(f"    trainer  score: margin mean {np.mean(margins_tr):+.4f}  sd {np.std(margins_tr):.4f}")
print(f"    leakfree score: margin mean {np.mean(margins_lf):+.4f}  sd {np.std(margins_lf):.4f}")
res['margin_trainer_mean'] = round(float(np.mean(margins_tr)), 4)
res['margin_leakfree_mean'] = round(float(np.mean(margins_lf)), 4)

print("\nRESULT_JSON " + json.dumps(res))
