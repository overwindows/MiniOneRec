"""Probe: is the A7.3 pairwise score a RELEVANCE signal or an ANSWER-TOKEN
shortcut?

src/sft_mind_pointwise_pairwise.py scores a candidate with

    pos = (labels != -100).nonzero()[0]        # == L-1, the ANSWER token row
    score = logits[i, pos, YES_ID]

In causal LM, logits[..., t, :] predicts token t+1. The answer token occupies
L-1, so row L-1 predicts a token that does not exist, and the CE loss never
trains it (the answer token at L-1 is predicted by row L-2).

But the hidden state at L-1 is computed FROM the answer token's embedding, so
it already knows whether the sequence ended in " Yes" or " No".

DECISIVE TEST: take the SAME prompt and append " Yes" vs " No". If the
pairwise score separates the two regardless of relevance, the BT term can be
satisfied by reading the answer token and carries no relevance signal.

We compare three quantities per candidate:
    at   = logits[L-1, YES] - logits[L-1, NO]   (what the trainer reads)
    pred = logits[L-2, YES] - logits[L-2, NO]   (the row that predicts the answer)
    and the same-prompt gap  at(" Yes") - at(" No")

Usage:
    python src/probe_pairwise_shortcut.py <ckpt> <n_candidates> [split]
"""
import sys, os, json, collections
sys.path.insert(0, '/import/ml-sc-scratch5/chenw/MiniOneRec')
os.chdir('/import/ml-sc-scratch5/chenw/MiniOneRec')
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data import MINDPointwiseSFTDataset

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'output_dir/small_runs/hp_neg1/final_checkpoint'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
SPLIT = sys.argv[3] if len(sys.argv) > 3 else 'dev'

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

rows = []
with torch.no_grad():
    for i in range(min(N, len(ds))):
        s = ds[i]
        ids = s['input_ids']
        L = ids.shape[0]
        vpos = int(np.nonzero(s['labels'].numpy() != -100)[0][0])
        assert vpos == L - 1, (vpos, L)
        prompt_ids = ids[:-1].tolist()          # everything before the answer

        def score_at(answer_id):
            """logit row at the ANSWER position, for a given answer token."""
            seq = torch.tensor([prompt_ids + [answer_id]], device='cuda')
            lg = model(seq).logits[0].float()
            return float(lg[L - 1, YES] - lg[L - 1, NO])

        def score_pred(answer_id):
            """logit row that PREDICTS the answer token."""
            seq = torch.tensor([prompt_ids + [answer_id]], device='cuda')
            lg = model(seq).logits[0].float()
            return float(lg[L - 2, YES] - lg[L - 2, NO])

        at_yes = score_at(YES)
        at_no = score_at(NO)
        pr_yes = score_pred(YES)
        pr_no = score_pred(NO)
        rows.append({'label': int(s['label']),
                     'at_yes': at_yes, 'at_no': at_no,
                     'pr_yes': pr_yes, 'pr_no': pr_no})

lab = np.array([r['label'] for r in rows])
gap_at = np.array([r['at_yes'] - r['at_no'] for r in rows])
gap_pr = np.array([r['pr_yes'] - r['pr_no'] for r in rows])

print(f"n candidates: {len(rows)}")
print()
print("SAME-PROMPT GAP  score(' Yes' ending) - score(' No' ending)")
print(f"  row AT the answer   (what the trainer reads): "
      f"mean {gap_at.mean():+.4f}  sd {gap_at.std():.4f}  "
      f"|gap|>1 in {100*(np.abs(gap_at)>1).mean():.1f}% of candidates")
print(f"  row PREDICTING the answer (eval semantics)  : "
      f"mean {gap_pr.mean():+.4f}  sd {gap_pr.std():.4f}  "
      f"|gap|>1 in {100*(np.abs(gap_pr)>1).mean():.1f}% of candidates")
print()
# How much of the variance in the trainer's score is explained by the label
# (i.e. by WHICH ANSWER TOKEN the sequence ends in)?
at_yes = np.array([r['at_yes'] for r in rows])
at_no = np.array([r['at_no'] for r in rows])
for nm, arr in (("at_yes", at_yes), ("at_no", at_no)):
    m0, m1 = arr[lab == 0].mean(), arr[lab == 1].mean()
    # between-label variance / total variance
    grand = arr.mean()
    between = lab.mean() * (m1 - grand) ** 2 + (1 - lab.mean()) * (m0 - grand) ** 2
    total = arr.var()
    print(f"  {nm}: mean|pos {m1:+.4f}  mean|neg {m0:+.4f}  "
          f"between/total var = {between/total:.3f}")
print()
print("INTERPRETATION")
print("  If the trainer's score separates ' Yes' from ' No' endings on the SAME")
print("  prompt (large |gap| regardless of relevance), then the BT term can be")
print("  driven to zero by reading the answer token -- it carries no ranking")
print("  signal, only a label leak.")

out = {'n': len(rows),
       'gap_at_mean': round(float(gap_at.mean()), 4),
       'gap_at_sd': round(float(gap_at.std()), 4),
       'gap_at_frac_abs_gt1': round(float((np.abs(gap_at) > 1).mean()), 4),
       'gap_pred_mean': round(float(gap_pr.mean()), 4),
       'gap_pred_sd': round(float(gap_pr.std()), 4),
       'gap_pred_frac_abs_gt1': round(float((np.abs(gap_pr) > 1).mean()), 4)}
print("RESULT_JSON " + json.dumps(out))
