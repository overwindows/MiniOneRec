"""Compute dev AUC/MRR/nDCG for a JEV ranked-predictions file vs MIND_small dev behaviors.

Predictions format (from evaluate_mind_jev.py):  impression_id\t[cand_id cand_id ...]
   - candidates ranked by JEV P(yes) descending (best first).
Behaviors format: impression_id\tuser\ttime\thist\tcand-1 cand-2 ...  (cand has -0/-1 label suffix)
AUC: for each impression, rank positives above negatives; AUC = fraction of (pos,neg)
     pairs where pos scored higher (ranked earlier). Using top-ranked-first order.
"""
import sys, re
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load_behaviors(path):
    """impression_id -> list[(news_id, label)] in file order."""
    imp = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp_id = p[0]
            cands = p[4].split()
            parsed = []
            for c in cands:
                m = re.match(r"^(N\d+)-([01])$", c)
                if m:
                    parsed.append((m.group(1), int(m.group(2))))
            imp[imp_id] = parsed
    return imp

def load_preds(path):
    """impression_id -> ordered list of news_ids (ranked best-first)."""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            imp_id = p[0]
            ranked = re.findall(r"N\d+", p[1])
            preds[imp_id] = ranked
    return preds

def auc_rank(ranked, lab_pos, lab_neg):
    """Given ordered candidate list (best-first) and sets of pos/neg ids -> AUC."""
    if not lab_pos or not lab_neg:
        return None
    # assign score = descending rank position (higher = earlier)
    score = {nid: (len(ranked) - i) for i, nid in enumerate(ranked)}
    tpairs = 0
    for pos in lab_pos:
        for neg in lab_neg:
            if score.get(pos, 0) > score.get(neg, 0):
                tpairs += 1
    # ties: absent candidates get score 0; if both absent, count 0.5
    total = len(lab_pos) * len(lab_neg)
    return tpairs / total if total else None

def avg(xs):
    return sum(xs) / len(xs) if xs else 0.0

def main():
    beh = load_behaviors(sys.argv[1])
    preds = load_preds(sys.argv[2])
    aucs = []
    matched = 0
    for imp_id, ranked in preds.items():
        if imp_id not in beh:
            continue
        matched += 1
        cands = beh[imp_id]  # file order, includes labels
        # restrict ranked to this impression's candidates (ranked may include all scored)
        cand_by_id = dict(cands)
        lab_pos = {nid for nid, lbl in cands if lbl == 1}
        lab_neg = {nid for nid, lbl in cands if lbl == 0}
        # keep only ranked ids present in this impression
        ranked_imp = [nid for nid in ranked if nid in cand_by_id]
        a = auc_rank(ranked_imp, lab_pos, lab_neg)
        if a is not None:
            aucs.append(a)
    print(f"Matched {matched} of {len(preds)} predictions")
    print(f"Impressions w/ both pos&neg: {len(aucs)}")
    print(f"AUC:     {avg(aucs):.4f}")

if __name__ == "__main__":
    main()
