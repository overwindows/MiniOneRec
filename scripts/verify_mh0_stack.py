#!/usr/bin/env python3
"""Does the MAX_HISTORY win survive the STACK ranker?

WHY
---
The MAX_HISTORY lever was measured at the LLM level: on the same 40,000 dev
impressions, ckpt-45500 with an unlimited history cap scored +0.00392 AUC
(SE 0.00036, t=+10.77) over the same weights with the cap at 30.

But the SUBMITTED system is not the LLM alone -- it is the v6 stack ranker
(33 features) applied on top of llm_score. Two things could eat the win:

  1. The ranker was FITTED on cap-30 llm_scores, so a shifted llm_score
     distribution (the cap change moves llm_score by mean +0.0174, sd 0.2432)
     could land off the fitted calibration.
  2. The ranker also consumes llm_rank / llm_z, the within-impression
     transforms of llm_score, which change with the cap too.

This script answers that with ZERO GPU: it slices the canonical dev feature
table down to the 40,000 chunk_90 impressions, swaps ONLY the llm_score
column between the cap-30 and the unlimited dumps, applies the v6 ranker to
both, and compares the STACK per-impression AUC paired.

The swap is exact because
  * the two arms differ ONLY in the history cap (same ckpt, same code path,
    same node type, same chunk), and
  * the score dumps are in candidate order, which is the row order of the
    feature table (both come from the same behaviors candidate list).

It also reports the LLM-only AUC of each source, as a sanity check that the
canonical feature table's llm_score is interchangeable with the screen dump.

Usage:
    python scripts/verify_mh0_stack.py \
        --features results/ranker/ck45500_dev_features_causalpop.tsv \
        --chunk data/MIND_large/dev/chunks/sub/d00/chunk_90.tsv \
        --base results/ranker/dev_scores/screen90/base/hs_hybrid_base_90.tsv \
        --new  results/ranker/dev_scores/screen90/base_mh0/hs_hybrid_base_mh0_90.tsv \
        --model results/ranker/ranker_v6_train_popwit.txt
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def load_scores(path):
    """impression_id -> list of llm scores (candidate order)."""
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                out[p[0]] = [float(v) for v in p[1:]]
    return out


def load_chunk(path):
    """impression_id -> (news_ids, labels)."""
    out = {}
    order = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            imp = p[0]
            cands = p[4].split()
            nids, labels = [], []
            for tk in cands:
                nid, lab = tk.rsplit("-", 1)
                nids.append(nid)
                labels.append(1 if lab == "1" else 0)
            out[imp] = (nids, labels)
            order.append(imp)
    return out, order


def per_impression_auc(scores, labels):
    if len(scores) == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rsum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def slice_features(feat_path, keep_ids, base, new, out_base, out_new):
    """Stream the dev feature table, keep the chunk_90 impressions, and write
    two copies with the llm_score column swapped."""
    with open(feat_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        llm_i = header.index("llm_score")
        nid_i = header.index("news_id")
        imp_i = header.index("impression_id")
        # per-impression cursor into the score list
        cur = None
        pos = 0
        n_rows = 0
        n_bad_order = 0
        with open(out_base, "w") as fb, open(out_new, "w") as fn:
            fb.write("\t".join(header) + "\n")
            fn.write("\t".join(header) + "\n")
            for line in f:
                p = line.rstrip("\n").split("\t")
                imp = p[imp_i]
                if imp not in keep_ids:
                    continue
                if imp != cur:
                    cur = imp
                    pos = 0
                sb = base.get(imp)
                sn = new.get(imp)
                if sb is None or sn is None:
                    raise SystemExit(f"missing scores for impression {imp}")
                if pos >= len(sb):
                    raise SystemExit(f"too many feature rows for impression {imp}")
                # sanity: the feature row's news_id must match the candidate
                # at the same position, otherwise the swap would misalign.
                if p[nid_i] != keep_ids[imp][0][pos]:
                    n_bad_order += 1
                pb = list(p)
                pn = list(p)
                pb[llm_i] = f"{sb[pos]:.6f}"
                pn[llm_i] = f"{sn[pos]:.6f}"
                fb.write("\t".join(pb) + "\n")
                fn.write("\t".join(pn) + "\n")
                pos += 1
                n_rows += 1
    return n_rows, n_bad_order


def stack_auc(feat_path, model, behaviors_path, tmpdir, tag):
    """Apply the ranker and return the per-impression AUC dict."""
    pred = os.path.join(tmpdir, f"pred_{tag}.txt")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    subprocess.run(
        [sys.executable, "src/apply_ranker_devpop.py",
         "--features", feat_path, "--model", model, "--output", pred],
        check=True, env=env)
    gt, _ = load_chunk(behaviors_path)
    out = {}
    with open(pred) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            imp = parts[0]
            if imp not in gt:
                continue
            nids, labels = gt[imp]
            pos = {n: i for i, n in enumerate(nids)}
            sc = [0.0] * len(nids)
            for rank, n in enumerate(parts[1:]):
                if n in pos:
                    sc[pos[n]] = -float(rank)
            a = per_impression_auc(sc, labels)
            if a is not None:
                out[imp] = a
    return out


def llm_auc(chunk, scores):
    vals = []
    for imp, (nids, labels) in chunk.items():
        s = scores.get(imp)
        if s is None or len(s) != len(labels):
            continue
        a = per_impression_auc(s, labels)
        if a is not None:
            vals.append(a)
    return np.array(vals)


def report(label, ab, an):
    d = an - ab
    se = d.std(ddof=1) / np.sqrt(len(d))
    delta = d.mean()
    t = delta / se if se > 0 else float("nan")
    n_up = int((d > 1e-12).sum())
    n_down = int((d < -1e-12).sum())
    print(f"  {label}")
    print(f"    base AUC : {ab.mean():.5f}")
    print(f"    new  AUC : {an.mean():.5f}")
    print(f"    PAIRED delta: {delta:+.5f}  (SE {se:.5f}, t={t:+.2f})")
    print(f"    improved/worse/tied: {n_up:,} / {n_down:,} / {len(d)-n_up-n_down:,}")
    print(f"    95% CI   : [{delta-1.96*se:+.5f}, {delta+1.96*se:+.5f}]")
    verdict = ("BETTER" if delta > 0 and t > 2 else
               "WORSE" if delta < 0 and t < -2 else
               "NO SIGNIFICANT CHANGE")
    print(f"    VERDICT  : {verdict} (|t| > 2 required)")
    return delta, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="results/ranker/ck45500_dev_features_causalpop.tsv")
    ap.add_argument("--chunk", default="data/MIND_large/dev/chunks/sub/d00/chunk_90.tsv")
    ap.add_argument("--base", default="results/ranker/dev_scores/screen90/base/hs_hybrid_base_90.tsv")
    ap.add_argument("--new", default="results/ranker/dev_scores/screen90/base_mh0/hs_hybrid_base_mh0_90.tsv")
    ap.add_argument("--model", default="results/ranker/ranker_v6_train_popwit.txt")
    args = ap.parse_args()

    base = load_scores(args.base)
    new = load_scores(args.new)
    chunk, order = load_chunk(args.chunk)
    keep = set(chunk)
    print(f"chunk impressions: {len(chunk):,}   base scores: {len(base):,}   new scores: {len(new):,}")

    # ---- LLM-only sanity check -------------------------------------------
    ab = llm_auc(chunk, base)
    an = llm_auc(chunk, new)
    print("\n=== LLM-only (per-impression AUC, chunk_90) ===")
    report("LLM only", ab, an)

    # The canonical feature table's own llm_score must be close to the
    # screen dump's, otherwise the swap below is not a clean control.
    canon = {}
    with open(args.features) as f:
        header = f.readline().rstrip("\n").split("\t")
        li = header.index("llm_score")
        ii = header.index("impression_id")
        cur, pos = None, 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[ii]
            if imp not in keep:
                continue
            if imp != cur:
                cur, pos = imp, 0
            canon.setdefault(imp, []).append(float(p[li]))
            pos += 1
    ac = llm_auc(chunk, canon)
    print(f"\n  sanity: canonical feature-table llm_score AUC = {ac.mean():.5f} "
          f"(n={len(ac):,})  vs screen base {ab.mean():.5f}")

    # ---- stack comparison -------------------------------------------------
    with tempfile.TemporaryDirectory(dir="results/ranker") as tmp:
        fb = os.path.join(tmp, "feats_base.tsv")
        fn = os.path.join(tmp, "feats_new.tsv")
        # `chunk` (not `keep`) is the lookup: chunk[imp] = (news_ids, labels),
        # so chunk[imp][0][pos] is the news_id of the pos-th candidate.
        n_rows, n_bad = slice_features(args.features, chunk, base, new, fb, fn)
        print(f"\n  sliced {n_rows:,} rows for {len(keep):,} impressions "
              f"(news_id order mismatches: {n_bad})")
        if n_bad:
            print("  FATAL: the feature table's row order is not the candidate order")
            sys.exit(1)

        print("\n=== STACK (v6 ranker) ===")
        sb = stack_auc(fb, args.model, args.chunk, tmp, "base")
        sn = stack_auc(fn, args.model, args.chunk, tmp, "new")
        common = sorted(set(sb) & set(sn))
        print(f"  impressions scored by both arms: {len(common):,}")
        ab2 = np.array([sb[i] for i in common])
        an2 = np.array([sn[i] for i in common])
        report("STACK v6", ab2, an2)


if __name__ == "__main__":
    main()
