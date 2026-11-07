#!/usr/bin/env python3
"""Replace the llm_score column of a feature table with a weighted ensemble
of multiple LLM score files.

Each score file: impression_id<TAB>score1<TAB>score2... (one line per
impression, scores in candidate order — same order as the feature table
rows within that impression).

Usage:
    python src/build_ensemble_feats.py \
        --base results/ranker/ck45500_dev_features_causalpop.tsv \
        --scores results/ranker/lg9b_ckpt45500_dev_scores.tsv \
                 results/ranker/lg9b_ep082_dev_scores.tsv \
                 results/ranker/dedupbf_ep275_dev_scores.tsv \
        --weights 0.5 0.3 0.2 \
        --out results/ranker/ckens_dev_features_causalpop.tsv
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert len(args.scores) == len(args.weights)
    total = sum(args.weights)
    ws = [w / total for w in args.weights]

    tables = []
    for path in args.scores:
        t = {}
        with open(path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    t[p[0]] = [float(v) for v in p[1:]]
        tables.append(t)
        print(f"loaded {path}: {len(t)} impressions", flush=True)

    n = 0
    n_miss = 0
    with open(args.base) as f, open(args.out, "w") as out:
        header = f.readline().rstrip("\n").split("\t")
        i_imp = header.index("impression_id")
        i_llm = header.index("llm_score")
        out.write("\t".join(header) + "\n")
        cur_imp = None
        cur_pos = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            imp = p[i_imp]
            if imp != cur_imp:
                cur_imp = imp
                cur_pos = 0
            ens = 0.0
            ok = True
            for w, t in zip(ws, tables):
                s = t.get(imp)
                if s is None or cur_pos >= len(s):
                    ok = False
                    break
                ens += w * s[cur_pos]
            if ok:
                p[i_llm] = f"{ens:.6f}"
            else:
                n_miss += 1
            out.write("\t".join(p) + "\n")
            cur_pos += 1
            n += 1
            if n % 5_000_000 == 0:
                print(f"  {n:,} rows", flush=True)
    print(f"Wrote {n:,} rows -> {args.out}  (missing-ensemble rows kept base llm_score: {n_miss})",
          flush=True)


if __name__ == "__main__":
    main()