"""Build a UNIFORM random sample of the MIND_large DEV split as a scoring chunk.

WHY A SUBSET INSTEAD OF A FULL PART
-----------------------------------
We want to compare the DPO / RLCD preference checkpoints against the SFT base
(ck45500) on dev. The ck45500 dev scores for ALL 376,471 impressions already
exist (results/ranker/dev_scores/ck45500/), so the comparison can be PAIRED:
per impression, AUC(DPO) - AUC(ck45500) on the SAME impression ids.

A paired delta has a much smaller standard error than the difference of two
independent AUCs:

    independent:  se(AUC_A - AUC_B) ~ sqrt(2) * 0.00043 = 0.00061
    paired (N):   se ~ 0.0485 / sqrt(N)

With N = 20,000 the paired SE is ~0.00034, i.e. a +0.001 AUC effect is a
2.9-sigma signal, while the same effect on independent AUCs would be a
1.6-sigma noise. The subset is therefore a DECISION instrument, not the final
reported number: the winner is afterwards scored on the FULL dev split.

The sample is uniform (not the over-length "parity" subset, which is a biased
tail of the distribution), drawn with a fixed seed so it is reproducible, and
the impression ORDER is preserved so the file looks like a normal chunk.

Usage:
    python scripts/build_dev_subset.py --n 20000 --seed 42 --part 90
"""
import argparse
import os
import random

CHUNK_DIR = "data/MIND_large/dev/chunks/sub/d00"
PARTS = ["00", "01", "02", "03", "04", "05"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--part", default="90",
                    help="output chunk index (must not collide with 00..05)")
    args = ap.parse_args()

    out = f"{CHUNK_DIR}/chunk_{args.part}.tsv"
    if os.path.exists(out):
        raise SystemExit(f"FATAL: {out} already exists -- remove it or pick another --part")

    lines = []
    for p in PARTS:
        with open(f"{CHUNK_DIR}/chunk_{p}.tsv") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    lines.append(line)
    total = len(lines)
    print(f"dev impressions available: {total:,}")

    if args.n >= total:
        raise SystemExit(f"FATAL: --n {args.n} >= dev size {total}")

    rng = random.Random(args.seed)
    idx = sorted(rng.sample(range(total), args.n))
    picked = [lines[i] for i in idx]

    os.makedirs(CHUNK_DIR, exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(picked) + "\n")
    print(f"wrote {out}: {len(picked):,} impressions (seed {args.seed})")

    # Sanity: the sample must be spread over the whole dev split, not clustered
    # in one part. Print the per-part hit counts.
    per = {}
    for i in idx:
        per[i * len(PARTS) // total] = per.get(i * len(PARTS) // total, 0) + 1
    print("per-source-part counts:",
          {PARTS[k]: v for k, v in sorted(per.items())})


if __name__ == "__main__":
    main()
