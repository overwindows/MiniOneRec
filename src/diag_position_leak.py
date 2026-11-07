#!/usr/bin/env python3
"""Is `position` (candidate order in the impression) predictive of the label?

WHY
---
The v4 ranker puts 27.6% of its gain on `imp_size` and 1.3% on
`position`/`position_frac`. `imp_size` is CONSTANT within an impression, so
it cannot rank candidates on its own -- it can only act through
interactions. `position` is the candidate's slot in the impression list.

If the MIND impression candidate order is randomised (it is, in the
official data), position carries no label information and any gain it
provides is an artifact of the LLM's own position bias. But if the order
correlates with the label in dev and NOT in test, the ranker would be
exploiting a train-only artifact.

This measures the click rate by position decile on dev, and the same on
the test feature table (which has no labels, so it measures the LLM score
by position instead -- if the LLM's score has a position bias in dev but
not test, that is the same failure).

Usage: python src/diag_position_leak.py
"""
import numpy as np

DEV = "results/ranker/large_dev_features_v2.tsv"
TEST = "results/ranker/lg9b_test_features_devpopcat.tsv"


def dev_position_stats(path, max_rows=4_000_000):
    """Click rate by position decile, and LLM score by position decile."""
    pos, lab, llm, imp = [], [], [], []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        i_pos, i_lab, i_llm, i_imp = (idx["position"], idx["label"],
                                      idx["llm_score"], idx["imp_size"])
        for n, line in enumerate(f):
            if n >= max_rows:
                break
            p = line.rstrip("\n").split("\t")
            pos.append(float(p[i_pos]))
            lab.append(int(p[i_lab]))
            llm.append(float(p[i_llm]))
            imp.append(float(p[i_imp]))
    return (np.array(pos), np.array(lab), np.array(llm), np.array(imp))


def test_position_stats(path, max_rows=4_000_000, stride=23):
    pos, llm, imp = [], [], []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        i_pos, i_llm, i_imp = idx["position"], idx["llm_score"], idx["imp_size"]
        for n, line in enumerate(f):
            if n % stride:
                continue
            p = line.rstrip("\n").split("\t")
            pos.append(float(p[i_pos]))
            llm.append(float(p[i_llm]))
            imp.append(float(p[i_imp]))
            if len(pos) >= max_rows:
                break
    return np.array(pos), np.array(llm), np.array(imp)


def main():
    print("Scanning DEV ...", flush=True)
    dpos, dlab, dllm, dimp = dev_position_stats(DEV)
    print(f"  {len(dpos):,} rows", flush=True)

    print("Scanning TEST ...", flush=True)
    tpos, tllm, timp = test_position_stats(TEST)
    print(f"  {len(tpos):,} rows", flush=True)

    print("\n=== DEV: click rate by position decile ===")
    print(f"{'decile':>8} {'pos range':>14} {'click rate':>11} {'llm mean':>10}")
    edges = np.percentile(dpos, np.linspace(0, 100, 11))
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        sel = (dpos >= lo) & (dpos <= hi if i == 9 else dpos < hi)
        if not sel.any():
            continue
        print(f"{i+1:>8} {f'{lo:.0f}-{hi:.0f}':>14} {dlab[sel].mean():>11.4f} "
              f"{dllm[sel].mean():>10.4f}")

    print("\n=== TEST: llm mean by position decile ===")
    print(f"{'decile':>8} {'pos range':>14} {'llm mean':>10}")
    tedges = np.percentile(tpos, np.linspace(0, 100, 11))
    for i in range(10):
        lo, hi = tedges[i], tedges[i + 1]
        sel = (tpos >= lo) & (tpos <= hi if i == 9 else tpos < hi)
        if not sel.any():
            continue
        print(f"{i+1:>8} {f'{lo:.0f}-{hi:.0f}':>14} {tllm[sel].mean():>10.4f}")

    # The decisive number: how much of the LLM's own score is explained by
    # position? If the LLM has a strong position bias in dev but not test,
    # the ranker's use of `position` will not transfer.
    print("\n=== position bias of the LLM score ===")
    for name, p, s in (("dev", dpos, dllm), ("test", tpos, tllm)):
        # correlation between position and score, within-impression would be
        # ideal but the global correlation is a good first signal
        r = np.corrcoef(p, s)[0, 1]
        # spread of the per-decile means
        e = np.percentile(p, np.linspace(0, 100, 11))
        means = []
        for i in range(10):
            sel = (p >= e[i]) & (p <= e[i + 1] if i == 9 else p < e[i + 1])
            if sel.any():
                means.append(s[sel].mean())
        means = np.array(means)
        print(f"  {name:<5} corr(pos, llm)={r:+.4f}   "
              f"decile-mean spread={means.max()-means.min():.4f}  "
              f"(llm std={s.std():.4f})")

    print("\n=== imp_size: is it constant within an impression? ===")
    # imp_size is per-impression, so it cannot rank within an impression.
    # Its 27.6% gain share must come from interactions.
    print(f"  dev  imp_size: mean={dimp.mean():.2f} std={dimp.std():.2f} "
          f"min={dimp.min():.0f} max={dimp.max():.0f}")
    print(f"  test imp_size: mean={timp.mean():.2f} std={timp.std():.2f} "
          f"min={timp.min():.0f} max={timp.max():.0f}")


if __name__ == "__main__":
    main()
