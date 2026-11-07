#!/usr/bin/env python3
"""Does the position->label relationship exist in TRAIN as well as DEV?

WHY
---
diag_position_leak.py found that on dev the click rate decays monotonically
with the candidate's slot in the impression list (9.29% at position 0-3 down
to 1.79% at position 92-291). If that relationship is a property of how the
MIND impression lists were built, it will also hold in TEST and the ranker's
`position` feature is legitimate. If it is a DEV-ONLY artifact (e.g. the dev
behaviors file was reordered or de-duplicated differently), then the ranker is
exploiting a train-only artifact and the gain will NOT transfer.

This reads the raw behaviors.tsv files directly -- no feature table needed --
so it is independent of any preprocessing we did.

Usage: python src/diag_position_train.py
"""
import numpy as np

SPLITS = {
    "train": "data/MIND_large/train/behaviors.tsv",
    "dev": "data/MIND_large/dev/behaviors.tsv",
}


def scan(path, max_imps=None):
    """Return (positions, labels) for every candidate row in the split."""
    pos, lab = [], []
    n_imp = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            for j, tk in enumerate(parts[4].split()):
                if "-" not in tk:
                    continue
                pos.append(j)
                lab.append(1 if tk.rsplit("-", 1)[1] == "1" else 0)
            n_imp += 1
            if max_imps and n_imp >= max_imps:
                break
    return np.asarray(pos), np.asarray(lab)


def report(name, pos, lab):
    print(f"\n=== {name}: click rate by position decile ===")
    print(f"{'decile':>8} {'pos range':>14} {'rows':>12} {'click rate':>11}")
    edges = np.percentile(pos, np.linspace(0, 100, 11))
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        sel = (pos >= lo) & (pos <= hi if i == 9 else pos < hi)
        if not sel.any():
            continue
        print(f"{i+1:>8} {f'{lo:.0f}-{hi:.0f}':>14} {sel.sum():>12,} "
              f"{lab[sel].mean():>11.4f}")

    # The single number that matters: AUC of position ALONE as a predictor.
    # (lower position = more likely clicked, so use -position)
    order = np.argsort(-pos, kind="stable")
    ranks = np.empty(len(pos))
    ranks[order] = np.arange(1, len(pos) + 1)
    npos = lab.sum()
    nneg = len(lab) - npos
    auc = (ranks[lab == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print(f"  position-alone AUC (global, -pos) = {auc:.4f}")
    print(f"  overall click rate = {lab.mean():.4f}   rows={len(lab):,}")

    # Within-impression AUC of position alone: the metric the leaderboard uses.
    # We need the impression boundaries, so recompute with group ids.
    return auc


def scan_grouped(path, max_imps=None):
    pos, lab, gid = [], [], []
    n_imp = 0
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            for j, tk in enumerate(parts[4].split()):
                if "-" not in tk:
                    continue
                pos.append(j)
                lab.append(1 if tk.rsplit("-", 1)[1] == "1" else 0)
                gid.append(n_imp)
            n_imp += 1
            if max_imps and n_imp >= max_imps:
                break
    return (np.asarray(pos), np.asarray(lab), np.asarray(gid))


def per_imp_auc(pos, lab, gid):
    """Mean per-impression AUC of -position as a predictor."""
    order = np.lexsort((-pos, gid))
    gs, ps, ys = gid[order], pos[order], lab[order]
    new_g = np.empty(len(gs), dtype=bool)
    new_g[0] = True
    new_g[1:] = gs[1:] != gs[:-1]
    st = np.flatnonzero(new_g)
    en = np.append(st[1:], len(gs))
    aucs = []
    for s0, e0 in zip(st, en):
        yy = ys[s0:e0]
        npos = yy.sum()
        nneg = len(yy) - npos
        if npos == 0 or nneg == 0:
            continue
        # rank of -pos ascending == rank of pos descending
        r = np.empty(e0 - s0)
        o = np.argsort(-ps[s0:e0], kind="stable")
        r[o] = np.arange(1, e0 - s0 + 1)
        aucs.append((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return float(np.mean(aucs)), len(aucs)


def main():
    for name, path in SPLITS.items():
        print(f"Scanning {name} ...", flush=True)
        pos, lab, gid = scan_grouped(path)
        report(name, pos, lab)
        a, n = per_imp_auc(pos, lab, gid)
        print(f"  position-alone PER-IMPRESSION AUC = {a:.4f}  (n={n:,} imps)")


if __name__ == "__main__":
    main()
