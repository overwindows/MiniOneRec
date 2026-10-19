# MIND dataset — what we have and how it compares to the paper

> Companion doc to `ENSEMBLE_HANDOFF.md` (§7, "Getting the dataset"). This focuses
> specifically on **our staged dataset** and reconciles it against the official
> figures in the MIND tech report (Wu et al., *MIND: A Large-scale Dataset for
> News Recommendation*, ACL 2020).
>
> - Paper: https://aclanthology.org/2020.acl-main.331/
> - Official dataset page: https://msnews.github.io/

---

## 1. Where our copy lives

| Location | Scope | Used by |
|---|---|---|
| ADLS Gen1 `msn-dni-c09`: `shares/users/wuc/data/MIND_large/{train,dev,test}` | MIND-large (competition repack) | All AML runs via `data_root` |
| ADLS Gen1 `msn-dni-c09`: `shares/users/wuc/data/MIND_small/{train,dev}` | MIND-small (no test) | Fast develop/test loops |
| `Q:\MiniOneRec\.offline\dev_behaviors.tsv` (376,471) / `test_behaviors.tsv` (2,370,727) | dev/test behaviors only | Offline score-space analysis |

Per-split files: `behaviors.tsv`, `news.tsv`, `entity_embedding.vec`, `relation_embedding.vec`;
train/dev also carry `cf_scores.tsv` and RL pointwise parquets (`rl_pointwise_neg{2p0,3p0}_*.parquet`).

---

## 2. Our measured counts (directly from the files, 2026-10-19)

Counted by streaming each file from the datastore — row counts for news/behaviors;
for users/clicks a distinct-user set and click-label tally over behaviors.

### 2a. MIND_large (our bundle)

| Split | News rows | Behaviors / impressions | Distinct users | Clicks |
|---|---|---|---|---|
| train | 101,527 | 2,232,748 | 711,222 | 3,383,656 |
| dev   |  72,023 |   376,471 | 255,990 |   574,845 |
| test  | 120,961 | 2,370,727 | 702,005 | 0 (no labels) |

- **Test has no click labels** — candidates are unscored `newsid`s (label column absent), which is why offline AUC/MRR are computed on **dev** only.
- News rows **overlap across splits** (same articles appear in multiple splits), so per-split row counts are **not** disjoint; the ~summed 294k is not unique-news.

### 2b. MIND_small (our bundle)

| Split | News rows | Behaviors / impressions | Distinct users | Clicks |
|---|---|---|---|---|
| train | 51,282 | 156,965 | 50,000 | 236,344 |
| dev   | 42,416 |  73,152 | 50,000 | 111,383 |
| test  | — not present in bundle — | | | |

---

## 3. Official figures (MIND tech report, ACL 2020)

### 3a. MIND-large — aggregate (paper Table 2)

| Metric | Official |
|---|---|
| Users | 1,000,000 |
| News (articles) | 161,013 |
| Behaviors / impressions | 15,777,377 |
| Click behaviors | 24,155,470 |
| Entities | 3,299,687 |
| News categories | 20 |
| Collection window | 6 weeks, 2019-10-12 → 2019-11-22 |
| User threshold | random sample of 1M users, each ≥5 news clicks |

### 3b. MIND-large — per-split *impressions* (paper §3.2; the only per-split numbers the paper gives)

| Split | Impressions (paper) | Ours | Δ |
|---|---|---|---|
| Train | 2,186,683 | 2,232,748 | +46,065 (+2.1%) |
| Val/Dev | 365,200 | 376,471 | +11,271 (+3.1%) |
| Test | 2,341,619 | 2,370,727 | +29,108 (+1.2%) |

### 3c. MIND-small

Official: **50,000 users** randomly sampled (with their behavior logs) from MIND.
The paper does **not** report per-split users/news/impressions/clicks for MIND-small
(and "MIND-small" is not in the 2020 paper text at all — it was added later at
msnews.github.io).

---

## 4. Reconciliation & discrepancies

1. **Impression counts track the paper closely but are not identical** (+1–3% per split, largest on dev). Our bundle is a downstream *competition repack* (Codabench MIND comp #13967), so modest re-sampling/filtering vs. the published split is expected. Same-day partitions, not byte-identical files.
2. **Unique news**: paper aggregate = 161,013. Our per-split news rows sum to ~294k **not** because there are 294k unique articles, but because news rows are reused across splits. We have not computed our own unique-news count; treat the paper's 161,013 as authoritative unless a dedup is run.
3. **Users**: paper reports only the 1M aggregate. Our train has 711,222 distinct users, test 702,005 — both subsets of the full 1M cohort, consistent with the paper's "random 1M-user sample" and with overlap between job/train and job/test cohorts.
4. **Test labels**: paper's test impressions (2,341,619) are **unlabeled** — matches ours having 0 clicks. This is by design (Codabench scores submitted rankings against hidden labels).
5. **MIND-small split**: our train AND dev each show **50,000 distinct users** — exactly the official MIND-small cohort size. Good sign the bundle is the authentic MIND-small. **Our MIND-small bundle has no `test/`** (only `train/` and `dev/`, plus `.extracted/`); anyone needing MIND-small test must fetch `MINDsmall_test.zip` from the public release.
6. **Our extra derived files** (cf_scores, RL parquets) are not part of the MIND release — they are team-generated (collaborative-filtering scores and pointwise RL training sets).

---

## 5. Getting the exact numbers again

Our counts are reproducible from the datastore with the WebHDFS streaming recipe in
`ENSEMBLE_HANDOFF.md §7b` (count lines for news/behaviors; tally distinct user ids and
`-1` label suffixes for clicks). The paper figures come from **Table 2** (§3a) and
**§3.2** (§3b); MIND-small's 50k users from msnews.github.io.

Public source (anyone, no Azure): `https://mind2016small.blob.core.windows.net/release/`
— `MINDlarge_{train,dev,test}.zip`, `MINDsmall_{train,dev,test}.zip`.

---

## Appendix — sources

- MIND paper (ACL 2020): https://aclanthology.org/2020.acl-main.331/
- MIND dataset page: https://msnews.github.io/
- Codabench comp #13967: https://www.codabench.org/competitions/13967/
