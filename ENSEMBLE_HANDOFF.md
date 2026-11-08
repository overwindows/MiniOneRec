# MIND ensemble + SOTA-push handoff runbook

**Author:** CodeGuru (Chen's assistant) · **Date:** 2026-10-19, updated 2026-11-08
**Repo:** `/import/ml-sc-scratch5/chenw/MiniOneRec` (GitHub `overwindows/MiniOneRec`, branch `sota`)

Section 0 is the 2026-11-08 rank fusion of the current submission candidates. Sections 1–8 are the 2026-10-19 handoff and are historical (team best then was 0.7144; that number is no longer current).

---

## 0. Rank fusion of the current candidates (2026-11-08)

Full-dev per-impression AUC on all 376,471 dev impressions. Test files cover all 2,370,727 test impressions. None of these zips have been submitted. The best previously submitted test score is still the v6 stack at 0.7232. Live leaderboard #1 on 2026-11-08 was dingjian at 0.7530.

Script: `results/ensemble/fuse_candidates.py`. Dev table: `results/ensemble/dev_auc.txt`.

### 0.1 Members

Each member is turned into a rank per candidate in the original behaviors order. Rank 1 is best. Score files are ranked by score descending, with ties broken by original position. Rank files are best-first news ids mapped back onto that same candidate list.

| Member | What it is | Dev source | Test source | Dev AUC |
|---|---|---|---|---|
| v6ens | v6 ranker on the ensemble feature table | `results/ranker/v6ens_dev_predictions.txt` | `results/ranker/v6ens_test_predictions.txt` | 0.724815 |
| v6 | v6 ranker, causal popularity | `results/ranker/v6_dev_predictions_causalpop.txt` | `results/ranker/v6_test_predictions.txt` | 0.723369 |
| ck45500 | 9B checkpoint 45500, raw LLM scores | `results/ranker/lg9b_ckpt45500_dev_scores.tsv` | `results/ranker/lg9b_ck45500_test_scores_all.tsv` | 0.718056 |
| ep082 | 9B epoch 082, the `lg9b_llmonly` submission | `results/ranker/lg9b_ep082_dev_scores.tsv` | `results/ranker/lg9b_ep082_test_scores_all.tsv` | 0.713363 |
| l13 | early LightGBM ranker (`ranker_l13_large.txt`) | `results/ranker/dev_ranker_preds.npy` | `results/ranker/test_ranker_predictions.txt` | 0.707374 |

Left out on purpose:

- `MINDlarge_test_ck45500_llmonly.zip` is a constant ranking (`[17,17,17,…]`). The fusion uses the score file, whose dev AUC is 0.718056.
- `lg9b_v3` and `lg9b_v4` were fit on all 376,471 dev impressions, so they have no held-out dev ranking to fuse or to score.
- `mh0v6` dev AUC is 0.726416, higher than any single member above, but its test rescore was not finished, so it is not in these zips.

Dev and test use the same representation. ck45500 and ep082 contribute raw scores. v6ens, v6, and l13 contribute ranks on both splits. The L13 dev `.npy` is used only to build those ranks; its raw values are not mixed into the z-score fusion, because the L13 test file is a ranking.

### 0.2 Pools

| Pool | Members | Test zip |
|---|---|---|
| core | v6ens, v6, ck45500 | yes |
| plus | core + ep082 | no; diagnostic only |
| wide | plus + l13 | yes |

`plus` exists to separate the two weaker members. Adding ep082 on top of core reaches at most 0.726465. Adding l13 is what moves the wide pool to 0.728388.

### 0.3 Methods

For one impression, member `m` has rank `r[m, i]` for candidate `i` (1 = best). The fused score is higher-is-better. The written rank is that score sorted descending, ties broken by original position.

**Borda (equal mean rank).** `s[i] = -sum_m r[m, i]`. A one-step rank change counts the same at the top of the list and at the bottom.

**Reciprocal-rank fusion, k = 60.** `s[i] = sum_m 1/(60 + r[m, i])`. Unweighted. A move from rank 1 to rank 2 changes the score more than a move from rank 30 to rank 31, so agreement at the top dominates the tail.

**Weighted Borda.** `s[i] = -sum_m w[m] * r[m, i]`. The weight is `max(AUC[m] - 0.70, 0.001)`, then renormalized to sum to 1. Measured weights:

| Pool | v6ens | v6 | ck45500 | ep082 | l13 |
|---|---|---|---|---|---|
| core | 0.375 | 0.353 | 0.273 | — | — |
| plus | 0.312 | 0.294 | 0.227 | 0.168 | — |
| wide | 0.285 | 0.269 | 0.208 | 0.154 | 0.085 |

Unnormalized weights are `max(AUC - 0.70, 0.001)`: v6ens 0.024815, v6 0.023369, ck45500 0.018056, ep082 0.013363, l13 0.0074. The L13 input is the prior 0.7074, not the recomputed 0.707374. That difference does not change the three-decimal weights above. Pool sums before renormalizing: core 0.06624, plus 0.079603, wide 0.087003.

**Within-impression z-score.** For ck45500 and ep082, z-score the raw scores inside the impression. For v6ens, v6, and l13, z-score `-r[m, i]`. `s[i]` is the sum of those z-scores, with equal member weight. Mean and variance are the population values (`variance = sum((x - mean)^2) / n`). If that variance is below `1e-12`, the member contributes zeros for that impression. This is the one method that keeps an LLM margin: a candidate far ahead of the others stays far ahead, instead of being collapsed to a one-step rank gap. A rank-only member still contributes a standardized rank, because that is all it has on both splits. L13's dev `.npy` holds raw scores, but those raw scores are not z-scored. Test L13 is a ranking, and the two splits have to use the same representation.

### 0.4 Dev AUC

| Fusion | Dev AUC |
|---|---|
| wide z-score | **0.728388** |
| wide RRF k=60 | 0.728255 |
| wide Borda | 0.728112 |
| wide weighted Borda | 0.727685 |
| plus weighted Borda | 0.726465 |
| plus z-score | 0.725846 |
| plus RRF k=60 | 0.725818 |
| core z-score | 0.725686 |
| core weighted Borda | 0.725577 |
| plus Borda | 0.725569 |
| core RRF k=60 | 0.725568 |
| core Borda | 0.725466 |

Best single member in the fusion is v6ens at 0.724815. Wide z-score is +0.0036 over that. Core z-score, which does not use ep082 or l13, is only +0.0009.

### 0.5 Test submissions

Text files are `results/ensemble/{pool}_{method}.txt`. Codabench zips, one member named `prediction.txt`, are `results/submissions/MINDlarge_test_ens_{pool}_{method}.zip`.

| Zip | Dev AUC |
|---|---|
| `results/submissions/MINDlarge_test_ens_wide_zscore.zip` | **0.728388** |
| `results/submissions/MINDlarge_test_ens_wide_rrf60.zip` | 0.728255 |
| `results/submissions/MINDlarge_test_ens_wide_borda.zip` | 0.728112 |
| `results/submissions/MINDlarge_test_ens_wide_wborda.zip` | 0.727685 |
| `results/submissions/MINDlarge_test_ens_core_zscore.zip` | 0.725686 |
| `results/submissions/MINDlarge_test_ens_core_wborda.zip` | 0.725577 |
| `results/submissions/MINDlarge_test_ens_core_rrf60.zip` | 0.725568 |
| `results/submissions/MINDlarge_test_ens_core_borda.zip` | 0.725466 |

Line format is the official one: `impression_id [r1,r2,...,rn]`, no spaces inside the brackets. `r_i` is the 1-based rank of the i-th candidate in behaviors order. `results/submissions/MINDlarge_test_ens_wide_zscore.zip` was checked with the official `parse_line` (`split` into two fields, then `json.loads` on the rank array) against `data/MIND_large/test/behaviors.tsv` and against the already-scored `results/ranker/submission_v6_test.txt`. All 2,370,727 lines match impression id and candidate count, and every rank list is a permutation of `1..n`.

Submit `MINDlarge_test_ens_wide_zscore.zip` first. Rebuild with:

```bash
python3 results/ensemble/fuse_candidates.py --write-test
for f in core_borda core_rrf60 core_wborda core_zscore \
         wide_borda wide_rrf60 wide_wborda wide_zscore; do
  bash scripts/build_submission_zip.sh \
    "results/ensemble/${f}.txt" \
    "results/submissions/MINDlarge_test_ens_${f}.zip"
done
```

`--write-test` always rescores dev first, then writes the eight test files. `plus` is scored on dev and is not written. Behaviors paths are `data/MIND_large/dev/behaviors.tsv` and `data/MIND_large/test/behaviors.tsv`. The run that produced these files finished 2026-11-08 09:11 UTC and printed `test impressions 2,370,727`.

### 0.6 How a line is built

Files are streamed in lockstep with behaviors, one impression at a time. The script exits if an impression id disagrees, if a score row has a different length than the candidate list, if a rank row is not a permutation of that list, or if the L13 array does not cover exactly the dev candidate rows. Nothing is joined by a dictionary that could reorder impressions.

Candidate ids: a dev behaviors item is `newsId-label`; a test item is a bare `newsId`. The id is the part before the last `-`. Duplicate ids in one impression are matched left to right against the ranked-id list.

Rank from a score row: sort by score descending, and by original index ascending on a tie, then assign ranks `1..n`. Rank from a news-id row: the row is best-first; walking it assigns rank 1, 2, 3, … back onto the behaviors order.

The leaderboard line is `impression_id [r1,r2,...,rn]` with no spaces after the commas. `r_i` is the rank of candidate `i` in behaviors order. Sort key for that rank is fused score descending, original index ascending. The written ranks are therefore a permutation of `1..n` even when two fused scores tie: the earlier candidate gets the better rank.

Score-file shape: `impression_id<TAB>score1<TAB>score2...`, scores in behaviors order. Rank-file shape: `impression_id newsId_best newsId_second ...`.

Dev AUC is the mean, over impressions, of

```
(sum of positive ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

Ranks in that formula are average ranks of tied fused scores, 1-based, from a sort by score ascending then original index. An impression with no positive or no negative is skipped. This dev split has none of those, so every number uses `n = 376,471`. The same pass recomputed the five single-member AUCs and they matched the table in 0.1 to six decimals, which is the alignment check for the inputs.

One difference from the file the scorer will read: dev AUC is computed on the fused scores, where a tie gets the average rank. The text file breaks that tie by original position, so the official `1/rank` score has no ties. On an impression with no fused-score tie the two AUCs are the same, because both are functions of the pairwise order. The reported dev numbers were not recomputed from the written rank files.

### 0.7 What each original submission zip actually is

The seven zips in `results/submissions/` before this fusion, and what was done with each:

| Zip | Dev AUC of the real model | Used in the fusion |
|---|---|---|
| `MINDlarge_test_v6ens.zip` | 0.724815 | yes, via `v6ens_*_predictions.txt` |
| `MINDlarge_test_v6_stack.zip` | 0.723369 | yes, via the causal-pop files |
| `MINDlarge_test_ck45500_llmonly.zip` | 0.718056 for the score file; the zip itself is all ties | score file only |
| `MINDlarge_test_lg9b_llmonly.zip` | 0.713363 | yes, it is ep082 |
| `MINDlarge_test_ranker_l13.zip` | 0.707374 | yes, via the npy and `test_ranker_predictions.txt` |
| `MINDlarge_test_lg9b_v3.zip` | no held-out number | no |
| `MINDlarge_test_lg9b_v4.zip` | no held-out number | no |

Identities that were checked, not assumed:

- v6 stack. `logs/convert_v6_test.log` converts `results/ranker/v6_test_predictions.txt` into `results/ranker/submission_v6_test.txt`. `logs/apply_v6_dev_causal.log` scores the same model, `results/ranker/ranker_v6_train_popwit.txt` (33 features, within-impression transforms plus popularity transforms), onto `v6_dev_predictions_causalpop.txt`. That file is the v6 member. The other dev file, `v6_dev_predictions.txt`, scores 0.723394 and is not a member.
- v6ens. `logs/apply_v6ens_dev.log` applies that same v6 model to the ensemble feature table. `MINDlarge_test_v6ens.zip` stores ranked news ids, not bracket ranks. The fusion reads `v6ens_test_predictions.txt` and emits bracket ranks.
- llmonly. The zip's first rank vector is `1 [8,15,13,2,11,12,16,14]`, which is the ranking of `results/ranker/lg9b_ep082_test_scores_all.tsv`. ep099 (`results/ranker/lg9b_ep099_dev_scores.tsv`, dev AUC 0.709160) does not match and is not a member.
- ckpt-45500. `results/submissions/MINDlarge_test_ck45500_llmonly.zip` starts `1 [17,17,17,…]`. Ranking `results/ranker/lg9b_ck45500_test_scores_all.tsv` for impression 1 gives `[8,15,11,2,12,14,16,13]`, so the zip is not that score file. A second copy of the same scores, `results/lg9b_ck45500_test_scores_all.tsv`, was not used. The fusion uses `results/ranker/lg9b_ck45500_test_scores_all.tsv`.
- L13. `logs/cmp_rankers_dev.log` applies `results/ranker/ranker_l13_large.txt` to `results/ranker/large_dev_features.tsv` and reports per-impression AUC 0.7074 on 376,471 impressions. `dev_ranker_preds.npy` is `float64` with shape `(14085557,)`, equal to the number of dev candidate rows. Walking it in behaviors order reproduces AUC 0.707374. Test ranks come from `results/ranker/test_ranker_predictions.txt` (`logs/apply_ranker_test.log`). The model is a LightGBM binary booster, 14 anonymous columns, 63 leaves.
- v3 and v4. `logs/train_ranker_v3.log` and `logs/train_ranker_v4.log` each load the full dev feature table, 14,085,557 rows, 376,471 impressions, missing llm 0, then save `ranker_v3_lg9b_joint_wit.txt` and `ranker_v4_lg9b_joint_wit_h24.txt`. Their pipelines (`logs/pipeline_v3.log`, `logs/pipeline_v4.log`) apply those models to test only. An AUC on the training impressions would be a fit statistic, so they were not given one and were not fused.

### 0.8 Files scored on dev and then not fused

Same metric, same 376,471 impressions, from the pass on 2026-11-08. None of these has a finished test prediction that was included.

| File | Dev AUC | Why it is not a member |
|---|---|---|
| `results/ranker/mh0v6_dev_predictions.txt` | 0.726416 | test rescore had not finished (`logs/pipeline_mh0_v6.log` still waiting on a test chunk) |
| `results/ranker/v6ens5c_dev_predictions.txt` | 0.725740 | dev file only; no test prediction in `results/ranker/` |
| `results/ranker/v6ens4c_dev_predictions.txt` | 0.724739 | same |
| `results/ranker/v6ens5b_dev_predictions.txt` | 0.724275 | same |
| `results/ranker/v6ens2_dev_predictions.txt` | 0.724051 | same |
| `results/ranker/v6ens4_dev_predictions.txt` | 0.723993 | same |
| `results/ranker/v6ens5_dev_predictions.txt` | 0.723782 | same |
| `results/ranker/v6_dev_predictions.txt` | 0.723394 | not the file the v6 stack zip was built from |
| `results/ranker/lg9b_ep099_dev_scores.tsv` | 0.709160 | not the llmonly zip |

v6ens5c is a better dev file than the v6ens member (0.724815), and mh0v6 is better than both. They are absent because the test side was missing, not because the dev number was worse.

### 0.9 Format check

Official parser, from `https://github.com/msnews/MIND/blob/master/evaluate.py`:

```python
def parse_line(l):
    impid, ranks = l.strip('\n').split()
    ranks = json.loads(ranks)
    return impid, ranks
```

`split()` with no arguments requires exactly two whitespace-separated fields. A space after a comma makes the unpack fail, which the scorer reports as `Invalid Input Format`. `json.loads` then requires a JSON array. The scorer turns each rank into `1/rank` and requires that value in `(0, 1]`, so every rank must be `>= 1`. `roc_auc_score` then requires the array length to equal the truth-file label length. Impression ids are compared as strings, in file order. The scorer does not require the ranks to be a permutation; this fusion's files are permutations anyway.

`scripts/build_submission_zip.sh` copies the text to a temp file named `prediction.txt`, runs `zip -q -X`, and asserts the archive has exactly one member of that name and the same byte size as the text. All eight archives passed that assert. Each inner file is 291,329,312 bytes. Deflate (`compress_type = 8`).

The full official parse was run on `results/submissions/MINDlarge_test_ens_wide_zscore.zip`, not on the loose text. Every one of the 2,370,727 lines:

- has a single `\n` and no `\r`
- splits into exactly two fields and `json.loads` succeeds
- has the same impression id as `data/MIND_large/test/behaviors.tsv` and as `results/ranker/submission_v6_test.txt` (the v6 file that already scored on Codabench)
- has a rank count equal to that impression's candidate count in both of those files
- is a permutation of `1..n` of Python `int`s

The other seven zips were not put through that full parse. They come from the same writer, the build script checked member name and byte size for each, and a line count of the eight text files before zipping was 2,370,727 each. First lines, which are impression 1 and 16 candidates:

| File | First line |
|---|---|
| `core_borda.txt`, `core_rrf60.txt` | `1 [8,15,11,3,12,14,16,13,2,1,9,5,7,6,10,4]` |
| `core_wborda.txt`, `core_zscore.txt` | `1 [8,15,12,3,11,14,16,13,2,1,9,5,7,6,10,4]` |
| `wide_borda.txt` | `1 [8,15,12,3,11,14,16,13,2,1,9,5,6,7,10,4]` |
| `wide_rrf60.txt`, `wide_wborda.txt` | `1 [8,15,12,3,11,14,16,13,2,1,9,5,7,6,10,4]` |
| `wide_zscore.txt` | `1 [8,15,11,3,13,14,16,12,2,1,9,5,6,7,10,4]` |

### 0.10 Where the files are

On `sc-vnc12.sambanovasystems.com`, user `chenw`:

- script: `/import/ml-sc-scratch5/chenw/MiniOneRec/results/ensemble/fuse_candidates.py`
- dev table: `results/ensemble/dev_auc.txt`
- test text: `results/ensemble/{core,wide}_{borda,rrf60,wborda,zscore}.txt` (exact size 291,329,312 bytes each; `ls -lh` prints that as 278M)
- zips: `results/submissions/MINDlarge_test_ens_*.zip`

Copy the zip to submit:

```bash
rsync -avP chenw@sc-vnc12.sambanovasystems.com:/import/ml-sc-scratch5/chenw/MiniOneRec/results/submissions/MINDlarge_test_ens_wide_zscore.zip /Users/chenw/Downloads/
```

Codabench upload, when wanted, is `scripts/submit_codabench.sh`. Default phase is 23196 (Official Test), task 29311, competition 13967. These eight zips have not been uploaded.

---

## 1. TL;DR — where things stand right now (2026-10-19)

- Team best remains **0.7144** = plain L13/L17 test ensemble (E3A). SOTA gap = +0.0182.
- A years-long score-space investigation is **CLOSED** (memory `project_minionerec_ensemble_investigation.md`): every existing/diverse model was scored + folded with **honest CV**. Result: **SUBCAT is the ONLY positive new member**.
- **In flight:** SUBCAT **test** score capture (`neat_bridge_n047m8d9zk`, created 2026-10-19 05:26 UTC, Running). When it lands I assemble **L13+L17+SUBCAT (0.4/0.4/0.2)** and submit to Codabench. This is the one live thread — everything else is done.
- Everything below the submission step is **ready and validated**; it only needs the subcat test scores to land.

---

## 2. The one live task (finish me first)

**Wait for SUBCAT test scores** → fold → submit.

| Piece | Location | Status |
|---|---|---|
| SUBCAT test AML job | `neat_bridge_n047m8d9zk` → `ens_subcat_test_scores/scores/` | Running (created 05:26 UTC) |
| Score watcher (background) | `Q:\MiniOneRec\.offline\watch_scores.py` (bg task, polls 5 min) | Running |
| Reusable L13/L17 TEST scores | `Q:\MiniOneRec\.offline\test_L13.txt`, `test_L17.txt` (2,370,727 rows ea) | Downloaded |
| Test behaviors | `Q:\MiniOneRec\.offline\test_behaviors.tsv` (2,370,727 rows) | Downloaded |
| General assembler | `Q:\MiniOneRec\.offline\assemble_sub.py` | Validated (2,370,727 imps) |
| Submission endpooint | `POST https://www.codabench.org/api/submission/submit_against_phase/` (Bearer `api_key` in `~/.codeguru/.../codabench_key`) | Ready |

**When `scores_subcat_test.txt` lands** (watcher prints `LANDED subcattest: ..._subcat_chat_scores.txt`):

```bash
cd /q/MiniOneRec/.offline
# fold L13 + L17 + SUBCAT (dev-sweep weights: w_subcat 0.2, w_base 0.4 each)
python assemble_sub.py \
  --behaviors test_behaviors.tsv \
  --scores test_L13.txt:L13 test_L17.txt:L17 scores_subcat_test.txt:SUBCAT \
  --weights 0.4 0.4 0.2 \
  --output /tmp/submission_l13l17subcat.txt
```

Then ZIP `prediction.txt` (rename) and POST to Codabench. Expected output: 2,370,727 lines `imp_id [r1,r2,...]` with bare numeric ids, candidate order per behaviors col5.

### Why SUBCAT (the data behind the bet)

Honest-CV fold on dev (50k/50k train-val, `fold_div_cv.py`), all 2,370,727-aligned:
- SUBCAT standalone **0.6961**, corr(L13+L17) 0.874 → **+0.0010 val / +0.0012 full** (first + member ever)
- 8B (properly evaled at healthy ckpt-18432) = **0.5479**, corr **0.365** → dilutes (diverse-but-no-signal)
- HIST50 0.6856 → wash on base, dilutes on subcat. 4B r2 0.6801 → wash. 8B test capture **cancelled** (dilutes on dev).

Caveat (record it, don't fight it): dev→test transfer is NOT guaranteed — DUAL was **+0.0033 dev / −0.0023 test**. Subcat's edge is smaller (+0.0010), but it is a *structurally different* diversity (explicit news-category feature) than DUAL (from-scratch encoder), so it's the best-shot left. If it lands >0.7144, team best updates; if not, this avenue is exhausted.

---

## 3. The offline evaluation toolkit (all in `Q:\MiniOneRec\.offline\`)

These run **locally, no GPU**, on score files from the datastore. This is the core reusable asset.

**Score-folding (honest CV — the DECISION metric):**
```bash
python .offline/fold_div_cv.py <new_scores.txt> <TAG> scores_L13.txt:L13 scores_L17.txt:L17
# prints standalone AUC, spearman corr vs L13+L17 (diversity), and a 50k/50k train-val
# sweep of the new member's weight on VAL only. Delta on val is the number to trust;
# full-set is reference only (dev-picked combos don't transfer to test).
```
Core score files present: `scores_L13.txt`, `scores_L17.txt`, `scores_4babs.txt`, `scores_subcat.txt`, `scores_hist50abs.txt`, `scores_8b_dev.txt`, `scores_dual.txt`, `scores_a73t.txt`, `scores_ep7chat.txt`, `scores_base_abs.txt`.

**Test-submission assembler (validated):** `assemble_sub.py` (see §2) — parameterized, any member set.

**Rank-fusion / alternates:** `eval_div_members.py`, `rank_fusion_clean.py`, `eval_rank_ensemble.py`, `cv_select.py`, `sweep_ensemble_weights.py`, `eval_equal_combos.py`, `eval_augmented.py`.

**Datastore tooling (WebHDFS REST, no SDK needed):** `watch_scores.py` (persistent 5-min poll+download), `grab_ens_scores.py`, `check_ckpt_health*.py` (shard-length inventory), `evaluate_mind_pointwise.py` warmup references.

**Dev behaviors** for offline scoring: `dev_behaviors.tsv` (376,471 rows).

---

## 4. How score captures work (AML launchers & gotchas)

**Launch an ensemble-eval score capture for any checkpoint:**
```bash
cd /q/MiniOneRec
python pipeline/run_ensemble_eval_pipeline.py \
  --experiment-name mind_ensemble_eval \
  --display-name "<my label>" \
  --wave1-models "shares/users/wuc/output_dir/sft_..._chat/final_checkpoint" \
  --output-subdir "ens_<tag>_scores" --split dev|test --max-history 30
```
Scores land at `output_dir/ens_<tag>_scores/scores/<model>_scores.txt` (one line per impression: `imp_id<TAB>s1<TAB>s2...`, row-aligned with behaviors).

**CRITICAL gotchas — every one has already burned a submission cycle:**

1. **`--wave1-models` MUST include the `shares/users/wuc/output_dir/` prefix.** The component's `pipeline_executor.py` does `${mount}/{p.strip()}` — it adds the mount dir but NOT `output_dir`. Missing prefix → HF treats abs path as repo id → **`OSError: Repo id must be in the form ...`**. Diagnostic tell: compare failing model path `INPUT_msndni/sft_...` vs working data path `INPUT_msndni/shares/users/wuc/data/...`.
2. **Role of `_AZUREML_SINGULARITY_JOB_UAI`** — the component sets this env var to bind the managed identity so the Singularity job can auth to the RW_MOUNT datastore (avoids `NoIdentityOnCompute`). Don't strip it.
3. **Cold fuse-mount warmup** (`src/evaluate_mind_pointwise.py`): waits up to 10 min for a marker file. Accepts **either** `model.safetensors` (single-shard) **or** `model.safetensors.index.json` (sharded, e.g. 8B=4 shards). This two-marker fix was pushed as commit `eda5dae`. **The eval component re-clones from git `dev`, so source fixes must be committed+pushed to take effect — a local edit is invisible to AML.**
4. **Verify checkpoint health before burning GPU** (`check_ckpt_health*.py`): the 8B `final_checkpoint` was **truncated** (2/4 shards, 6.5GB vs 16.38GB) → its eval was ~chance (0.497) as a broken-model artifact, NOT a real signal. Healthy 1.7B finals = single-shard 4.06GB; 4B-large has **no** `final_checkpoint` (only ckpts 31232+, all complete 8.82GB).

**Status checking:** `python pipeline/...` + `ml.jobs.get(jid)` via SDK; `ml.jobs.download(jid, download_path=...)` (NO `logs_only` kwarg — throws TypeError) → log under `artifacts/user_logs/std_log.txt`. Launch on `recall` (8× A100, `Singularity.ND96amrs_A100_v4`) — queue contention is normal; parent Running + child Queued = waiting for GPU, not broken.

---

## 5. Codabench submission recipe

1. Assemble `prediction.txt` (format above).
2. ZIP it (single file inside).
3. `POST https://www.codabench.org/api/submission/submit_against_phase/` with the participant's `api_key` as Bearer and `files=@<zip>`.
   - Participant `api_key` is stored locally (Codabench) — it's the same key used for previous submissions (`submission_l13l17dual.zip`, id 992349 → 0.7121).
4. Leaderboard API shows only the owner's **BEST** score per task — a new queued sub won't appear until scored. Poll the submission's `status` until `Finished`.
5. Format must be byte-identical to the accepted E3A `leaderboard_submission.txt` (0.7144): bare numeric ids, `\d+ [r1,r2,...]`, 2,370,727 lines. The `assemble_sub.py` validation already confirmed it produces 2,370,727.

Known result for reference: L13/L17/DUAL (0.35/0.25/0.40) = **0.7121** test (id 992349) — DUAL dropped it. Team best 0.7144 is plain L13/L17.

---

## 6. Verified failures (don't redo)

| Attempt | Result | Why it failed / won't work |
|---|---|---|
| L13/L17/DUAL submit | 0.7121 (−0.0023) | dual %encoder doesn't transfer to test |
| Re-ensembling existing 1.7B members | saturates ~0.709 dev | weight tuning = wash (+0.0004), 3rd/4th member always dilutes |
| 4B, 8B, 2B-abs dev preds rank-fused | all dilute | 8B was truncated artifact; others genuinely weak |
| 8B as member | dilutes (−0.0007 val) | 0.5479 standalone, corr 0.365 = diversity w/o signal |
| HIST50 as member | wash/dilutes | 0.6856, corr 0.858 |
| A7.3 pairwise (title-only) | 0.6422 large (no transfer) | +1.07 small did NOT scale to large |

**Biggest remaining arch lever (never built):** a true NRMS/InfoNCE **dual-encoder** *trainer* scaffold in-repo (DUAL was a one-off trained member, not a reusable component). That's the only path toward a genuinely new, maximally-decorrelated member rather than another 1.7B SFT variant.

---

## 7. Getting the dataset (MIND) — how to obtain it

Two ways: **already staged on the AML datastore** (primary, what all our runs use), or **download from source**.

### 7a. On the AML datastore (primary — ready to use)

MIND_large lives on ADLS Gen1 `msn-dni-c09` at `shares/users/wuc/data/MIND_large/` with `{dev,test,train}/`:

| Split | behaviors.tsv | news.tsv |
|---|---|---|
| train | 1.37 GB | 0.08 GB |
| dev | 0.23 GB | 0.08 GB |
| test | 1.46 GB | 0.10 GB |

Each split also has `entity_embedding.vec`, `relation_embedding.vec`; train/dev additionally have `cf_scores.tsv` and RL pointwise parquets (`rl_pointwise_neg{2p0,3p0}_*.parquet`). Pipelines reference these via the `data_root` arg (`shares/users/wuc/data/MIND_large`) — no download needed inside AML.

**MIND_small** is at `shares/users/wuc/data/MIND_small/` (use for fast develop-test loops), and a copy at `shares/users/wuc/data/MIND/` (raw MS-MIND, not the competition repack).

### 7ab. Already-downloaded on this machine (offline ready)

For offline analysis you don't need to re-download anything — these are already local:
- `Q:\MiniOneRec\.offline\dev_behaviors.tsv` — dev behaviors, 376,471 rows
- `Q:\MiniOneRec\.offline\test_behaviors.tsv` — test behaviors, 2,370,727 rows
- `Q:\MiniOneRec\.offline\test_L13.txt`, `test_L17.txt` — L13/L17 **test** scores (2,370,727 rows each)
- `Q:\MiniOneRec\.offline\scores_*.txt` — all the per-model dev score vectors (L13, L17, subcat, hist50abs, 4babs, 8b_dev, dual, a73t, ep7chat, base_abs) that drive the offline ensemble analysis.

### 7b. Download-to-local (WebHDFS REST — no SDK, no AML needed)

Reusable recipe (see `.offline/` watcher/download scripts for the exact pattern):

```python
from pipeline.azure_auth import get_credential
import requests, urllib.parse
cred = get_credential()                      # InteractiveAzureCredential chain
tok = cred.get_token("https://datalake.azure.net/.default").token  # datalake SCOPE
base = "https://msn-dni-c09.azuredatalakestore.net/webhdfs/v1"
# list:
q = urllib.parse.urlencode({"op":"LISTSTATUS","listtype":"raw"})
r = requests.get(f"{base}/shares/users/wuc/data/MIND_large/test?{q}",
                 headers={"Authorization":f"Bearer {tok}"}, timeout=30)
# download (stream to avoid OOM on the 1.4GB behaviors):
q = urllib.parse.urlencode({"op":"OPEN"})
r = requests.get(f"{base}/shares/users/wuc/data/MIND_large/test/behaviors.tsv?{q}",
                 headers={"Authorization":f"Bearer {tok}"}, timeout=1800, stream=True)
open("behaviors.tsv","wb").write(b"".join(c for c in r.iter_content(1<<20)))
```

Auth gotcha: datalake operations need the **`https://datalake.azure.net/.default`** token scope; the **management** scope (`https://management.azure.com/.default`) 401s on WebHDFS (and vice-versa for job REST).

### 7c. From the public source (external users / no Azure access)

The competition is **Codabench MIND News Recommendation #13967** ([codabench.org/competitions/13967](https://www.codabench.org/competitions/13967/)), whose official files come from the public **MIND** dataset:
- Public download: **`https://mind2016small.blob.core.windows.net/release/`** — `MINDlarge_train.zip`, `MINDlarge_dev.zip`, `MINDlarge_test.zip` (and `MINDsmall_*.zip`). These are the same splits MS uses for the competition.
- Layout per zip: `{train,dev,test}/behaviors.tsv` + `news.tsv` (+ `entity_embedding.vec` / `relation_embedding.vec` where provided).
- `behaviors.tsv` format (tab-separated, 5 cols): `impression_id<TAB>user_id<TAB>time<TAB>history(newsid list)<TAB>impressions(newsid-label[,newsid-label...])` — label `0`=negative, `1`=positive click. Test split candidates have NO label (format `newsid` only).
- `news.tsv` (6 cols): `newsid<TAB>category<TAB>subcategory<TAB>title<TAB>abstract<TAB>url<TAB>title_entities<TAB>abstract_entities`.

If a successor has no Azure access, they can reproduce all offline analysis with these public files + the score/assembler scripts in `.offline/` (dev behaviors = `MINDlarge_dev.zip`).

---

## 8. Quick reference (paths + IDs)

- **Datastore** (ADLS Gen1 `msn-dni-c09`): WebHDFS REST `https://msn-dni-c09.azuredatalakestore.net/webhdfs/v1`. Auth: `pipeline.azure_auth.get_credential()`; datalake ops need `https://datalake.azure.net/.default` token; management REST needs `https://management.azure.com/.default` (mismatch → 401).
- **Workspace:** sub `b6dc87f3-c479-49c8-8cb5-7896da3ff895`, RG `AMLStudio`, workspace `NewsFeedL2_AML`.
- **In-flight IDs:** SUBCAT test = `neat_bridge_n047m8d9zk`. 8B test capture cancelled. 8B dev (done) = `lime_cumin_zsw8tfnby8` → `scores_8b_dev.txt`.
- **Model checkpoints on datastore** (`output_dir/sft_mind_pointwise_large_*`):
  - Core: `Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat` (L13), `Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30` (L17)
  - SUBCAT: `..._hist30_abstract_subcat_chat/final_checkpoint` (4.06GB healthy)
  - 8B: `..._Qwen3-8B_bs256_ep5_neg2.0_hist30_chat/checkpoint-18432` (healthy ckpt; final truncated)
- **Memory:** `~/.codeguru/projects/Q--CodeGuru/memory/project_minionerec_ensemble_investigation.md` + `..._sota.md` hold the full reasoning trail; `docs/MIND_SURVEY.md` has the broader survey.
