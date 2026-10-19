# MIND ensemble + SOTA-push handoff runbook

**Author:** CodeGuru (Chen's assistant) · **Date:** 2026-10-19
**Repo:** `Q:\MiniOneRec` (GitHub `overwindows/MiniOneRec`, branch `dev`)
**Goal context:** Beat team-best **0.7144 AUC** on Codabench MIND comp #13967; SOTA = `zetik` **0.7326**, top-5 = 0.7226.

This runbook hands off the *current in-flight work* and the *offline evaluation toolkit* so a successor can pick up where I left off without re-deriving anything.

---

## 1. TL;DR — where things stand right now

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
