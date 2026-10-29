# MIND Experiment Plan: Path to SOTA

> Generated: 2026-02-25
> Updated: 2026-10-27 (user clarification of the research goal)
> **Project acceptance target: AUC >= 0.80. This goal has not been achieved.**
> Autonomous scope decision (2026-10-28 UTC): competition MIND-large test is
> the ultimate acceptance benchmark; official dev is the development gate.
> MIND-small screening is separate and cannot establish the competition goal.
> Training AUC and selected partial-dev windows do not establish this target.
> The completed full-training JEV campaign is an experiment milestone, not
> completion of the overall research goal. Its matched partial MIND-small dev
> AUC was 0.67203545 versus retained 0.67598829; the full candidate was rejected.
>
> Recorded Best (submitted): **71.44% AUC on test** / 71.52% dev (E3.A: L1.3 + L1.7 ensemble)
> **Rank: #14 / 101 on Codabench (MIND News Recommendation, competition 13967)**
>
> **Recorded leaderboard snapshot (2026-09-27, private/test):**
> - #1 in that snapshot: **zetik 0.7326** AUC (2026-06-23) — historical reference, not the 0.80 acceptance target
> - #2: newsbang 0.7316 · #3: i4never 0.7258 · #4: qinzixuan 0.7249 · #5: jianchunyang 0.7226
> - **overwindows: 0.7144 AUC** (MRR 0.3621, nDCG@5 0.3975, nDCG@10 0.4539), 2026-04-21
>
> **Historical gap to that snapshot's #1: +0.0182 AUC** · to top-5 (0.7226): +0.0082 · to top-3: +0.0114
> Those historical gaps do not define project success. The user requires
> AUC >= 0.80; keep dataset/split/candidate coverage explicit and do not infer
> a valid target gap by comparing results from different benchmarks.

## JEV-UNIFORM-1: corrected full-support training campaign

Registered2026-10-28 UTC before its first allocation or model fit. This is a
new data-support hypothesis, not a retry or budget reset of JEV-JOINT-1 or
JEV-LOWRANK-1. Neither prior experiment is restarted.

- Use the opt-in uniform sampler,seed42,1:1 negative budget,history30 and
  title+abstract, covering all156965 official MIND-small training impressions
  and all236344 positives. Corrected sampled population:472626 rows,
  236282 negatives,zero single-class impressions. Canonical row SHA256:
  `fd9f50c2668cd50ae03f918636010f84e43563c73c711997ef721ea30a9a3a84`.
- Reuse266352 exact identity/label-matched frozen cache rows; extract only
  the206274 missing negatives in four serial shards of65536/65536/65536/9666.
  Keep the frozen APUS OpenJev4B,legacy prompt,max_prompt_tokens4095 and
  2560-dimensional normalized decision representation unchanged.
- Before each extraction, require bitwise feature AND margin equality on
  the first16 rows of the original doubled cache, reconstructed in their
  original contiguous16-row batch. Upload only this checksum-pinned private
  parity package as an AML downloaded-folder input; never commit features.
  A parity failure stops the campaign; do not lower comparison precision.
- Feature budget: at most four new parents,one Standard Ranking A100 per
  parent,21600-second worker cap AND six hours from parent creation including
  queue/setup. Serial allocation,no blind retry or infrastructure replacement.
  Feature extraction/download/assembly aggregate ceiling26 hours from the
  controller's first start; unresolved submission intent stops rather than
  creating a duplicate. Whole campaign including CPU and conditional dev
  ceiling34 hours from that start. Explicit cancellation at deadline.
- Download immutable feature shards with bounded32MB ranges and SHA256
  chunk ledgers; verify whole files,all row identities/order/labels,finite
  float32 features/margins and exactly-once complete target coverage before
  exposing the assembled matrix. Keep large files on the Q data drive.
- CPU protocol,registered before fitting: at most two hours/four CPU threads
  for selection and final refit,within the campaign aggregate ceiling.
  Temporary validation is the last official training day,excluding both
  original retained-head fitting and selection cohorts. Exclude ALL
  validation impressions from counts; fitting-row counts additionally omit
  the row's entire impression fold. Never use dev for model selection.
- Controls: fixed retained rich-text head, and that same fixed head with a
  freshly fitted safe four-column popularity correction,L2=.01,pairwise.
  Three corrected-data candidates use the full rich representation plus
  safe popularity:L2=.01 row-weighted binary,impression-balanced binary,
  and half balanced-binary/half impression-balanced pairwise.
  All use streamed float64 L-BFGS,history100,at most250 iterations,unchanged
  strict convergence checks; nonconvergence is a failure,not a fitted model.
  Rich candidate initialization is the exact retained linear predictor
  transformed into fitting-only normalization,with zero new-popularity
  weights. Conditional final refit starts from the selected temporary head,
  transformed into all-training normalization. This explicitly new
  budget/initialization does not authorize
  another iteration of the stopped legacy-data joint experiment.
- Highest internal impression AUC selects among the three candidates,
  listed order breaks ties. Require >=.005 AUC over BOTH controls and no
  MRR/nDCG@5/nDCG@10 regression against either. Gate-closed means no final
  refit and no dev GPU allocation. Internal scores remain sampled-training
  selection metrics,not official-dev quality claims.
- Only if the gate opens,refit the selected fixed recipe using ALL472626
  rows/all156965 impressions,zero final holdout exclusions; final popularity
  counts use all5843444 raw official-training exposures. Freeze/checksum
  the final head,population and evaluator before reading new dev scores.
- Conditional fresh official MIND-small dev window: zero-based41000-45999
  (impressions41001-46000),5000 impressions,every candidate in source order.
  At most one additional Standard Ranking A100 parent,same six-hour bounds,
  no retries. Save ALL frozen dev features for CPU model/control replay.
  Score the frozen uniform candidate and retained hybrid on identical
  candidates; independently reconstruct metrics and paired bootstrap.
  Never tune or reselect on this window.
- Keep the prior matched official-dev retained model as reference until a
  valid new comparison exists. No train-only result,partial MIND-small dev
  result,cache completion or sampler correction establishes the overall
  competition MIND-large/test AUC>=.80 acceptance goal.

Execution checkpoint2026-10-28 UTC: source
`15c1c56bb519acd998d737a6b57697f36c84f1fc` is frozen in a detached checkout.
First parent `nice_diamond_b4qt3rl0df` was created15:18:43.717488UTC;
its original six-hour deadline is21:18:43.717488UTC. Actual worker
`cb362a5f-7c92-40ef-9f23-7374958d07c4` was Running in environment setup at
15:27UTC,with source checkout and the server's one Standard Ranking A100/
21600-second allocation verified. This is not verified feature completion.
The existing guarded controller owns the four serial stages; do not relaunch.
Training/evaluation implementation is separately frozen at
`2f8b7da8328c4b4dfb6157f16209b9648fe043d6`. Default historical solver
limits remain100; only the new uniform-data caller requests250 and an
algebraically prediction-preserving warm start. No corrected-data fit has
occurred. Independent actual-corpus comparison also checked all51282 training
articles' text/category/subcategory fields against the original dataset loader.

The shared research page/feed was successfully deployed15:32:57UTC with the
qualified submission snapshot. Receipt
`jev-uniform-01-start-publication-verification.json` and companion
`jev-uniform-01-start-publication-content-verification.json` preserve all
twenty prior official-dev table cells and other shared assets/auth unchanged.
The generic publisher receipt's parent/deadline refer to the historical
content campaign; its companion explicitly records the active uniform parent.
No browser-rendering verification or new quality score is claimed.

Full corrected cache completed2026-10-29 00:37:50UTC. All four actual
workers completed within their original bounds: `nice_diamond_b4qt3rl0df`,
`tender_frog_s3n13z5cs2`, `bold_animal_qc9376sybl`, `wheat_rain_mvrkst8p69`.
Complete matrix SHA256
`9d4384d27392334945be007c4e2aa400528129d7b69e54e358e026876fe1cc72`;
frozen-margin array SHA256
`760adae3aa4694406f20ec369e11fdbede7fca37e1e2ad6ddaa175da8db787d1`.
Independent proof `jev-uniform-01-complete-cache-independent-verification.json`
reconstructed every row's origin without using the assembly reuse map:
all266352 reused feature/margin rows and206274 new rows match bit-for-bit.
All472626 identities/labels occur exactly once,all values are finite and
all156965 impressions have both sampled classes. This is data-integrity
evidence,not a benchmark improvement.

The frozen CPU protocol started2026-10-29 00:38:34UTC from source2f8b7da,
four threads,two-hour maximum including preflight. Conservative numerical
cutoff02:37:34UTC; the original whole-campaign cutoff remains unchanged.
Output `Q:\CodeGuru\scratch\jev-uniform-01-training`,attached shell
`jev-uniform-01-training`. No dev allocation or new official-dev score.

Complete-cache/fitting snapshot published2026-10-29 00:46:15UTC.
Receipts `jev-uniform-01-cache-publication-verification.json` and
`jev-uniform-01-cache-publication-content-verification.json` bind the
independent all-row origin proof and preserve all twenty prior official-dev
table cells,other shared assets and auth. No browser-rendering check.
Internal corrected-data validation contains66350 sampled rows across the
same23988 cohort-disjoint impressions;406276 rows fit the temporary heads.
These are not the legacy66042/403222 sampled-row counts and are not dev scores.

CPU selection and all-data refit completed2026-10-29 02:29:46UTC,within the
original two-hour cap. All three candidates genuinely converged:103/109/103
iterations. The row-weighted binary recipe passed the preregistered gate:
internal sampled-training validation AUC0.7168085555 versus stronger safe
control0.7051538661,gain0.0116546894,with no secondary-metric regression.
These are NOT official-dev benchmark scores. Its final refit converged in89
iterations using all472626 rows/all156965 impressions,zero final holdout
exclusions and all5843444 raw training exposures for final population counts.
Independent verification completed02:32:38UTC:all20 metric cells/all5 arms'
66350 predictions,raw population,cohort exclusions,fitting normalizations
and every fitting objective including the final refit were replayed.
Proof `jev-uniform-01-training-independent-verification.json`;
training result SHA256
`04fb19a7a490cf2aa21c4aca1b3fb904f62783a83bea3228728c6b794408bc66`;
frozen final head SHA256
`b03bd7b568ae1c59c608d1fca094d6eff17fde76980881526b831eb36427280b`.
The conditional single-dev controller was submitted separately;
it never overwrites the completed feature-campaign receipt. The reserved
official-dev41001-46000 window contains exactly5000 impressions and189068
candidates,all retained in source order. No official-dev score exists yet;
the original whole-campaign cutoff2026-10-30 01:17:07.923179UTC is unchanged.
Actual dev parent `plum_arm_rs8wvtxmmy` was created02:37:38.356349UTC;
deadline08:37:38.356349UTC including queue/setup. Its server allocation,
feature-source revision,private parity input and frozen gate/head tags were
verified. Actual worker `55097d46-41d9-4acc-9797-a5d7be756c6a` was Queued
at02:37:56UTC; parent Running does not establish GPU execution.
Existing attached shell `jev-uniform-01-dev` owns monitoring/cancellation,
full archive download and frozen CPU evaluation. Do not relaunch.
Submission/final-refit snapshot published2026-10-29 02:40:18UTC. Receipts
`jev-uniform-01-dev-submission-publication-verification.json` and
`jev-uniform-01-dev-submission-publication-content-verification.json`
preserve all20 prior official-dev table cells and other shared assets/auth.
The generic receipt's parent is historical;the companion binds the actual
uniform dev parent and frozen final head. No browser-rendering check.
Independent dev replay implementation `verify_jev_uniform_dev.py` is prepared
for post-completion use;metric tie/multiple-positive fixtures and both frozen
populations' feature mathematics on every189068 candidate passed. No dev
prediction replay or quality result is claimed before actual completion.

**Final corrected-uniform decision2026-10-29 07:56UTC: do not promote;
retain the previous hybrid.** The one registered dev parent/actual worker
both Completed within the original bound. Every189068x2560 dev feature and
margin was saved and whole-file verified. Frozen evaluator finished07:55:22UTC;
independent full4-arm prediction replay/all16 metrics/paired2048-resample
intervals completed07:56:02UTC,with exact official candidate IDs/order/labels.

| Model on identical official MIND-small dev41001-46000 | AUC | MRR | nDCG5 | nDCG10 |
|---|---:|---:|---:|---:|
| Corrected uniform binary,not promoted |0.67541344|0.32684045|0.36128404|0.42226587|
| Retained hybrid |0.67653934|0.32564648|0.35958556|0.42124512|
| Retained text |0.66675783|0.31765986|0.34902490|0.41208096|
| Frozen JEV |0.63083724|0.30081776|0.32475281|0.38632547|

Primary uniform-minus-retained AUC delta-0.0011259018,95%CI
[-0.0070570115,+0.0046361550]. All four metric-difference intervals include
zero;this is no demonstrated gain,not a statistically established loss.
The internal train-only gain did not establish official-dev improvement.
Sampler full-support correctness remains verified,but its causal quality
benefit is not demonstrated by this combined training/population change.
Do not tune or reselect on this already inspected window and do not retry
this closed allocation. All caches/failed recipes are retained for audit.
Proof `jev-uniform-01-dev-independent-verification.json`;
result SHA256 `5c7e272218bbbc22befb6b31320e204180dac50635a1dc0ae413590e60c41bab`;
full-dev archive SHA256
`b74de94c657f30593023c4948832cf37db52c3cebbb237dc61da12c6101fdb9f`.
This is partial MIND-small dev,not full dev,MIND-large or competition-test
success. Overall AUC>=0.80 remains unmet;the research task is not complete.
Verified negative-result page/feed published2026-10-29 08:01:54UTC.
Receipt `jev-uniform-01-result-publication-verification.json` binds the
actual completed parent/worker and independent result/score hashes;all16
new metric cells match the independent proof,all20 prior official-dev
cells and other shared assets/auth remain unchanged. No browser check.

Continuation registered2026-10-29 UTC: no retry of the closed small-data
candidate. Read-only competition-readiness audit found all configured
MIND_large train/dev/test news/behaviors files and both recorded E3.A model
configs;config availability is NOT weight/provenance or benchmark verification.
The three exact cached-test export paths named by the existing waiter are
absent. Historical0.7144 remains recorded,not newly reproduced.
Audit `jev-competition-data-readiness.json`;no test scores/labels read,no GPU.
Registered/now-completed step: exclusive full TRAINING-only source intake on Q,
at most30 minutes within the unchanged original aggregate deadline;zero
GPU allocations,zero model fitting,no dev/test contents read. Pin observed
server lengths/modification versions before and after bounded32MB ranges,
retain chunk ledgers,independently rehash whole local files,and enumerate
EVERY training impression/exposure/news identity. Record first-observed
content hashes without pretending official external provenance was verified.
Source intake is preparation,not a new model or quality result. Do not
silently restart/overwrite partial output or weaken source-version checks.
Completed attached intake shell `jev-large-training-intake`;controller
`prepare_jev_large_training.py`;state `jev-mind-large-train-source-state.json`;
output `Q:\CodeGuru\scratch\jev-mind-large-train-source`. It is operational
preparation only;no large-data model has been fitted or evaluated.

### JEV-LARGE-INPUT-1: completed source intake and lossless training contract

Source intake completed2026-10-29 08:20:15UTC within its original30-minute
allocation. Entire configured MIND-large TRAINING source:2232748 impressions,
101527 news,83507374 candidate exposures,3383656 positives and80123718 negatives;
zero unresolved candidate/history articles and zero single-class impressions.
Training source report SHA256
`542b4d659fb4ab84b7bfbdb7e4d15af4650797737aa334520396df95e3ec0d29`.
Whole news SHA256
`8d16f3e4df6ac8d8943c5f29e9155d0daf73047f1e41d37133013fe0e19953e9`;
whole behaviors SHA256
`886505b6e8b753fe6bc2d8988d959c2cb5865d85832829d0ad91fdfb54e17548`.
These bind the configured training source,not independently verified public
canonical archive bytes. No dev/test contents were read and no model was fitted.

Registered before actual full-contract encoding: `src/full_impression_contract.py`
retains EVERY positive and negative in original impression/candidate order;
no negative sampling and no first-positive-only reduction. Flat int32 news
indices,uint8 labels,uint64 impression offsets,source impression identities,
int32 last30 histories with left-zero padding,and int32 training-date ordinals.
Article identities are sorted lexicographically,indexed from1;0 means padding.
Article category/subcategory/title/abstract are retained in the same index order.
The source report and complete raw news/behaviors bytes must match pinned hashes.
An independent full source-origin replay checks all encoded candidates/labels,
histories/dates/identities/article metadata,file inventories and daily totals;
only its separate verified marker permits use of this prepared training input.

Execution: a new immutable source checkout,exclusive Q output
`Q:\CodeGuru\scratch\jev-mind-large-full-contract`,four CPU threads,zero GPU
allocations/model fits/dev-or-test contents. Encoding AND complete replay share
one hard30-minute ceiling within the UNCHANGED aggregate cutoff
2026-10-30 01:17:07.923179UTC,including controller startup. Preserve partial
outputs and explicit failure receipts;never blindly restart or overwrite.
The last training calendar day is marked for TEMPORARY train-only selection;
final fitting still requires all2232748 impressions/all83507374 exposures.
This input-integrity milestone is not an AUC result,a model improvement,a
competition submission,or proof of the0.80 goal. A new model hypothesis,
train-only gate and genuinely held-out official-dev readout remain necessary.

Full contract and entire source-origin replay COMPLETED2026-10-29 08:42:49UTC
from source`6869eb71fa2465694197c0bbfb1bc026e68785a6`,within the registered
30-minute ceiling. Every2232748 impressions/83507374 candidates/3383656
positives/80123718 negatives replayed. Manifest SHA256
`4daa3377b4601aca4ae5b72e45ed2931becb8dd648dd23998f17fb3e1342a5f4`;
verification SHA256
`de60c130dcd7ea853ce8d2b7fbe81b02b83d6b604fbbba99b96bbfa1ec841b25`.
Temporary last training day2019-11-14 contains431517 impressions/17400106
exposures;earlier fitting days contain1801231 impressions/66107268 exposures.
Largest impression has300 candidates. No training/GPU/dev/test use occurred.
Producer source file bytes are Windows CRLF;that exact producer fingerprint
must not be confused with the Linux LF Git blob. Consume the pinned verified
artifact/file hashes across platforms,not an assumption of identical newline bytes.

### JEV-LARGE-SEMANTIC-1: frozen news representations and all-impression learning

Registered before new feature extraction or scientific fitting. New hypothesis:
encode each training ARTICLE once with the unchanged frozen JEV backbone,then
learn user/history-to-candidate interactions over EVERY full training impression.
This tests both scalable personalization and full exposure/positive use instead
of repeating a long frozen cross-encoder for every sampled candidate.

- One distinct Standard Ranking A100 worker21600s;parent/controller deadline
  six hours from actual parent creation,INCLUDING queue/setup. No parallel GPU,
  retry,resubmission,or restart of any closed campaign. The unchanged aggregate
  cutoff2026-10-30 01:17:07.923179UTC remains binding. Article extraction
  additionally has at most two hours,inside that same six-hour allocation.
  After the GPU parent/worker are both completed,read-only artifact retrieval
  has at most one CPU-only hour within the SAME original aggregate cutoff;
  that does not extend or reopen the GPU allocation.
- Verify the COMPLETE input contract/proof/file hashes. Before new extraction,
  reproduce the original contiguous16-row frozen features/margins by float32
  bit patterns;never relax parity. Fingerprint complete local frozen model
  weights/tokenizer/config before and after extraction. Those are first-observed
  model-byte fingerprints,not new external canonical-provenance verification.
- Article states use empty history and category/title/abstract under the existing
  legacy decision contract. Only the article text prefix may be shortened to
  fit a COMPLETE512-token prompt;the task/criteria/answer boundary remain intact.
  Store all101527 article identities and2560-dimensional float32 features;
  padding index0 is exactly zero. Backbone weights never receive gradients.
- Only two fixed recipes,in listed tie-breaking order: shared128-dimensional
  GELU/LayerNorm news projection with masked mean history,or candidate-conditioned
  history attention with learned30-position recency bias. Article feature vectors
  are L2-normalized. Both use a learned article bias,unit-normalized projected
  candidate/user cosine scaled10;attention uses cosine scaled4. Empty histories
  have explicitly zero user contribution,not unmasked padding or NaN.
- Objective: full-impression softmax over EVERY candidate,normalized over ALL
  positives in that impression,then equal impression mean. No negative sampling,
  positive subsampling,oversized-impression dropping,or incomplete final batch.
  Maximum8192 candidate exposures/256 impressions per batch. For each recipe:
  three fixed complete epochs,seed42(initialization),42/43/44(epoch shuffles),
  AdamW learning-rate0.001,weight-decay0.0001,gradient norm cap1. No early stopping
  or convergence claim. Record full impression/exposure/positive coverage each epoch.
- Temporary fitting uses ALL1801231 pre-last-day training impressions;selection
  evaluates ALL431517 last-day TRAINING impressions/17400106 candidates. Controls:
  earlier-day-only smoothed CTR(prior20 exposures),frozen article-only JEV margin,
  and untrained masked mean article-semantic cosine. Target-dependent control
  statistics completely exclude the last day;the learned models do not use
  those popularity statistics as training-row features.
- Choose highest last-training-day AUC among the two fixed endpoints,listed order
  breaking ties. Allocation gate requires AUC gain>=0.005 over the highest-AUC
  safe control AND no MRR/nDCG5/nDCG10 regression against that same control.
  These are TRAINING-only allocation statistics,never official-dev/test claims.
- Only if that gate opens,reinitialize the selected recipe and train the SAME
  three complete epochs on ALL2232748 training impressions/83507374 exposures/
  3383656 positives/80123718 negatives,zero final holdout exclusions. Save a
  frozen final head and full feature/prediction/coverage evidence. Complete
  independent output/metric/source verification is required before any dev gate.
- No official dev/test contents enter this worker. A future full official
  MIND-large-dev evaluation must be registered only after verified frozen
  selection/refit;it must disclose overlap with previously inspected MIND-small
  dev rather than falsely call the whole benchmark untouched. No retuning on
  dev scores;competition-test AUC>=0.80 remains the ultimate unmet criterion.
- The dedicated component uses a private pinned environment and no multi-node
  setup/third-party telemetry login. Dependencies match the frozen runtime's
  Torch2.6.0/Transformers5.2.0/fla0.4.2 stack;exact original16-row parity is still
  mandatory rather than inferred from package versions.

Public benchmark context checked2026-10-29UTC: the official MIND leaderboard
currently lists0.7326 as its highest AUC(zetik,2026-06-23),and0.7144 for
overwindows(2026-04-21). Our0.80 criterion is an explicit stretch target,not
the current listed official score. Primary source:`https://msnews.github.io/`.
That listing does NOT reproduce our historical checkpoint/submission artifacts.

Actual single parent`jev_large_semantic_20261029_01` submitted2026-10-29
09:06:23.720214UTC from immutable source
`083d63aceee48e99468b59637394da7f74cb88cd`. Server-verified one Standard Ranking
A100,worker21600s;parent/controller deadline2026-10-29 15:06:23.720214UTC,
including queue/setup. Worker`896345fd-6a9b-455f-8b9d-6c1137c892d2` Queued
at09:06:48UTC,parent Running. Contract/proof/parity input hashes and source
revision match the registration. Controller`ROOT\run_jev_large_semantic.py`,
attached shell`jev-large-semantic-01`,state
`ROOT\jev-large-semantic-01-campaign.json`;do not duplicate or resubmit.
Outputs`shares/users/wuc/output_dir/jev_readout_recovery_20261028/large_semantic_01`;
bounded complete readback to`Q:\CodeGuru\scratch\jev-large-semantic-01`.
No model/head/gate result exists at submission;no new dev/test evaluation.

Parent and worker FAILED observed2026-10-29 09:29:58UTC. Immutable source checkout
was verified in the worker. Failure occurred before original16 parity,new article
features or fitting:importing`Qwen3_5ForConditionalGeneration` entered
`fla.ops.simple_gla.parallel` and Triton JIT source extraction raised
`AttributeError: 'NoneType' object has no attribute 'start'`;Transformers wrapped
it as a model-class import error. This is an environment/import failure,NOT a
failed quality gate. No new model/head/gate/dev/test result exists. Controller
preserved the known parent and failed receipt;no retry/new allocation. Complete
verified large-training inputs and prior results remain valid.

Follow-up source hardening records Python and actual Torch/Transformers/FLA/
Triton versions and requires both `pip check` and the real model-class import
before the training driver reads inputs/hashes model weights. This is an
early-detection guard,NOT a verified dependency compatibility fix. Exact Linux
runtime reproduction remains required before any separately approved allocation;
do not change kernels,relax original16 parity or silently relaunch this campaign.
Terminal-failure page/feed deployed2026-10-29 09:34:58UTC. Receipt
`ROOT\jev-large-semantic-01-failure-publication-verification.json` binds the
preserved failed campaign and worker log,confirms Failed from a fresh cloud read,
and preserves all36 previous official-dev metric cells/shared assets/auth.
There is no new score or dependency-fix claim.

### JEV-LARGE-SEMANTIC-1R: qualified interpreter recovery, no scientific retuning

Autonomous continuation after the initialization failure:local isolated WSL
managed CPython3.11.9,uv0.6.14,with CUDA disabled,installed the original framework
pins and successfully imported the real`Qwen3_5ForConditionalGeneration`.
`pip check` passed;actual FLA kernel source inspection retained its definition.
Diagnostic`ROOT\jev-large-linux-runtime-original.json` SHA256
`2fdb94ba8b15852c5a495b8255bdf5c87d9561c77f574ab3d1a6635d53a9c6b3`.
No model weights were loaded,no model was fitted,no local CUDA execution.
This establishes a qualified alternative,NOT the exact image-specific root cause.

The worker now requires managed Python3.11.9 from pinned uv0.6.14,clears inherited
Python path/home overrides,and applies the complete version-only Linux
`requirements.lock.txt` generated from that successful environment. The original
direct dependency manifest and all scientific recipes/data/metrics/gates remain
unchanged. Real import/pip preflights remain mandatory;original16 float32-bit
parity is still required BEFORE any new article extraction or training.

One explicitly registered environment-only recovery parent
`jev_large_semantic_20261029_02`,unique output suffix`large_semantic_02`.
The original failed parent/receipt/outputs remain untouched. Only submit after
freshly confirming both original parent and worker Failed,CPU qualification
hash/contents,immutable source and complete input contract. No inspected new
feature,training,selection,dev or test result informed this recovery.

One serial Standard Ranking A100,worker18000s maximum,no automatic recovery
retry. Crucially the controller's GPU cutoff remains the FIRST parent's
**2026-10-29 15:06:23.720214UTC**,not a fresh six-hour window;queue/bootstrap/
installation consume that remaining allowance. Existing aggregate cutoff
**2026-10-30 01:17:07.923179UTC** is unchanged. All-data fixed3-epoch refit remains
conditional on the original gate;incomplete work is a failure,not a shortcut.
No automatic official-dev allocation. Complete artifacts and independent
verification remain prerequisites for a separately frozen official-dev readout.

Recovery02 was submitted09:51:39UTC but its controller correctly failed closed
before accepting the allocation:the Azure service serializes integer input
`max_seconds` as the canonical STRING`"18000"`,whereas the mocked SDK graph used
integer18000. Parent cancellation was requested09:51:56UTC;the known queued
worker had no start time or user-code log. Preserve this failed control-plane
receipt;do not describe it as a new model/runtime failure or a completed GPU fit.

The corrected continuation name is`jev_large_semantic_20261029_03`,unique output
`large_semantic_03`,only after FRESH terminal cancellation of both02 parent and
worker and confirmation of no02 worker start/user-code log. Accept the exact
canonical server literal`"18000"` as equivalent to integer18000,but reject
other limits/strings. All other source/allocation/input checks remain unchanged.
This does NOT reset the first parent's15:06:23.720214UTC cutoff or aggregate
cutoff. No overlapping workers,no scientific retuning,no automatic03 retry.

Independent post-training verification is prepared before any selection result:
all five arms'raw scores/complete source order/all per-impression and aggregate
metrics must be replayed. Derived neural CPU/GPU FP32 score replay has fixed
atol1e-4/rtol1e-5;ranking metrics are recomputed from the ORIGINAL stored raw
scores with atol1e-12 and the same NumPy1.26.4 tie ordering. This does not alter
the mandatory BIT-EXACT original16 frozen-JEV feature/margin parity gate.

Subsequent fresh RunHistory evidence supersedes the earlier queued snapshot:
02 worker actually STARTED09:53:48.272616UTC despite parent cancellation.
Parent is Canceled/Jobs child metadata CancelRequested,while the execution
RunHistory is Running and its user log proves exact a64561f checkout and managed
CPython3.11.9 bootstrap. Therefore03 preflight halted BEFORE any submission
intent/cloud parent;no03 worker overlaps02. Do not claim02 never ran or launched
only after terminal cancellation. No additional job is submitted to fix this.

Reconcile the ALREADY-EXISTING02 worker,with a separate immutable receipt from
the failed guardian. Revalidate exact source/contract/parity/canonical18000s/
single Standard A100 and retain canceled-parent metadata verbatim. Actual start
+18000s = **2026-10-29 14:53:48.272616UTC**,earlier than the unchanged original
15:06:23.720214UTC cutoff. The observer never submits/allocates anything and
stops on Failed/Canceled or the earlier deadline.

If,and only if,the actual worker reaches Completed,its exact checkout/runtime
import/parity and complete artifacts can be independently verified despite the
canceled orchestrator. Never describe the parent as Completed. Accepting any
training-only gate additionally requires complete all-five-arm source-order
score replay/all20 aggregate metric cells/all per-impression metrics,all-data
fixed3-epoch coverage and intact selected head. No automatic dev allocation.
This transparently disclosed orchestrator anomaly is not a quality result and
does not relax data,parity,metric,refit or resource requirements.

#### Recovery02 representation failure and numerical-runtime reconstruction

Observed2026-10-29 10:14UTC;subsequent process/platform evidence confirms the
user process exited1 at10:10:46.307916UTC and Singularity platform activity
reported Failed10:13:53UTC. Parent remains Canceled,Jobs child CancelRequested,
and RunHistory may remain stale Running. Preserve all three statuses rather
than treating stale RunHistory as continuing model fitting.

The actual GPU worker passed the managed-Python model-class import,then FAILED
mandatory original16 bit-exact parity:35236/40960 feature uint32 bit patterns
differ. The printed uint32 subtraction is NOT a real-valued feature error.
Margin parity was not reached. No new article extraction,neural training,
selection gate,final refit or new official-dev/test score exists. Keep the
original comparison precision and all scientific/data/resource constraints.

Runtime reconstruction found a concrete previously missed version-drift
mechanism:successful stage00 used fresh Conda Python3.11.18 and PyPI
Torch2.6.0,not the recovery's2.6.0+cu124 build. Its broad requirements resolve
FAILED2026-10-28 15:28:01UTC and was silently treated as nonfatal;the nominal
Transformers5.2.0 pin therefore does not prove its installed version.
Flash-linear-attention0.4.2 was installed afterward and its published metadata
depends on unpinned Transformers. The immutable AML image Dockerfile itself
pins4.57.1 and Python3.10,not the working Qwen3.5 implementation.
Official PyPI release timestamps before the actual FLA installation identify
Transformers5.21.0,tokenizers0.23.3 and Hub2.3.0 as the reconstructed candidate.
This is a resolver/source-based inference,NOT directly captured old runtime
metadata or a demonstrated parity fix. A separate isolated CPU environment
qualifies that candidate;do not mutate the preserved5.2 qualification.

Before any further GPU allocation,the previous process/platform terminal
evidence must be authenticated and recorded,original absolute GPU and campaign
cutoffs retained,and a new explicit bounded continuation registered. Do not
blindly launch the never-submitted03 candidate. No overlapping allocation and
no automatic retry. CPU import success is insufficient:the original bit-exact
16-row gate remains mandatory before article extraction or fitting.

The trainer now persists model byte hashes before loading and runtime/module
source hashes,token-input hashes,and actual16-row features/margins before the
unchanged parity assertion. A comparison report separates differing bit counts
from real-valued absolute errors. This improves failure evidence,not acceptance
thresholds. CPU fixtures cover signed-zero/one-ULP/nonfinite differences and
verify that failure preserves artifacts without reaching any new articles.

#### Bounded reconstructed-runtime continuation04

Registered2026-10-29 UTC before any04 submission,after independent CPU-only
qualification of the reconstructed candidate. Managed Python3.11.9,
PyPI Torch2.6.0/torchvision0.21.0,Transformers5.21.0,tokenizers0.23.3,
Hub2.3.0,safetensors0.8.0,FLA/fla-core0.4.2,Triton3.2.0 pass pip check and
the real Qwen3_5 class import with CUDA disabled. A complete new package
lock freezes the actual qualified environment. Preserve the earlier5.2
qualification and failed02 receipts. This is a runtime reconstruction,
NOT evidence that original16 parity or any training/quality gate now passes.

Unique new parent`jev_large_semantic_20261029_04`,unique output
`large_semantic_04`;never submit the old03 candidate. At most one new Standard
Ranking A100,worker14400s maximum,no automatic retry. The ORIGINAL
2026-10-29 15:06:23.720214UTC GPU cutoff and aggregate
2026-10-30 01:17:07.923179UTC cutoff remain unchanged. Submission requires
more than14400s plus600s setup/queue margin before the original GPU cutoff;
cancel if a delayed actual worker start plus14400s would exceed that cutoff.
The controller retains the earlier absolute deadline in all cases.

Because canceled-parent metadata stopped propagating,02's authenticated
execution-wrapper completion (all user processes collected,exit1) plus the
Singularity platform's final Failed event are the terminal-execution evidence,
not the stale RunHistory Running value. Require fresh readback of both logs,
identical hashes to the preserved failure receipt,exact old worker/source,
zero existing04 parent and zero03 parent before any submission intent.
Record parent/Jobs/RunHistory statuses verbatim and the platform terminal event
separately. A traceback or canceled parent alone is insufficient. No overlap,
permission workaround,status relabeling,scientific retuning or budget reset.

All data/recipes/three-epoch coverage/internal gate/final-all-training-refit
requirements stay fixed. Persist runtime/source/model/input hashes and the
actual16-row output,then require unchanged BIT-EXACT features AND margins.
Failure stops04 before any new articles;do not loosen tolerances or retry.
Only a genuinely Completed worker with independently verified complete output
can establish an internal gate. Official dev requires separate registration;
competition-test AUC>=0.80 remains unmet.

Actual04 submission confirmed2026-10-29 10:45:27.767650UTC,parent
`jev_large_semantic_20261029_04`,source
`ce4a8e6939eccaecefc37603e72dfc90fa0fcfdd`. Server allocation revalidated:
one Standard Ranking A100/14400s,canonical max_seconds`"14400"`,complete
downloaded contract/original16 archive and exact unique output. Worker
`389888ac-193e-4133-84e6-94514d4b29f2` Starting at10:45:38UTC.
Fresh02 wrapper/platform hashes matched the preserved failure receipt before
submission despite Jobs and RunHistory now both stale Running. No03 parent
exists. Controller`run_jev_large_semantic_reconstructed.py`,attached shell
`jev-large-semantic-04`;state`jev-large-semantic-04-campaign.json`.
No actual04 parity or new feature/training/quality result yet exists at this
submission snapshot. The absolute GPU and aggregate cutoffs are unchanged.
The qualified04 page/feed was deployed10:48:13UTC,receipt
`jev-large-semantic-04-start-publication.json`;all36 previous official-dev
metric cells and other shared assets/auth preserved. Actual worker Queued at
10:48:53UTC;no worker start,parity pass or training/quality result claimed.
The prepared complete independent verifier now additionally rereads the
original16 artifact and verifies every feature AND margin bit;five fixtures
cover exact data,signed zero,one ULP,dtype,reference and inventory changes.

#### Actual04 completion and independent numerical-dispatch diagnosis

Parent04 and worker`389888ac-193e-4133-84e6-94514d4b29f2` both Completed,
observed2026-10-29 12:56:34UTC,inside the original training GPU cutoff.
Original16 feature AND margin bit-exact parity passed. The worker extracted
all101527 training article representations and reports a passing train-only
gate,then a fresh three-epoch candidate-attention fit on all2232748 training
impressions/83507374 exposures/3383656 positives per epoch,zero final exclusions.
Frozen final head SHA256:
`c9523970501838fa165e7995400608cbc34527c5ed37a82035f81acd7bc35c6d`.
Frozen result SHA256:
`5c5169a84b3a092762694a32b12b6c927dd99780d660a4db3cde99b0240d2432`.
These are not official-dev/test quality results or evidence of AUC>=0.80.

The first independent replay failed per-impression metric equality on the
popularity arm. Actual raw scores/AUC matched,but local NumPy1.26.4 AVX512
argsort ordered equal scores differently:the first impression's MRR was
1/31 locally versus1/28 on the worker. Calling the original official metric
helpers locally produced the same discrepancy,isolating numerical dispatch
rather than an independent metric formula error. With AVX512 NumPy dispatch
disabled,the first10000 complete impressions/all40000 metric cells reproduce
worker metrics to maximum1.1102230246251565e-16. The complete all-five-arm replay
is rerunning with this worker-compatible dispatch. Keep the failed log.
No score/metric/parity tolerance,recipe,gate or recorded worker metric changes.
The training gate is NOT independently accepted until the whole replay passes.

#### Separately bounded complete official MIND-large dev evaluation

Registered2026-10-29 UTC before reading the actual new large-dev contents or
allocating this evaluation. Conditional on the complete independent training
proof accepting the unchanged internal gate and final all-training head above.

- Unique parent`jev_large_frozen_dev_20261029_01`,output`large_frozen_dev_01`.
  One Standard Ranking A100,one worker21600s maximum,no automatic retry,no
  concurrent training allocation. Six hours from parent creation includes
  queue/bootstrap/worker;the guardian cancels the parent at that deadline.
  This is a separate conditional evaluation,NOT a training retry or reset:
  the old training GPU cutoff remains unchanged,and the ORIGINAL aggregate
  ceiling2026-10-30 01:17:07.923179UTC remains binding. Only submit when a full
  six hours plus600s margin fit before that ceiling.
- Primary population:EVERY configured official MIND-large dev impression,
  every candidate in raw order,all positive targets,no subsampling or skipped
  impressions. Verify configured complete source bytes/identities/metadata;
  do not newly assert independently established canonical public provenance.
  Retain the fixed last30 history with explicit left-zero padding.
- Freeze the final candidate-attention head and complete all-training feature
  prefix bit-for-bit. Reuse training article identities only if consumed
  article metadata agrees exactly. Append sorted new dev article identities
  under the same article-only fixed512-token prompt contract and unchanged
  original16 BIT-EXACT feature AND margin gate. Rehash frozen model bytes
  before/after extraction and require exact training model fingerprint equality.
- No optimizer,model fitting,recipe selection,early stopping or dev-driven
  calibration/tuning. Four prespecified arms:the frozen final candidate_attention,
  popularity from ALL official training impressions,untrained semantic_mean,
  and frozen_article_margin. New article popularity uses the all-training
  global click prior,not a statistic from dev labels.
- Compute equal-impression AUC,official multi-positive MRR,nDCG5,nDCG10.
  Freeze NumPy1.26.4 default argsort-reversed tie semantics with non-AVX512
  dispatch;set NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512_SKX,AVX512_CLX,
  AVX512_CNL,AVX512_ICL before import and record effective CPU dispatch.
- Conservative overlap disclosure treats ALL official MIND-small dev as
  previously exposed. Signature=SHA256 of compact JSON containing last30
  ordered history news IDs and the sorted candidate-news multiset,excluding
  labels,impression IDs,users and time. Report overlap counts and a prespecified
  signature-unmatched secondary cohort in addition to the complete primary.
  Unmatched does NOT guarantee pristine or never inspected. Do not select
  a favorable window after seeing results.
- Full raw-source/encoded-identity/history/label/overlap replay,original16 bits,
  exact article-prefix/new-article union,all four raw score vectors,every
  per-impression/aggregate metric cell and frozen-head fingerprint must be
  independently verified before quality claims. Derived CPU/GPU score tolerance
  remains atol1e-4/rtol1e-5;raw-score metric replay remains atol1e-12/rtol0;
  original16 remains BIT-EXACT. The independent verifier does not reproduce
  historical optimizer trajectories.
- A matched retained-hybrid evaluation on this complete population is not
  available in this bounded design. Do not compare this result to historical
  partial-small-dev numbers as a matched improvement. Official-dev results
  cannot satisfy the competition-test AUC>=0.80 target.

## JEV-CONTENT-1: autonomous continuation toward AUC 0.80

Registered 2026-10-28 UTC, before fitting or inspecting new dev labels.
The completed full-training campaign is not restarted or its budget extended.

- New hypothesis: a nonlinear ranker can combine the unchanged frozen JEV
  margin with candidate-aware semantic history, category/subcategory affinity,
  and cross-fitted popularity, rather than only adding a linear correction.
- CPU selection: four threads, at most three fixed histogram-boosting recipes
  and one final refit, no backbone fitting. TF-IDF/SVD is fitted on official
  training news only. No dev metadata or labels enter recipe selection.
- Internal selection uses the last calendar day of official training as a
  temporary chronological validation split. All its impressions are excluded
  from every target-dependent statistic, including statistics of fitting rows.
  Fitting-row popularity additionally excludes that row's entire impression
  fold. Only the original untrained frozen JEV margin is read from the cache;
  supervised full-data text/hybrid margins must not enter this selection.
- Fixed recipes: frozen-margin/popularity control (15 leaves,120 iterations),
  content model (15 leaves,120 iterations), content model (31 leaves,160 iterations).
  Learning rate0.06,minimum leaf100,L2=10,seed42,no early stopping; each sampled
  impression has unit total weight. Highest internal impression AUC selects
  the content recipe, with listed order breaking ties.
- Allocation gate: selected content model must exceed the control's internal
  AUC by at least0.005. Internal metrics are sampled-training selection only,
  never official-dev, competition-test, or SOTA evidence.
- If the gate opens, refit the selected recipe on all156965 official training
  impressions/all469264 existing sampled rows. All5843444 raw training
  exposures contribute to popularity counts. No impression remains held out.
- Register the next independent official MIND-small dev window as
  zero-based36000 through40999 (impressions36001-41000),5000 impressions,
  with every candidate in source order. No tuning/reselection on that window.
  Candidate scores must be computed using the frozen selected model.
- At most one new Standard Ranking A100 evaluation parent, one
  `Singularity.ND12am_A100_v4`,21600-second worker limit and six hours from
  parent creation. Evaluation only, no GPU training, no infrastructure retry.
  Submit only after the CPU gate passes and the frozen evaluator is verified.
  Existing source-bundled frozen controls must be scored on that same window.
- Do not claim promotion or project completion from train-only metrics,
  an unmatched previous window, a partial-dev AUC, or merely finishing this
  stage. Ultimate acceptance remains competition-test AUC >=0.80.

CPU preprocessing recovery: the first attempt stopped on the exact row-order
guard before any candidate fit. Cached margins start with impressions18400,
60823,5864,138636,146239, whereas source behaviors start1,2,3,4,5.
Align source contexts by impression/candidate/label to the untouched cached
order; keep the full identity/order equality check. Original protocol/log and
output directory are preserved; use a new output directory for the corrected
attempt. No recipe, selection split, model weights, dev window, or GPU budget
was changed. Ten targeted ranker/evaluation fixtures pass after the correction.

Historical CPU milestone before dev evaluation: content_31 selected on the internal training split,
then refitted on all156965 impressions/469264 sampled rows, zero final holdout
exclusions. Model SHA256
`1a26a04bfafc220f99759ed246cf43d8bd26fbf416d3aa1a4bf99fb4e7c187a8`.
The population archive is byte-identical to the independently verified
all-exposure training population; training news metadata matches exactly.
At that milestone no new official-dev score was available. Internal selection is not quality
evidence. The single registered frozen-control GPU parent is
`great_bee_w6zjxd296x`, submitted2026-10-28 07:09:03UTC; its original creation
deadline is2026-10-28 13:09:03UTC. Server source/one-A100/Standard/21600s/
dev-start36000 inputs were checked. Guarded controller
`run_jev_content_stage.py` owns this parent; do not relaunch it or create a
duplicate. The completed older full-training campaign is unchanged.

### JEV-CONTENT-1 official-dev outcome: reject the new candidate

Both parent and worker completed within the registered creation deadline.
Official MIND-small dev36001-41000 contains5000 impressions/189786 candidates.
These are official-dev metrics, not the sampled internal training selection.

| Arm | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| New content_31 |0.66352816|0.32453836|0.35758897|0.41642992|
| Retained hybrid |0.68333835|0.32895720|0.36757944|0.42566814|
| Frozen JEV |0.62876796|0.29765285|0.32437785|0.38393704|
| Full-training text control |0.66834303|0.32210110|0.35475335|0.41710299|
| Retained text control |0.67152351|0.32330346|0.35721167|0.41691052|

Primary new-minus-retained AUC=-0.01981019667664715, paired95%CI
[-0.026089836906870292,-0.01361294879170905],2048 impression resamples,
seed42,unadjusted. Reject content_31 and retain the prior hybrid. MRR delta
-0.00441884 has CI crossingzero; nDCG5/10 regress with entirely negative
intervals. All20 metric cells,exact source coverage,four paired intervals,
and128 full-impression frozen content/tree and retained-correction predictions
were replayed. Result SHA
`5bd0c1bb3de4fb5de7555e63adb21a03bb5ffe8515dedd20f6e89e877d7276b3`;
raw scores SHA
`54528c1477c9bb953beb060677ec5c244879609fcda22828c9b0d6e797cecd45`.

The different-window retained value0.68333835 is not an improvement over
earlier-window0.67598829. The new classifier used frozen margins and compact
TF-IDF/SVD features, not the retained2560-dimensional text readout. Its failure
does not causally isolate which architectural/data/objective choice mattered.
No new GPU allocation or retry is authorized by this completed stage.
The competition-test AUC>=0.80 goal remains unachieved.

The final negative-result report was deployed successfully on2026-10-28
at13:08UTC. Publication receipt
`jev-content-01-final-publication-verification.json` binds the result,
raw scores,page,and feed; other site assets/auth are unchanged.

### CPU-only full-feature cache audit after the content-stage rejection

Continue without allocating another GPU or tuning on the now-exposed dev
window. Restore all469264 training rows'2560-dimensional frozen representations
from the existing checksum-pinned legacy/holdout/four stage caches. Preparation
is bounded to two hours,four CPU threads,and at most6GB of unique artifact
payload,with the existing four-attempt identical-read transport policy.
Use ignored data-drive staging
`Q:\CodeGuru\scratch\jev-train-cache-20261028`, not the system drive, which had
only6.5GB free at preflight. Existing artifacts and failures are preserved.

Require whole-source checksums,exact canonical row metadata,unique complete
candidate/label/order coverage,exact frozen-margin equality for every fitting
row,existing128 hidden-feature spots,and independent full-matrix text-head
mean/scale reconstruction. No new fitting,dev labels,model-quality metric,
GPU parent,or acceptance claim belongs to this cache-preparation stage.
After the audit, any further model recipe/selection split must be registered
before fitting; supervised retained-head training cohorts must be excluded
from that head's internal validation. The feature-family change in
JEV-CONTENT-1 is an observation,not a proven causal diagnosis.

Preparation started2026-10-28 13:10:46UTC. The four targeted cache-controller
tests passed, and the six immutable source/row mappings were checked before
launch. The first artifact downloads were observed progressing on the data
drive. Controller `prepare_jev_full_feature_cache.py`, shell
`jev-full-feature-cache-preparation`, log
`jev-full-feature-cache-preparation.log`; do not launch a duplicate or reset
its budget. This is running preparation, not a completed audit or new score.

Audit completed2026-10-28 13:32:38UTC: all source hashes, exact unique
coverage/frozen margins,128 feature spots,and full469264-row normalization
reconstruction passed. Final float32 matrix shape469264x2560, SHA256
`01b41d83408ddda2d697f83d667f8358fc4015c6fa88f81ae9dab4126017c701`.
No model-quality result or GPU allocation follows from this audit.
An additional independent replay matched ALL469264 previously saved full-text
training scores exactly (maximum absolute error0.0), not just128 spots.
Proof `jev-full-cache-all-text-replay.json`; this verifies assembly/model
replay, not an improvement in benchmark quality.

### JEV-JOINT-1: preserve rich features and test impression-balanced objectives

Registered after the cache audit and before fitting. CPU only, four threads,
two hours including optional final refit, no new GPU/dev evaluation.
Unlike the rejected content-only model, these heads retain all2560 frozen
representation dimensions and add the existing four popularity features.

- Temporary validation: official training's last calendar day, excluding
  every impression in the retained head's original131071-row fitting cohort
  AND its original256-impression selection holdout. This is cohort-disjoint
  last-day selection, not a fully chronological split: other last-day
  impressions may train the models. No official-dev labels are read.
- All temporary validation impressions are excluded from ALL population
  counts and normalization fitting. Fitting population features additionally
  exclude their entire impression hash fold. Validation uses fitting-only
  population totals. Do not reuse the retained behavioral counts or head.
- Strong controls: retained text head plus a freshly fitted pairwise
  popularity correction (L2=0.01), and joint rich-text/popularity binary
  row-weighted head (L2=0.01). Original retained text SHA279ef6c4 is fixed;
  its fitting/selection cohorts are bound to their existing exact hashes.
- Three candidates, fixed order: impression-balanced binary L2=0.01,
  impression-balanced binary L2=0.1, equal mixture of impression-balanced
  binary and equal-impression pairwise loss with L2=0.01.
  Binary components use every fitting row, including single-class groups.
  Pairwise components omit only groups without a contrast; they never
  remove those groups from binary training or population statistics.
- Float64 streamed L-BFGS, maximum100 iterations,2564 train-normalized
  features, no giant float64 full-matrix copies. No stochastic search.
  Highest internal impression AUC selects the candidate; fixed order breaks
  ties. Gate requires >=0.005 AUC over BOTH controls and no MRR/nDCG5/nDCG10
  regression against either. These are sampled-training selection metrics,
  not official-dev/competition evidence or an independent confirmation.
- Only if that gate passes, refit the selected head on all156965 official
  training impressions/all469264 sampled rows, with all5843444 raw exposures
  contributing to counts and zero final holdout exclusions. Bind final
  model/population/selection/inputs by checksum.
- This stage authorizes no GPU submission. Any later evaluation needs a
  separately registered protocol; competition-test AUC>=0.80 remains unmet.

Preflight established23988 cohort-disjoint validation impressions and132977
fitting impressions. The retained head's fitting plus original selection
cohorts contain31200 impressions. Eighteen targeted streamed-head/population
tests passed, including finite-difference gradients for all four objectives,
dense binary equivalence, excluded-label/feature invariance,and the strict
two-control allocation gate.

Solver recovery registered2026-10-28 after the first joint binary CONTROL
hit its100-iteration limit at13:58UTC. No candidate validation score or final
refit was produced; the one retained-text/fresh-popularity internal control
score is not benchmark evidence. Preserve the failed state/log/controller.
The dense prior uses L-BFGS history100 (confirmed from installed PyTorch);
the streamed solver had inherited SciPy's shorter default history.
Set the streamed solver's history explicitly to100 and expose iteration/loss
progress plus numerical nonconvergence diagnostics. This is a solver-memory
recovery hypothesis, not a proven explanation until convergence is observed.
Keep100 maximum iterations,identical tolerances/objectives/recipes/split/gate
and four threads. Do NOT accept unconverged heads. Use a separate recovery
directory, with a conservative recovery cutoff2026-10-28 15:38:04.949646UTC
(one minute before the initial state timestamp plus two hours, accounting
for its observed six-second import startup); do not reset the original
two-hour budget. No GPU/dev evaluation or original result is overwritten.

Read-only stationarity accounting on the ORIGINAL text heads found maximum
absolute objective gradients9.3865e-6(full) and1.3586e-5(retained), under their
row-weighted binary L2=0.01 objectives. Both are small and similar; this does
NOT identify numerical convergence as the cause of the dev regression.
Proof `jev-original-head-stationarity-diagnosis.json`; no fits/dev reads/GPU.

JEV-JOINT-1 stopped after its recovery also hit100 iterations, at14:20:29UTC.
Loss0.603812956998,maximum gradient1.48788e-5. Increased optimizer history
did NOT establish convergence; do not retry that control, loosen tolerances,
evaluate an unconverged head, or describe the stage as a quality result.
Neither attempt produced candidate validation scores or a final model.

### JEV-LOWRANK-1: fixed rich-text base with a compact whitened residual

Registered before fitting; a NEW statistical model, not another retry or
acceptance of the failed2564-parameter control. Keep the existing CPU cutoff
2026-10-28 15:38:04.949646UTC, four threads, no GPU/dev evaluation.

Preserve the exact retained2560-dimensional text head as the BASE margin.
Add a low-rank learned correction rather than replacing that strong base:
fixed seed42 Gaussian projection of frozen features into64 dimensions,
plus the same four popularity features. Fit centering/whitening ONLY on the
same132977 fitting impressions; discard covariance directions with eigenvalue
<=max(largest_eigenvalue*1e-6,1e-12). The23988 last-day validation impressions
still exclude both the retained head's fitting and original selection cohorts.
Reuse checksum-bound fitting-only population/counts and freshly fitted
four-feature pairwise control from the stopped stage, not old dev-tuned counts.

Three fixed residual recipes: impression-balanced binary L2=0.01,
impression-balanced binary L2=0.1, and half balanced-binary/half equal-impression
pairwise L2=0.01. The unchanged streamed solver must genuinely converge within
100 iterations/history100/unchanged tolerances. Controls are the fixed retained
text margin and that retained text plus its training-only fresh popularity
correction. The best candidate must beat BOTH by>=0.005 internal AUC with no
MRR/nDCG5/nDCG10 regression. Fixed recipe order breaks ties.

Only if the gate passes, refit whitening and the selected residual on all
156965 training impressions/all469264 sampled rows/all5843444 raw exposures;
keep the fixed Gaussian projection and retained base, zero final holdouts.
All whitening/model/population/selection artifacts are checksum-bound.
Compact residual rank is at most68; retaining the rich base distinguishes
this hypothesis from the rejected compact content-only replacement.
This is train-only selection, not benchmark improvement or AUC0.80 evidence.
Twenty-eight targeted fixtures passed before launch, including all three
compact recipes' synthetic convergence, immutable rich-base preservation,
validation-excluded whitening,constant-direction removal,and model archive
prediction replay. No tolerance/iteration/gate relaxation was made.

JEV-LOWRANK-1 completed2026-10-28 14:31:49UTC. All three recipes genuinely
converged in5/6/7 iterations, but the gate CLOSED:

| Internal sampled-training validation arm | AUC |
|---|---:|
| Retained rich text |0.69570720|
| Retained rich text + safe fresh popularity |0.71339398|
| Compact balanced binary L2=0.01 |0.71441457|
| Compact balanced binary L2=0.1 |0.71199909|
| Compact mixed objective L2=0.01 |0.71504886|

Best gain over the strong control was0.00165488, below the registered0.005.
Do not lower the threshold or call0.71505 an official-dev result.
No final all-data refit or GPU/dev evaluation was performed.
Independent verification at14:36:23UTC replayed all20 metric cells, ALL five
arms'66042 validation-row predictions, exact original-cohort exclusions,
entire raw training population excluding all validation impressions, and
all403222 fitting rows' whitening mean/covariance. Proof
`jev-lowrank-01-independent-verification.json`. Retain the previous model;
competition-test AUC>=0.80 remains unmet.

### Read-only negative-sampling coverage audit

Before any further head sweep or GPU allocation, inspect the existing
training sampler's population support. Source inspection found that the
legacy1:1 sampler floors the requested hard count; with one positive it
requests ZERO same-positive-category negatives, and does not backfill from
hard negatives when the easy stratum is empty. Quantify this on the actual
official training data and reproduce the checksum-pinned469264 cached rows
exactly before attributing any effect. This is a potential training-support
limitation, not a proven cause of the dev regression.

CPU only/four threads/within the same15:38:04.949646UTC cutoff. No model fits,
dev reads, default sampler changes, or GPU submissions. Count zero-probability
raw-negative exposures, avoidable single-class sampled impressions, and
cache reuse/new-feature demand for a fixed seed42 uniform-without-replacement
alternative. A future corrected sampler/evaluation needs separate registration;
do not silently reinterpret prior caches as uniformly sampled.

Audit completed2026-10-28 14:40:33UTC and reproduced ALL469264 historical
cached rows in exact order. The legacy sampler gives480584 raw-negative
exposures (8.57099% of5607100) ZERO inclusion probability, across81033
impressions. It produces3212 avoidable positive-only sampled impressions
despite every raw impression containing negatives. These are proven
sampling-support defects, not a proven causal explanation of model quality.

Register a CPU-only corrective preparation: add an explicit opt-in
`negative_sampling="uniform"` mode, preserving the legacy DEFAULT and its
exact row sequence. Uniform sampling is without replacement, size
min(number_of_positives,number_of_negatives),seed42. Prepare metadata/reuse
indices only, not a model or GPU job. Reproduce both the legacy metadata SHA
and independent uniform-preview SHA before accepting it. Expected corrected
population:472626 rows/all156965 impressions,236344 positive/236282 negative;
zero sampled single-class groups. Existing verified features can supply266352
rows;206274 new frozen feature rows would still be needed. That missing work
is NOT submitted or authorized by this preparation. Preserve all old caches,
do not change existing training/evaluation defaults, and use the same CPU
cutoff. Benchmark benefit must still be established by a future controlled run.

Corrective metadata preparation completed2026-10-28 14:48:17UTC, source
`1e75297d0a72862453e15da4838c455bd0b66a76`. Sixteen targeted tests plus15
subtests passed, including existing JEV decision/dataset behavior.
Actual full-data construction reproduced legacy DEFAULT rows/order/LF bytes
SHA`b4dba4f73cdf5878ba1d67582d8b81899240e13c3670f38b8b26ee04605af254`
and independently previewed uniform rows
SHA`fd9f50c2668cd50ae03f918636010f84e43563c73c711997ef721ea30a9a3a84`.
All472626 corrected identities and266352 reuse mappings were checked;206274
pending rows are negatives, no new positive extraction is needed.
Ignored output`Q:\CodeGuru\scratch\jev-uniform-01-preparation` contains only
metadata/pending rows/reuse indices/proof. No feature values were fabricated,
copied, or fitted; no GPU extraction was submitted. The corrected model is
NOT trained and has no benchmark score. No active CPU/GPU experiment remains.

The shared research page and feed deployed successfully2026-10-28
14:55:44UTC with the verified sampler defect, corrected-preparation status,
and closed compact-model gate. Publication receipts
`jev-sampling-audit-publication-verification.json` and
`jev-sampling-audit-publication-content-verification.json` bind these proofs.
All20 prior official-dev table cells and other site assets/auth are unchanged.
No signed-in browser-rendering verification or new benchmark result is claimed.

## 🚀 Azure ML Pipeline Support

This experiment plan has been updated to use **Azure ML pipelines** for scalable training and evaluation on A100 GPU clusters.

**Available Pipelines:**
- ✅ **Training Pipeline** (`pipeline/run_pipeline.py`) - Point-wise SFT with DeepSpeed + **integrated evaluation**
- ✅ **Evaluation Pipeline** (`pipeline/run_eval_pipeline.py`) - Standalone evaluation (optional, for re-evaluation)
- ✅ **CoT RL Pipeline** (`pipeline/run_cot_rl_pipeline.py`) - Chain-of-Thought RL training + **integrated evaluation**

**Pipeline Features:**
- 8x A100 80GB GPUs per job
- Automatic environment setup
- DeepSpeed multi-GPU training
- **Integrated evaluation** after training (saves results to checkpoint directory)
- Model checkpoints saved to mounted datastore (`shares/users/wuc/output_dir/`)
- Experiment tracking in Azure ML Studio

**New Parameters:**
- `--output-root`: Model output directory (default: `shares/users/wuc/output_dir`)
- `--run-eval`: Run evaluation after training (default: 1)
- `--eval-split`: Evaluation split - dev/test (default: dev)
- `--use-abstract`: Use news abstracts in addition to titles (default: 0)

**Coming Soon:**
- Multi-task training pipeline
- RL fine-tuning pipeline
- Ensemble & cascade evaluation

See [pipeline/README.md](pipeline/README.md) for detailed pipeline usage.

---

## Table of Contents

1. [Current Status](#current-status)
2. [Available Approaches](#available-approaches)
3. [Experiment Phases](#experiment-phases)
4. [Detailed Experiment Configurations](#detailed-experiment-configurations)
5. [Execution Schedule](#execution-schedule)
6. [Tracking & Evaluation](#tracking--evaluation)

---

## Current Status

### Best Results Achieved

| Approach | Model | Dataset | AUC | MRR | nDCG@5 | nDCG@10 |
|----------|-------|---------|-----|-----|--------|---------|
| Zero-shot baseline | Qwen3-1.7B | MINDsmall | 56.41% | 25.32% | 26.83% | 32.99% |
| Zero-shot baseline | Qwen3-4B-Instruct-2507 | MINDsmall | 58.11% | 27.44% | 28.83% | 35.01% |
| Point-wise SFT | Qwen3-4B-Instruct | MINDsmall | 66.30% | 31.28% | 34.77% | 41.10% |
| Point-wise SFT | Qwen3-1.7B | MINDsmall | 67.68% | 33.62% | 37.38% | 43.32% |
| Point-wise SFT | Qwen3-1.7B-Base | MINDsmall | 69.39% | 33.70% | 37.61% | 43.82% |
| Point-wise + RL | Qwen3-1.7B-Base | MINDsmall | **69.69%** | 34.02% | 38.02% | 44.23% |
| Ranking SFT | Qwen3-1.7B-Base | MINDsmall | 66.17% | 45.20% | 50.09% | 56.45% |

### Reference Baselines

| Model | AUC | Source |
|-------|-----|--------|
| NRMS | 67.76% | Published baseline |
| MIND Leaderboard SOTA | ~72.72% | Competition |

---

## 📊 Experiments Tracking Table

### Progress Summary

| Phase | Total | Pending | Running | Completed | Failed | Best AUC |
|-------|-------|---------|---------|-----------|--------|----------|
| Phase 1: Point-wise SFT | 6 | 6 | 0 | 0 | 0 | - |
| Phase 2: RL Fine-tuning | 4 | 4 | 0 | 0 | 0 | - |
| Phase 3: Ensemble & Cascade | 5 | 5 | 0 | 0 | 0 | - |
| Phase 4: Multi-task | 3 | 3 | 0 | 0 | 0 | - |
| Phase 5: Advanced | 3 | 3 | 0 | 0 | 0 | - |
| **Total** | **21** | **21** | **0** | **0** | **0** | **-** |

**Historical baseline snapshot**: 0/21 (0%) | Baseline: 69.69% AUC | Historical target: 72.72% AUC.
**Current project acceptance target**: competition-test AUC >=80.00%; not achieved.
Do not treat the historical baseline or a partial MIND-small dev result as the
same benchmark when computing the remaining target gap.

---

### Completed Experiments Archive

| Exp ID | Date | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|------|-------|---------|--------|-----|-----|--------|---------|-------|
| **Baseline** | 2026-02 | Qwen3-1.7B-Base | MINDsmall | Point-wise + RL | 69.69% | 34.02% | 38.02% | 44.23% | Starting point |

---

### Phase 1: Point-wise SFT Optimization

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **P1.1** | ✅ Completed | Qwen3-1.7B | MINDsmall | NEG_RATIO=3.0 | 0.6845 | 0.3303 | 0.3665 | 0.4285 | More negatives; checkpoint-17835; (pre-pipeline, no W&B) |
| **P1.2** | ✅ Completed | Qwen3-1.7B | MINDsmall | NUM_EPOCHS=7 | 0.6886 | 0.3363 | 0.3738 | 0.4345 | Longer training; [W&B](https://wandb.ai/wuchen/MIND/runs/8a6omaep) |
| **P1.3** | 🔄 Re-run | Qwen3-1.7B | MINDsmall | USE_ABSTRACT=1, CUTOFF=4096 | 0.6662 | 0.3279 | 0.3648 | 0.4252 | With abstracts (prev cutoff=2048, re-running with 4096); [W&B](https://wandb.ai/wuchen/huggingface/runs/awafic20) |
| **P1.4** | ✅ Completed | Qwen3-1.7B | MINDsmall | MAX_HISTORY=50 | 0.6886 | 0.3338 | 0.3709 | 0.4327 | More history; [W&B](https://wandb.ai/wuchen/huggingface/runs/vnjv3dlq) |
| **P1.5** | ✅ Completed | Qwen3-1.7B | MINDsmall | NEG=3.0, EP=7, HIST=50 | 0.6804 | 0.3292 | 0.3662 | 0.4284 | Combined best; [W&B](https://wandb.ai/wuchen/huggingface/runs/rpifd679) |
| **P1.6** | ⬜ Pending | Qwen3-4B-Instruct | MINDsmall | Default + 4B model | - | - | - | - | Scale to 4B (first try before 8B) |
| **P1.7** | ✅ Completed | Qwen3-1.7B | MINDsmall | USE_SUBCATEGORY=1 | 0.6767 | 0.3303 | 0.3670 | 0.4281 | Add subcategory to prompt; [W&B](https://wandb.ai/wuchen/huggingface/runs/2g7326e9) |
| **P1.0** | ✅ Completed | Qwen3-1.7B | MINDsmall | ep5 default baseline | 0.6861 | 0.3326 | 0.3702 | 0.4301 | anchor for all P1.x comparisons; [W&B (old)](https://wandb.ai/wuchen/MIND/runs/9d7qw5o7) · [W&B (new)](https://wandb.ai/wuchen/huggingface/runs/er5a0t4z) |

**Status Legend**: ⬜ Pending | 🔄 Running | ✅ Completed | ❌ Failed

---

### Key Finding: Abstract in Training vs Evaluation

> **Abstract helps at BOTH train and eval time on MINDlarge (E1.1 result, 2026-04-27).**
>
> E1.1 directly tested: L1.3 (abstract-trained) evaluated WITHOUT abstracts → 0.6945 AUC,
> which is 0.0104 LOWER than the same model evaluated WITH abstracts (0.7049).
> The hypothesis that "title-only eval is better" was disproved for MINDlarge.
>
> **Rule: always match train and eval abstract setting. For abstract-trained models, use `USE_ABSTRACT=1` at eval.**

---

### Phase 1L: Point-wise SFT on MINDlarge

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **L1.1** | ✅ Completed | Qwen3-1.7B | MINDlarge | NEG_RATIO=3.0, HIST=30 | 0.6883 | 0.3316 | 0.3678 | 0.4310 | NEG=3.0 hurts on large too; [W&B](https://wandb.ai/wuchen/huggingface/runs/49ovq991) |
| **L1.2** | ✅ Completed | Qwen3-1.7B | MINDlarge | NUM_EPOCHS=7 | 0.6932 | 0.3359 | 0.3738 | 0.4373 | checkpoint-70144; [W&B](https://wandb.ai/wuchen/huggingface/runs/h70c32sr) |
| **L1.3** | ✅ Completed | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1 | **0.7049** 🏆 | 0.3461 | 0.3847 | 0.4479 | **NEW BEST**; [W&B](https://wandb.ai/wuchen/huggingface/runs/d3dr5t8d) |
| **L1.4** | ✅ Completed | Qwen3-1.7B | MINDlarge | MAX_HISTORY=50 | 0.6863 | 0.3310 | 0.3675 | 0.4307 | HIST=50 hurts on large; [W&B](https://wandb.ai/wuchen/huggingface/runs/gskwfdbe) |
| **L1.5** | ✅ Completed | Qwen3-1.7B | MINDlarge | NEG=3.0, EP=7, HIST=50 | 0.6863 | 0.3310 | 0.3675 | 0.4307 | HIST=50 bottleneck; [W&B](https://wandb.ai/wuchen/huggingface/runs/36ur6v25) |
| **L1.6** | 🔄 Running | Qwen3-4B | MINDlarge | USE_ABSTRACT=1, micro_bs=4 | - | - | - | - | ckpt-37376 (19% train) = 0.6819; resumed with timeout+patience fixes; final expected 0.72-0.73+ |
| **L1.7** | ✅ Completed | Qwen3-1.7B-Base | MINDlarge | Base model (non-instruct) | 0.6946 | 0.3373 | 0.3767 | 0.4392 | No chat template; [W&B](https://wandb.ai/wuchen/MIND/runs/i9h0gzwr) |
| **L1.8** | 🔄 Running | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1, EP=7, micro_bs=2 | - | - | - | - | Previous run (0.6622, W&B p03ej16f) was misconfigured (wrong params/UAI missing); fixed UAI + resubmitted 2026-06-08; run ID: neat_cassava_hxz75hmjyk |
| **L1.9** | ✅ Completed | Qwen3-1.7B-Base | MINDlarge | USE_ABSTRACT=1, Base model | 0.6880 | 0.3368 | 0.3769 | 0.4382 | Abstract didn't boost base; [W&B](https://wandb.ai/wuchen/huggingface/runs/gzijusub) |
| **L1.3x** | ✅ Completed | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1, EP=5, NEG=2.0, gckpt (pipeline re-run) | 0.6622 | - | - | - | AML-pipeline reproduction of L1.3 setting (bold_foot_75cv39dfth); **−0.0427 vs L1.3 0.7049** — gradient-checkpointing/pipeline run diverges from original; original L1.3 remains the reference |
| **L1.13** | ❌ Failed | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1, EP=7, NEG=3.0 | 0.6647 | 0.3166 | 0.3493 | 0.4116 | **−0.0402 vs L1.3**; combining EP=7+NEG=3.0+abstract collapses; NEG=3.0 consistently toxic on large |

---

### Phase 1E: Abstract Train → Title-Only Eval (Cross-Condition)

> **Result: Hypothesis DISPROVED for MINDlarge.** Abstract helps at BOTH train and eval time.
> E1.1 shows title-only eval on an abstract-trained model (0.6945) is WORSE than abstract eval (0.7049).
> Abstract prompts provide richer signal at eval time too — keep `USE_ABSTRACT=1` for both train and eval.

| Exp ID | Status | Source | Eval Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|--------|-------------|-----|-----|--------|---------|-------|
| **E1.1** | ✅ Completed | L1.3 (abstract-trained) | eval USE_ABSTRACT=0 | 0.6945 | 0.3364 | 0.3732 | 0.4375 | **WORSE** than L1.3 0.7049; hypothesis disproved — abstract needed at eval too |
| **E1.2** | ✅ Completed | L1.3 (abstract-trained) | eval USE_ABSTRACT=1 | **0.7051** | 0.3463 | 0.3849 | 0.4481 | Confirms pipeline fix: matches L1.3 original (0.7049 → 0.7051, negligible diff) |
| **E1.3** | ❌ Cancelled | L1.9 (base, abstract-trained) | eval USE_ABSTRACT=0 | - | - | - | - | Cancelled: E1.1 disproved the hypothesis; no need to test on L1.9 |
| **E1.4** | ❌ Cancelled | L1.6 (4B, abstract-trained) | eval USE_ABSTRACT=0 | - | - | - | - | Cancelled: abstract must stay at eval time |

**Conclusion: Always use `USE_ABSTRACT=1` at both train and eval for abstract-trained models on MINDlarge.**

---

---

### Phase 6: Advanced Strategies (Beyond Scaling)

> **Motivation**: Scaling model size is closed (4B < 1.7B on MINDsmall). These four directions target the remaining ~0.012 AUC gap to SOTA through data quality, training curriculum, richer features, and collaborative filtering signals.

#### 6A: Hard Negative Mining

| Exp ID | Status | Model | Config | AUC | Notes |
|--------|--------|-------|--------|-----|-------|
| **H1.1** | ✅ Completed | Qwen3-1.7B | 100% same-category negatives | 0.6622 | Completed 2026-05-16; AUC 0.6622 < L1.3 0.7049; hard negatives alone not sufficient; abstract likely needed |
| **H1.2** | ⬜ Pending | Qwen3-1.7B | Model-mined hard negatives | - | Use L1.3 predictions to find borderline negatives (near-zero score) |

**Why**: Current sampling is 50% same-category / 50% random. With 100% same-category negatives the model can't rely on coarse category matching and must learn finer title/abstract-level signals.

**Command (H1.1)**:
```powershell
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_h1-1_hard_neg `
  --display-name "H1.1: Hard negatives 100% same-category (large)" `
  --model-path Qwen/Qwen3-1.7B `
  --data-root "shares/users/wuc/data/MIND_large" `
  --output-root "shares/users/wuc/output_dir" `
  --batch-size 256 `
  --micro-batch-size 4 `
  --num-epochs 5 `
  --neg-ratio 2.0 `
  --hard-neg-ratio 1.0 `
  --max-history 30 `
  --use-chat-template 1 `
  --use-abstract 1 `
  --run-eval 1 `
  --eval-split dev
```

#### 6B: Curriculum Learning (MINDsmall → MINDlarge)

| Exp ID | Status | Model | Config | AUC | Notes |
|--------|--------|-------|--------|-----|-------|
| **C1.1** | 🔄 Running | Qwen3-1.7B | SFT MINDsmall (ep5) → resume on MINDlarge | - | Failed twice (2026-05-14, 2026-05-15) due to missing UAI for datastore mount; fixed & resubmitted 2026-06-08; run ID: affable_board_ndq03p3rb3 |
| **C1.2** | ✅ Completed | Qwen3-1.7B | SFT MINDsmall abstract → resume on MINDlarge abstract | 0.6951 | AUC 0.6951 < L1.3 0.7049; curriculum abstract hurt; 4d 7h runtime; checkpoint-13600 |

**Why**: MINDsmall teaches basic recommendation patterns faster (fewer impressions, tighter feedback loop). MINDlarge then adapts the model to a larger vocabulary and distribution. Direct MINDlarge training may converge to a suboptimal solution that curriculum avoids.

**Command (C1.1)**:
```powershell
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_c1-1_curriculum `
  --display-name "C1.1: Curriculum MINDsmall→MINDlarge" `
  --model-path Qwen/Qwen3-1.7B `
  --data-root "shares/users/wuc/data/MIND_large" `
  --output-root "shares/users/wuc/output_dir" `
  --batch-size 256 `
  --micro-batch-size 4 `
  --num-epochs 5 `
  --neg-ratio 2.0 `
  --max-history 30 `
  --use-chat-template 1 `
  --use-abstract 0 `
  --run-eval 1 `
  --eval-split dev `
  --resume-from-checkpoint "shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_chat/final_checkpoint"
```

**Command (C1.2)**:
```powershell
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_c1-2_curriculum_abstract `
  --display-name "C1.2: Curriculum abstract MINDsmall→MINDlarge" `
  --model-path Qwen/Qwen3-1.7B `
  --data-root "shares/users/wuc/data/MIND_large" `
  --output-root "shares/users/wuc/output_dir" `
  --batch-size 256 `
  --micro-batch-size 4 `
  --num-epochs 5 `
  --neg-ratio 2.0 `
  --max-history 30 `
  --use-chat-template 1 `
  --use-abstract 1 `
  --run-eval 1 `
  --eval-split dev `
  --resume-from-checkpoint "shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint"
```

#### 6C: Richer Input Features (Abstract + Subcategory)

| Exp ID | Status | Model | Config | AUC | Notes |
|--------|--------|-------|--------|-----|-------|
| **F1.1** | ❌ Abandoned | Qwen3-1.7B | USE_ABSTRACT=1 + USE_SUBCATEGORY=1 | 0.6962 (0.78ep) | AUC 0.6962 at 0.78ep; subcategory consistently hurts (P1.7 also worse); prompts too long → extreme slowdown; not worth full training |
| **F1.2** | ⬜ Pending | Qwen3-1.7B | Profile summary prepended to prompt | - | Offline generate user interest profile with Qwen3-4B-Instruct, prepend at eval |

**Why**: L1.3 uses abstract but no subcategory. P1.7 used subcategory but no abstract and got 0.6767 (worse than P1.0's 0.6861). Combined signal hasn't been tested on MINDlarge.

**Command (F1.1)**:
```powershell
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_f1-1_abstract_subcat `
  --display-name "F1.1: Abstract + Subcategory (large)" `
  --model-path Qwen/Qwen3-1.7B `
  --data-root "shares/users/wuc/data/MIND_large" `
  --output-root "shares/users/wuc/output_dir" `
  --batch-size 256 `
  --micro-batch-size 4 `
  --num-epochs 5 `
  --neg-ratio 2.0 `
  --max-history 30 `
  --use-chat-template 1 `
  --use-abstract 1 `
  --use-subcategory 1 `
  --run-eval 1 `
  --eval-split dev
```

#### 6D: Collaborative Filtering Signal

| Exp ID | Status | Approach | AUC | Notes |
|--------|--------|----------|-----|-------|
| **CF1.1** | 🔄 Running | CF score ensemble (item-based CF + L1.3) | 0.7051 (α=0.3) | alpha=0.3: AUC 0.7051 = same as L1.3 baseline (no gain); try α=0.1, 0.2, 0.5 |
| **CF1.2** | ⬜ Pending | CF-augmented prompt | - | Prepend "Users with similar reading history also clicked: X, Y, Z" to LLM prompt |

**Why**: LLM scores purely from text content. CF captures user-item interaction patterns (users who read A also read B) that are invisible to content-only models. This is a fundamentally different signal — ideal ensemble partner.

**CF1.1 Commands**:

Step 1 — Compute CF scores (CPU-only, run on compute node or locally):
```bash
python src/compute_cf_scores.py \
  --train_behaviors $DATA_ROOT/train/behaviors.tsv \
  --eval_behaviors  $DATA_ROOT/dev/behaviors.tsv \
  --output          $DATA_ROOT/dev/cf_scores.tsv \
  --max_history 30
```

Step 2 — Eval with CF blend (add to existing eval command, sweep alpha):
```bash
# alpha=0.1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$DATA_ROOT \
  USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$EVAL_DIR/dev_cf_alpha01.txt \
  bash scripts/eval_mind_pointwise.sh $MODEL_PATH dev \
  --cf_scores_file $DATA_ROOT/dev/cf_scores.tsv --cf_alpha 0.1

# alpha=0.2
# alpha=0.3  (default, recommended starting point)
# alpha=0.4
# alpha=0.5
```

Or via AML eval pipeline with extra args passed through `--debug` + manual run.

**Note**: `src/compute_cf_scores.py` and `--cf_scores_file` / `--cf_alpha` args are already implemented in `evaluate_mind_pointwise.py`. No training required — pure inference-time blend.

---

### Phase 4M: Multi-task (Extra Checkpoints)

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **M4.0** | ❌ Failed | Qwen3-1.7B | MINDsmall | POINTWISE_RATIO=0.7, ep3 | 0.6213 | 0.2833 | 0.3124 | 0.3765 | Multitask hurts badly; [W&B](https://wandb.ai/wuchen/MIND/runs/5nb9n11h) |
| **M4.1** | 🔄 Running | Qwen3-1.7B | MINDlarge | POINTWISE_RATIO=0.5, EP=5, Abstract | - | - | - | - | Proper test: large+abstract+EP5+multitask pw. History: shy_fowl env-crash→great_roof OOM (ranking-branch loss.backward)→fixed micro_bs=2→khaki_wall_zgqz3nvhk5 RUNNING |

### Phase 1Lx: Qwen3.5-2B architecture reproduction (2026-09-29)

> **Low-hanging-fruit direction**: reproduce the winning L1.3-style setting (MINDlarge abstract pointwise)
> but swap the 1.7B model for the new Qwen3.5-2B (linear-attention / chunk_gated_delta_rule backbone).
> NOTE: Qwen3.5-2B is far more VRAM-hungry on long abstract sequences — full-dev eval at batch 8 OOMs
> at qwen3_5.py:375 (`torch_chunk_gated_delta_rule`). Training checkpoints stay intact; standalone eval
> must use `--batch-size 4` (user-directed permanent fix, in config_mind_train.json).
>
> **r7 eval crash (2026-09-29)**: even bs4 eval over full large dev crashed ~45 impressions in with the same
> `illegal memory access` at qwen3_5.py:375 (both chat `neat_band` and Base `happy_energy`; training COMPLETED,
> `final_checkpoint` saved). Re-submitted standalone eval at **bs1** (safest): chat `sharp_napa_myhlzwqvh3`,
> Base `maroon_brush_xvk6c59278`.
>
> **Qwen3.5 eval sdpa fix (2026-10-06, commit 3131435)**: `evaluate_mind_pointwise.py` detected Qwen3.5
> via `model_type=="qwen3_5"`, which FAILED for these checkpoints → fell back to Flash Attention 2 → illegal
> memory access crash on the GatedDeltaNet hybrid backbone. Fixed to sniff the `architectures` string
> (`ForConditionalGeneration`/`Qwen3_5`) exactly like the trainer (sft_mind_pointwise_ds.py:180), forcing sdpa.
> Also fixed `config_mind_train.json` eval glob `*$(basename model)*` selecting the wrong sibling checkpoint
> (chat vs Base both live under the shared output_root) → now builds the exact OUTPUT_NAME deterministically.
> Verified working: dev@5000 eval of r11b chat completed cleanly on the ranking VC (`olive_steelpan_spqkjfn91t`).
> **Gotcha**: the standalone eval component `run_eval_pipeline.py` is pinned to the **recall** VC (node-starved /
> `Not enough Standard tier quota`); submit eval on the **ranking** VC (like training) instead.
>
> **r7 eval v4 fix (2026-10-03)**: bs1 pointwise evals `brave_rhythm_grkhnk42yf`/`goofy_beet_q8cg4r2mr0`
> STILL OOM'd (`illegal memory access` @ qwen3_5.py:375 `torch_chunk_gated_delta_rule` `torch.zeros(...)`) —
> root cause was `config_mind_eval.json` lacking `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (only the
> JEV eval config had it). Added it to the pointwise/ranking eval command (commit 7c52269), pushed. Resubmitted
> with the fix live: chat → `loving_fig_d6sw5hr69d`, Base → `nifty_rabbit_8d4hr31sxd` (both Standard tier, bs1).

| Exp ID | Status | Model | Dataset | Config | AUC | Notes |
|--------|--------|-------|---------|--------|-----|-------|
| **r7** | ✅ Trained / 🔄 Eval bs1 (crashed at bs4) | Qwen3.5-2B-Base | MINDlarge | abstract, gckpt | - | training COMPLETED → eval `maroon_brush_xvk6c59278` (bs1) resubmitted; bs4 eval `stoic_clock_ftbzl0cqyj` failed |
| **r7** | ✅ Trained / 🔄 Eval bs1 (crashed at bs4) | Qwen3.5-2B (chat) | MINDlarge | abstract, gckpt | - | training COMPLETED → eval `sharp_napa_myhlzwqvh3` (bs1) resubmitted; bs4 eval `good_fennel_5pm69xdzq3` failed |
| **r8** | 🔄 Submitted | Qwen3.5-2B (chat) | MINDlarge | abstract, **EP=7** | - | L1.2-repro (EP7) on 2B; tests if undertrained 2B gains from more epochs; run `sleepy_pasta_g9hytyykq9` |
| **r9** | 🔄 Submitted | Qwen3.5-2B (chat) | MINDlarge | abstract, **HIST=50** | - | 2B linear-attention backbone natively handles long history — HIST may help where it hurt 1.7B; run `olden_drain_3b6psyybwj` → stale-run reuse (FAILED) → `joyful_juice_zh8gm2z86r` |
| **r10** | 🔄 Submitted | Qwen3.5-2B (chat) | MINDlarge | abstract, **EP=5** (L1.3 winner repro) | - | **Exact L1.3-winner setting (0.7049) on 2B** — the true prior-best repro; run `upbeat_leaf_43yqnzr0h8` |
| **r11b** | ✅ Completed | Qwen3.5-2B (chat) | MINDlarge | abstract, ckpt4096, EP=5, ckpt4096, s300k (30万样本上限) | **0.6542** dev@5000 | `serene_rice_88lx58jm5j` trained + checkpoint saved; **dev@5000 eval `olive_steelpan_spqkjfn91t` (ranking-VC) COMPLETED: AUC 0.6542**, MRR 0.3041, nDCG@5 0.3290, nDCG@10 0.3952. Note: **s300k = only 30万 samples** (feasible 5860-step run) — VALID but on a reduced train set, so not directly comparable to full-L1.3-repro R10 (no eval fix there yet). |
| **r11b** | ✅ Completed | Qwen3.5-2B (Base) | MINDlarge | abstract, ckpt4096, EP=5, s300k | **0.6272** dev@5000 | **Base eval COMPLETED: AUC 0.6272** dev@5000 (recomputed from pred file keyed by impression ID; 5000/5000 matched). Chat (0.6542) > Base (0.6272) — chat-template training helps 2B. Both positive; Qwen3.5-2B is a working L1.x reproduction. |

### Phase 1P: Qwen3.5-2B P-series rerun (MIND_small — full data, no s300k cap)

> User directive: for all historical Qwen3-1.7B experiments, swap the model to **Qwen3.5-2B** and
> rerun with *identical* settings — research FIRST on full **MIND_small** (`--train-sample 0`),
> promote winners to MIND_large after validation. Full MIND_small dev = 73,152 impressions (AUC
> computed from ADLS pred files via `calc_mind_metrics.py`, impression-ID keyed, 73,152/73,152 matched).

| Exp | Model | Dataset | Config | AUC | vs 1.7B | Notes |
|-----|-------|---------|--------|-----|---------|-------|
| **P1.0** | Qwen3.5-2B | MINDsmall | ep5 default | **0.6360** | 1.7B 0.6861 (−0.0501) | `affable_hamster_dfksgqqfjm` COMPLETED. 2B clearly underperforms 1.7B on full small. |
| **P1.1** | Qwen3.5-2B | MINDsmall | neg3.0 | **0.6287** | 1.7B 0.6985 (−0.0698) | `dreamy_monkey_rb7j50sqc3` COMPLETED. |
| **P1.5** | Qwen3.5-2B | MINDsmall | neg3 ep7 hist50 | **0.6189** | 1.7B 0.6804 (−0.0615) | `neat_potato_7myz9f6k4w` COMPLETED. |
| **P1.2** | Qwen3.5-2B | MINDsmall | ep7 | **0.6424** | 1.7B 0.6886 (−0.0462) | `olden_fennel_zvl1kjpwnz` COMPLETED. ep7 = 2B's best alongside hist50 — supports undertraining theory. |
| **P1.3** | Qwen3.5-2B | MINDsmall | abstract ckpt4096 | **0.6385** | 1.7B 0.6662 (−0.0277) | `keen_drain_9d0g467ccp` COMPLETED. |
| **P1.4** | Qwen3.5-2B | MINDsmall | hist50 | **0.6429** | 1.7B 0.6886 (−0.0457) | `serene_station_c7g9br7ph1` COMPLETED. |
| **P1.7** | Qwen3.5-2B | MINDsmall | subcategory | **0.6253** | 1.7B 0.6767 (−0.0514) | `gentle_root_d1c59cx392` COMPLETED. |

> **Read (2026-10-09, all 7 completed)**: Qwen3.5-2B is consistently **~0.05–0.07 AUC below**
> Qwen3-1.7B on every P-series config → the ~0.05–0.07 deficit is confirmed accurate & reproduced.
> Best 2B configs are **ep7 (P1.2 0.6424) and hist50 (P1.4 0.6429)** — both add more training signal,
> consistent with the "linear-attention 2B is undertrained at 5 epochs" hypothesis; stacking everything
> (P1.5) or subcategory (P1.7) hurts. No 2B config closes the gap to 1.7B. Decision point: the 2B
> L-series on MIND_large will likely show the same deficit — whether to keep scaling the 2B path or
> fall back to 1.7B as the primary ranker.

### Phase 1S: Sub-2B rivals P-series sweep (dev-AUC results — task #23/#24)

> COMPARISON (2026-10-11, all dev evals complete): evaluate whether *other* sub-2B base models
> (Gemma-2-2B, Qwen2.5-1.5B, Qwen3-1.7B) on the same P-series MIND_small recipes beat our Qwen3.5-2B.
> Dev AUC collected via `pipeline/collect_eval_aucs.py`.

| Model | P1.2 (ep7) | P1.4 (hist50) | P1.5 | P1.7 (subcat) |
|-------|-----------|--------------|------|---------------|
| **Qwen3.5-2B** (ours, ref) | 0.6424 | 0.6429 | 0.6189 | 0.6253 |
| **Qwen3-1.7B** | ~0.68 (hist) | 0.6612 | 0.6705 | 0.6582 |
| **Gemma-2-2B** | 0.6420 | 0.6504 | 0.6387 | 0.6314 |

> **Read**: Qwen3-1.7B remains the strongest sub-2B (P1.5 0.6705, P1.7 0.6582 > any 2B). Gemma-2-2B
> (best P1.4 0.6504) is competitive with our Qwen3.5-2B and is a cheap **ensemble diversity** member.
> Qwen3.5-2B's P1.7/p1.5 still lag; the 2B linear-attention deficit vs 1.7B holds across families.
> Ensemble: combining Qwen3-1.7B + Gemma-2-2B + Qwen3.5-2B diversity is the highest-certainty path.

### Phase 1L: Qwen3.5-2B L-series rerun (MIND_small — research phase)

> RESEARCH-FIRST: rerun the historical Qwen3-1.7B L1.x sweep (incl. F1.1/H1.1) on Qwen3.5-2B with
> identical settings on full MIND_small (`train_sample=0`); promote winners to MIND_large after.
> Baseline (L1.3) = ep5, neg2.0, hist30, abstract, chat. Model `Qwen/Qwen3.5-2B` (L1.7/L1.9 = `-Base`,
> no chat). Submitted via `pipeline/submit_2b_l1x_lseries.py`.

| Exp | Config | 2B AUC | vs 1.7B | Notes |
|-----|--------|--------|---------|-------|
| **L1.1** | neg3.0 | **0.6287** | 0.6883 (−0.0596) | `polite_vulture_6qxy937lfx` COMPLETED. |
| **L1.2** | ep7 | **0.6424** | 0.6932 (−0.0508) | `sleepy_cassava_4qtqckylh0` COMPLETED. |
| **L1.3** | abstract (baseline) | *(running)* | 0.7049 | `gentle_soca_k0yplhlqdz` RUNNING |
| **L1.4** | hist50 | **0.6429** | 0.6863 (−0.0434) | `placid_nut_mvf48tgbm3` COMPLETED. |
| **L1.5** | neg3 ep7 hist50 | **0.6189** | 0.6804 (−0.0615) | `lemon_pasta_t8ct4b0nkx` COMPLETED. |
| **L1.7** | Base, no chat | *(running)* | 0.6946 | `cool_bottle_9hwv3v9tl4` RUNNING |
| **L1.9** | Base abstract | *(running)* | 0.6880 | `nice_peach_wzg73q6dl8` RUNNING |
| **L1.13** | abstract ep7 neg3 | *(running)* | 0.6647 | `sweet_cassava_33crdfhp1d` RUNNING |
| **F1.1** | abstract+subcat | *(running)* | 0.6962 (0.78ep) | `blue_plum_21l227nncx` RUNNING |
| **H1.1** | hard-neg 100% | *(rerunning)* | 0.6622 | Original `goofy_box_nghcbmkp1z` polluted by eval-name collision w/ P1.0 → rerun `salmon_squash_t0b7yt5zls` with `_hardneg` naming fix. |

> **Interim (2026-10-09, 4 complete)**: L1.1/L1.2/L1.4/L1.5 match the P-series AUCs exactly (same
> settings → same numbers → pipeline reproducible). Same ~0.05–0.06 deficit vs 1.7B. **L1.4 hist50
> 0.6429 and L1.2 ep7 0.6424 are best so far** — same direction as P-series. H1.1 excluded pending
> clean rerun (collision bug).

> **Bug found & fixed (2026-10-09)**: `OUTPUT_NAME` in `scripts/sft_mind_pointwise_ds.sh` and
> `pipeline/components/mind_train/config_mind_train.json` **did not include `HARD_NEG_RATIO`**, so
> H1.1 (hard_neg=1.0, neg2.0) collided with P1.0 (hard_neg=0, neg2.0) at
> `..._bs256_ep5_neg2.0_hist30_chat`. Added `_hardneg` suffix for non-default (≠0/≠0.5) hard-neg
> configs — future hard-neg runs get a distinct dir. No existing default dirs rename; only future
> non-default hard-neg runs are affected.

### Secondary Track: OpenJev (JEV) decision-model eval (2026-09-29)

> **JEV track**: use OpenJev as a "no-UL" decision model scoring each candidate's P(yes) on 5000-dev impressions
> (wheat_plastic_ryxqfpg987, `jev-dev-batch5k-ok`). Predictions at `shares/users/wuc/models/jev_eval_results/dev_jev_predictions.txt`.

> **Qwen (stock, frozen) baseline**: NOT yet evaluated on the same noul dev@5000 contract.
> Every Qwen row in this doc is a *trained* model. The only frozen/untrained Qwen numbers live
> in the zero-shot baseline table (lines 62-66: Qwen3-1.7B 56.41%, Qwen3-4B-Instruct 58.11%),
> but those use the generic pointwise prompt on MINDsmall — **not** the noul A/B opener — so they
> are NOT directly comparable to the JEV stock reference 0.6432. TODO: run a frozen Qwen3.5-2B
> noul eval on dev@5000 for an apples-to-apples stock comparison.

| Exp | Model | Split | Max impressions | AUC | Notes |
|-----|-------|-------|-----------------|-----|-------|
| **JEV** | APUS-OpenJev-v1-4B | dev@5000 | 5000 | 0.6432 | Pointwise AUC recomputed from predictions; secondary track reference |
| **JEV-9B** | APUS-OpenJev-v1-9B | dev@5000 | 5000 | **0.6329** | neat_stone_3427s7bjrr → OOM → FIXED (chunk batch 32 + `expandable_segments`) → `tender_carrot_sbxpx7q8xv` COMPLETED: AUC **0.6329**, MRR 0.3037, nDCG@5 0.3306, nDCG@10 0.3920. **9B slightly WORSE than 4B (0.6432)** — size alone doesn't help on noul. |

> **JEV-9B OOM (2026-09-30)**: `evaluate_mind_jev.py:score_yes_batch` padded the entire impression
> into one `use_cache=False` forward; the 9B delta-rule linear attention materialized `k_cumdecay =
> attn @ (k_beta*g.exp())` as a ~3.62 GiB workspace with only 2.73 GiB free (model held 76.5 GiB).
> Fixed by scoring in 32-row chunks + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (error's own
> suggestion, reclaims the 17 GiB reserved-but-unallocated fragmentation). Pushed to dev `eddf1a1`.

> **r7-ev2 eval `NoIdentityOnCompute` (2026-09-30)**: both sharp_napa + maroon_brush failed at eval_node
> with Singluarity `NoIdentityOnCompute` (nodes never ran python — infra auth, NOT OOM). Resub at bs1:
> chat `upbeat_nerve_xhyfgsv0sz`, Base `yellow_yak_xh87jbrdmg`.

### Secondary Track: JEV-4B training campaign (2026-09-30)

> Train APUS-OpenJev-v1-4B on MIND across 4 paradigms (in dependency order) to make
> a news-specialized decision model. Historical train/eval use the **legacy noul
> A/B prompt** (`src/jev_decision.py:render_noul_prompt`); candidate-normalized
> P(A)=P(yes) is the rank score. The recorded stock JEV reference is 0.6432;
> matched prompt, preprocessing and impression subsets must be verified before
> attributing differences to training.
> Baseline config: MINDlarge abstract, hist30, neg1.0, cutoff4096, DS ZeRO-2 (mini env),
> sdpa, micro-batch 2, lr 1e-4 (JEV-published). Eval via `run_jev_eval.py` (noul scorer).

| Campaign | Paradigm | Run ID | Mode | KL/term | AUC(dev) | Notes |
|----------|----------|--------|------|---------|----------|-------|
| C1 | SFT + within-model distill | `affable_fork_57j81ydc2b` | FAILED | kl_beta=0.1 R-Drop | - | `num_items_in_batch` signature crash (transformers 5.x) |
| C1 | SFT + within-model distill | `frank_chicken_fct2hw878v` | CANCELLED | kl_beta=0.1 R-Drop | - | zero-loss answer-index bug found during C2 offline smoke; model would learn nothing |
| C1c | SFT (plain CE, no KL) | `lemon_ship_f2nfshcfh9` | FAILED | enable_kl=0 | - | fresh run (force_rerun); still diverges → see C1fix NB |
| C1fix | SFT plain CE, no distill | `sincere_garlic_7t99m7nxw6` | ⚠️ **0.4959** | enable_kl=0 EP2 small | **0.4959** | **Confirmed: model NEVER learned.** The `dev_jev_predictions.txt` (5000 MIND_small dev impressions, recovered from datalake + rescored offline) gives **AUC 0.4959 ≈ chance** — because the NaN in-band eval triggered EarlyStopping at 0.14ep. Train loss 12-150 is normal full-vocab CE (NOT divergence); the NaN eval was the only real fault. |
| C1clean | SFT plain CE (no KL, no in-band eval) | `gentle_receipt_fhmt5lttvj` (and retries `quirky_corn_yl0crrlz4b`, `olive_pin_sbgmwq2gqx`) | ⚠️ NaN@step0 | enable_kl=0 EP2 small | - | **Second killer found: the manual CE path.** `compute_loss` did `labels=None` forward → materialized FULL (B,L,V) logits → sliced answer row → manual CE. That NaNs on the very FIRST `training_step` (all 8 ranks `FloatingPointError`, before any backward) on every retry — even with the CE cast to fp32 (so it's the forward logits traveling NaN, not the CE). |
| C1clean2 | SFT plain CE via STOCK HF loss | `lemon_frog_d217q5r20p` | Training and evaluation worker COMPLETED | enable_kl=0 MIND_small EP3 | **0.5452** | **Fix (commit 6c6bbd6): use `outputs.loss` (stock HF next-token CE) instead of the manual full-logits path.** Training completed (11001 steps); final checkpoint saved. Recovery eval `calm_stem_v4mpzf0hs8`, child `c3918e4d-2c9f-4503-a50d-14d3612d7d48`, scored 5,000 MIND_small dev impressions on ranking Standard / one A100 at 2026-10-19 08:48 UTC. Saved-rank AUC independently recomputed as 0.54524778; all 5,000 impression IDs and candidate sets match the input sequence. Below recorded frozen JEV 0.6432 by 0.0980; preprocessing alignment must still be confirmed before causal attribution. Original full-dev eval canceled. |

> **C1clean2 completed scoring (2026-10-19 08:48 UTC):** evaluator-reported
> AUC **0.5452**, MRR **0.2328**, nDCG@5 **0.2427**, nDCG@10 **0.3122**.
> Independent reconstruction from saved ranked news IDs gives AUC **0.54524778**
> (agrees to four decimals), MRR **0.23427276**, nDCG@5 **0.24434410**, and
> nDCG@10 **0.31373802**. Do not silently substitute reconstructed MRR/nDCG for
> the scorer's metrics: raw scores and ties were not saved, and Python's stable
> prediction ordering differs from NumPy's metric tie ordering. This is a
> reproducibility limitation, not a separately evaluated model improvement.
> Prediction SHA256: `53c9ee052c6eacc61bc665dfa8d05869ee280af984955c5c759907f4f40e9ca9`.
> A clean full-budget run now has a real result, but has not beaten the recorded
> frozen reference. C2/C3/C4 remain gated; diagnose scoring/training alignment
> and capture raw scores in future evaluations before promoting this checkpoint.

> **Earlier C1clean2 triage (2026-10-19, before recovery):** Training is **COMPLETED**.
> The eval parent is **RUNNING**, but its child is **Queued** on recall / Standard /
> `Singularity.ND96amrs_A100_v4`. It did start: the application log reached 150
> impressions before SIGTERM at 03:02 UTC. The scheduler subsequently recorded
> eviction by a higher SLA tier (03:06 UTC), insufficient Standard quota for the
> eight-unit allocation, a rejected quota reservation, and another deployment/requeue.
> This is an infrastructure interruption, not evidence of a model/evaluator failure.
> Predictions are written only after the scoring loop completes, so the existing
> output directory is still empty. The actual path is
> `shares/users/wuc/output_dir/sft_jev4b_small_bs128_ep3_neg1.0_hist30_kl0.1_abstract/jev_eval_results/dev_jev_predictions.txt`
> (**beside** `final_checkpoint`, not inside it).
> The submitted eval uses `max_impressions=0` (full MIND_small dev); recompute the
> first-5000-impression metric separately before comparing with stock JEV's 0.6432.
> Initial triage left the original job queued; the recovery below supersedes it.
> C2/C3/C4 remain gated on a completed evaluation.

> **C1clean2 recovery (2026-10-19):** Live VC quota inspection using ARM API
> `2021-03-01-preview` showed recall Standard at 32/33 A100 units versus ranking
> Standard at 9/908. The old submitter's `instance_count=1` meant one **eight-GPU
> ND96 node**, not one GPU, although the scorer uses only GPU 0.
> Added `--virtual-cluster` and `--instance-type` while preserving existing defaults.
> Submitted `calm_stem_v4mpzf0hs8` with `Singularity.ND12am_A100_v4` on ranking /
> Standard and `max_impressions=5000`; verified the actual child allocation.
> The superseded `happy_ice_17qv9vphsr` and its worker are confirmed **Canceled**.
> Fresh execution is set on the pipeline (`job.settings.force_rerun=True`):
> the previous `node.settings.force_rerun` field was ignored by AML.
> The recovery changes only submission resources and the impression limit, not
> checkpoint weights, prompt, history, abstracts, or scorer.
> Worker **Running** confirmed at 04:28 UTC; the runtime log shows environment
> setup (torch installation), not completed scoring. Follow-up is scheduled every
> 15 minutes in the active CLI session to inspect logs and publish verified status/results.
> Setup subsequently logged the pre-existing optional `requirements.txt` / `pyarrow<19`
> resolution conflict and a failed optional causal-conv1d build, but continued through
> the separate critical-dependency installs to flash-attn installation. Do not confuse
> these warnings with a terminal eval failure; actual scoring still needs verification.
> At 04:34 UTC, setup completed, 42,416 news articles and all checkpoint weights
> loaded, and the scorer entered the evaluation loop. At 04:35 UTC the counter
> reached **1 completed impression**, verifying actual inference on the single A100.
> Missing causal-conv1d selects the correct but slower reference implementation.
> **Webpage correction (2026-10-19):** the actual live NewsRank page is on the
> AAD-protected `chrona-showcase-internal` Static Web App, not the deprecated Blob
> host. Earlier Blob feed uploads did not update that page. The progress publisher
> now stages JSON into the canonical shared-site master and deploys a complete
> snapshot, preserving other pages, navigation, and authentication configuration.

> **C1 autopsy (2026-10-16)**: BOTH C1 "collapses" share ONE root cause — the **in-band HF eval loop**.
> 1. **Train loss on full-vocab CE is legitimately O(50-150)** (JEV vocab ~150k, CE over next-token of
>    the run), NOT a 2-class ~0.7 — reading it as "divergence" was a red herring. Weights stay finite.
> 2. **The in-band eval (`eval_strategy="steps"` + `metric_for_best_model=eval_loss` +
>    `EarlyStoppingCallback`) emits `eval_loss=NaN` from the very first eval** (epoch 0.034) and HF
>    EarlyStopping kills the run at ~0.14ep — before the model learns anything. That's why both C1
>    (R-Drop) and C1fix (plain CE) came out at chance (0.4975 / **0.4959**). The R-Drop KL was *not*
>    the cause.
> 3. **Fix (implemented in `src/sft_mind_jev.py`)**: drop the in-band HF eval entirely
>    (`eval_strategy="no"`, `load_best_model_at_end=False`, no EarlyStopping, no eval_dataset). Dev AUC
>    is produced only by the external `evaluate_mind_jev.py` scorer after training — matching every
>    other MiniOneRec pipeline. Also added a NaN `FloatingPointError` guard in `compute_loss` so a real
>    divergence halts loudly instead of silently poisoning state. The AML component does
>    `git clone && checkout dev`, so **the fix must be pushed before resubmitting**.
> 4. C1fix WAS externally scored (recovered `dev_jev_predictions.txt`, 5000 MIND_small dev impressions,
>    rescored offline -> AUC **0.4959**). This confirms the eval-path diagnosis but the checkpoint is
>    useless (killed at 0.14ep). **Next: push the fix and resubmit C1-clean (plain CE, no in-band eval),
>    then external-eval for a real dev AUC.**
| C2 | RLCD | pending | DPO-style | contrastive | pending | `sft_mind_jev_rlcd.py` + `jev_noul_pair_data.py`: ground-truth (clicked>unclicked) pairs, self-distill P(A) margin; smoke-first |
| C3 | RL (GRPO) | pending | verl | pointwise_binary | pending | on C1 ckpt via `rl_mind_verl.py` |
| C4 | Teacher OPD | pending | KL-to-strong-teacher | frozen L1.3 1.7B | pending | `sft_mind_jev_opd.py` |

#### Approved follow-up: candidate-only click decision ablations (2026-10-20)

**Implemented; GPU smoke follow-up below, no new research AUC yet.** Historical
defaults remain `full_vocab` + `legacy`. The new full-depth loss is
`BCEWithLogits(z_A - z_B, click_label)`, equivalent to two-class CE over A/B,
not CE over the whole vocabulary. It forwards only the prompt, requests one
logit position, and computes the loss in FP32. A/B remain output coordinates;
literal Yes/No tokens are not substituted.

`native_click` asks whether the user would click the displayed candidate and uses
the canonical Noul true/false candidate descriptions. Its rendered prompt was
compared byte-for-byte with the released upstream contract at model revision
`f0b880feb751a789b4c6a062d03fbbc23d384991`. That revision's actual tokenizer
returned A=32 and B=33 and passed joint prompt-plus-answer boundary checks for
both modes. This verifies the published contract, **not** byte identity with the
staged AML snapshot. The repository still uses Transformers 5.2.0 rather than
the released OpenJet executor's 5.16.1 pin.

Published JEV trains low/high exits with candidate-space losses plus detached
deep-to-shallow KL. This implementation is a **full-depth ablation**, not that
two-exit reproduction. Earlier two-pass R-Drop must not be described as native
distillation. Decision mode rejects `--enable-kl 1`; no KL term is implemented.
Neither the old prompt nor the loss has been established as the cause of 0.5452.

| Control | Loss | Prompt | Purpose |
|---|---|---|---|
| C1clean2 | `full_vocab` | `legacy` | Completed historical baseline, AUC 0.5452 |
| C1 decision-only | `decision` | `legacy` | Isolate candidate normalization |
| C1 native-click | `decision` | `native_click` | Compare prompt semantics under the same loss |
| Matched frozen controls | None | Both modes | Separate prompting from weight updates |

Keep split, seed, sampled examples, history, abstracts, cutoff and dev@5000
identical. Native mode rejects overlong prompts instead of silently cropping:
training validates every sampled prompt before its first optimization step,
with prompt budget `min(cutoff_len - 1, 8192)`; evaluation admits at most 8192.
Legacy retains historical training truncation. Before causal comparison, audit
real-data lengths and select a shared budget/history policy that admits all
compared examples without truncation. Do not silently drop failed examples.

The default allocation remains one eight-A100 node; the smoke examples explicitly
request one A100 and a 3600-second command timeout. The user approved publishing
and executing this bounded smoke on 2026-10-20. `--source-revision` now pins the
worker checkout to a full commit SHA (default: local HEAD); publish that commit
before submitting. `--code-revision` remains only a cache-buster. Workers clone
inside the job's private code directory, without deleting a shared repository.
If GitHub write permission is unavailable, `--bundle-source` uploads committed
local HEAD as an incremental Git bundle inside the AML component. The worker
fetches that bundle before checking out the same immutable source SHA; this
does not require a fork, permission change or an uncommitted source overlay.

```powershell
python pipeline\run_jev_train.py --bundle-source --smoke-test --data-root shares/users/wuc/data/MIND_small --loss-type decision --prompt-contract legacy --enable-kl 0 --sample 32 --num-epochs 1 --batch-size 8 --micro-batch-size 1 --instance-type Singularity.ND12am_A100_v4 --timeout-seconds 3600 --display-name C1-decision-legacy-smoke
python pipeline\run_jev_train.py --bundle-source --smoke-test --data-root shares/users/wuc/data/MIND_small --loss-type decision --prompt-contract native_click --enable-kl 0 --sample 32 --num-epochs 1 --batch-size 8 --micro-batch-size 1 --instance-type Singularity.ND12am_A100_v4 --timeout-seconds 3600 --display-name C1-decision-native-smoke
```

Non-default training modes append `_lossdecision_promptlegacy` or
`_lossdecision_promptnative_click` to the historical output basename, preserving
C1clean2. Checkpoints save `jev_loss_type` and `jev_prompt_contract` in config.
Evaluate with `--prompt-contract auto` to restore the saved prompt; old configs
without metadata retain legacy behavior. Explicit evaluation overrides append
`_legacy` or `_native_click` to `jev_eval_results`, isolating prompt controls.

CPU checks cover candidate-CE equivalence, target exclusion, gradients, boundary
validation, dataset/scorer parity, metadata restoration, and an actual tiny
hybrid Qwen3.5 Trainer loop with gradient accumulation. AML submission tests use
the SDK but mock job creation; actual GPU execution is recorded below. A staged-model GPU
smoke and full real-data length audit remain required before a long run.
The actual staged tokenizer JSON, tokenizer config and chat template now match
the pinned public revision byte-for-byte. Actual MIND_small smoke preflight
admitted all 256 sampled examples (125 positive / 131 negative), with 182-2428
prompt tokens against the 4095-token budget and A/B IDs 32/33. This verifies the
smoke sample, not all training data or the identity of checkpoint weight shards.
C2/C3/C4 remain gated; no new webpage publication is implied by these edits.

**GPU smoke follow-up (2026-10-20):** GitHub push was denied to the signed-in
account, so committed source was uploaded through the authorized AML component
as a Git bundle. Fresh-public-clone reconstruction verified the bundle before
submission. First smoke `gray_snake_kpfsw591t3`, worker
`894b0495-82f4-48ae-9e0a-c8ededc673c3`, checked out
`bcab5be231218ca666f85ecda9807ed95db4dfed` and used exactly one ranking Standard
A100. All 256 prompts passed the budget check; 16/16 optimizer steps completed
at 01:11 UTC, with finite logged losses 1.657 (step 1) and 1.511 (step-10 window).
It was superseded during expensive resumable checkpoint writes, not because of
a loss/gradient failure. No completed final checkpoint or new AUC is claimed.
The explicit `--smoke-test` mode now requires 1-256 samples and one epoch, disables
intermediate/resumable saves, and retains the normal final-model save. Production
checkpointing remains unchanged.

**Corrected GPU smoke: COMPLETED (2026-10-20).** Parent
`ashy_boniato_r8dlmrx6tt` and worker `ac541480-7639-4ca0-a7d8-c2f97c665af7`
were both verified Completed at 02:21 UTC. The worker checked out exact source
`188b9b5b5e3a83bb0ad89ee8246a2aa82ac7448e` from AML component
`jev_train:4-188b9b5b5e3a`, not the stale GitHub branch. One ranking Standard A100,
32 examples, one epoch, global batch 8 / micro-batch 1, cutoff 4096, hist30,
abstracts, native-click prompt, candidate-only loss, KL disabled, and explicit
smoke mode were verified. All 4 optimizer steps completed; logged average train
loss was 2.259, with finite gradients. This is a functionality smoke, **not an
overfitting result, a dev evaluation, or evidence of improved ranking quality**.

The final model save was logged at 02:15 UTC; the training command completed at
02:16 UTC. Independent datastore readback at
02:23 UTC checked the saved config/tokenizer/chat template, A/B IDs 32/33, all
427 safetensor entries' shapes/dtypes/byte extents, and agreement with the reported
9,682,956,896-byte model file length. The entire weight payload was not downloaded or
reloaded; inventory/extent consistency is not a full numerical reload check.
The checkpoint is:

```text
shares/users/wuc/output_dir/jev_decision_smoke_20261020_finalonly/sft_jev4b_small_bs8_ep1_neg1.0_hist30_kl0.1_abstract_lossdecision_promptnative_click/final_checkpoint
```

The superseded first worker is confirmed Canceled before replacement allocation;
no two smoke GPUs were run concurrently. Monitoring also exposed local-time
`expiresOn` being treated as UTC; the credential helper now prefers the CLI's
epoch `expires_on`, with local-time fallback and regression coverage. GitHub push
remains denied to the current account; source delivery through the authorized
AML Git bundle succeeded, without permission changes or an uncommitted overlay.

**Next research gate:** full real-data length audit, matched frozen controls, and
loss-only versus native-prompt training/evaluation. No full-budget redesigned
training or new dev AUC was launched. The last measured full SFT AUC remains
0.5452; C2/C3/C4 remain gated. No webpage deployment was performed for this smoke.

**Research continuation (2026-10-20):** the smoke is not the research endpoint.
`src/audit_jev_inputs.py` now audits the entire deterministic training sampler
and every candidate in the first 5000 dev impressions under both prompt modes.
It records data/tokenizer hashes, an ordered input fingerprint, lengths and
over-budget counts without truncation. The full MIND_small sampler contains
469,264 pairs (236,344 positives, 232,920 negatives); these are not a new subsample.
The existing negative sampler is unchanged, including its slightly sub-1:1
realized ratio. **The full input-length audit passed:** every training pair
and every dev@5000 candidate fits the shared 4095-token prompt budget under
both contracts, preserving cutoff4096, hist30 and abstracts without truncation.

| Split | Candidate pairs | Legacy max tokens | Native-click max tokens | Over-budget pairs (either mode) |
|---|---:|---:|---:|---:|
| Full sampled train | 469,264 | 2960 | 2981 | 0 |
| Dev@5000 | 187,009 | 2875 | 2896 | 0 |

Median training lengths are 1412/1433 tokens (legacy/native); p99 is 2271/2292.
`jev-full-input-audit.json` records complete data/tokenizer hashes and ordered
input fingerprints; `jev-shared-input-admission.json` records the control's
chosen budget. This establishes a matched nontruncating protocol; it does not
establish why the historical SFT lost AUC.
An independent raw-input scan checked all 156,965 training impressions /
5,843,444 candidates and the 5000 dev impressions / 187,009 candidates. Both
splits have zero missing candidate news IDs and zero missing retained-history
news IDs. The sampler's 236,344 positive pairs account for every raw training
positive; negative subsampling remains deliberate, not a missing-news artifact.

The evaluator supports `--prompt-contract both`, loading the same frozen weights
once and evaluating legacy/native-click sequentially on the same dev impressions.
`--max-prompt-tokens` makes the shared budget fail-fast rather than assigning a
neutral score to an oversized candidate. Every candidate ID, label and unrounded
score is written to `.scores.jsonl` alongside full-precision `.metrics.json`;
partial score files retain `.partial` until a successful pass. Explicit
`--output-root` prevents collisions with historical evaluations. Evaluation now
uses the same immutable source pin and authorized `--bundle-source` mechanism as
training, with a job-private clone and optional command timeout.

**Paired frozen control COMPLETED and independently verified:** parent
`bright_ear_vnr75xb8tw`, worker `8445c8d2-9f3a-4c0e-b1fd-0158070ebe60`,
source `10b4379407d11e3e3a00c6cae3a75042eeaf731c`. The real server-side allocation
was verified as one ranking Standard `Singularity.ND12am_A100_v4` with a
43,200-second command timeout. Both contracts use hist30, abstracts,
dev@5000 (187,009 candidates), and the admitted 4095-token prompt budget.
Results are isolated under
`shares/users/wuc/output_dir/jev_decision_campaign_20261020/frozen`.
Both jobs were verified Completed on 2026-10-20 at 16:48 UTC. Independent
readback checked all 5000 impression IDs, 187,009 candidate IDs/labels/scores,
exact dev input hashes, and reconstructed all four metrics from raw scores.
This resolves the raw-score/tie limitation for these controls, not
retroactively for the historical C1 output.

| Frozen prompt | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Legacy | 0.63623517 | 0.30691291 | 0.33506672 | 0.39611543 |
| Native-click | 0.63313036 | 0.29945088 | 0.32628386 | 0.38803464 |
| Native minus legacy | -0.00310481 | -0.00746203 | -0.00878286 | -0.00808079 |

The paired impression-bootstrap AUC 95% interval is
[-0.00711694, +0.00112362] (2048 resamples, seed42). It includes zero:
**no prompt-only AUC improvement is established**, and the data do not establish
a definite AUC decrease either. Secondary unadjusted intervals are
MRR [-0.01208285, -0.00309624], nDCG@5 [-0.01362144, -0.00404372], and
nDCG@10 [-0.01257312, -0.00390352]. They favor legacy but are not adjusted for
multiple comparisons and are not repeated-training-seed estimates.
Keep the historical frozen reference 0.6432 separately labeled; this run's
matched frozen legacy reference is **0.63623517**, not a reproduction claim
for that older measurement. No weight adaptation occurred in these controls.

**Research decision:** prioritize `decision` loss with the **legacy** prompt,
holding the historical full-data sampler and optimizer recipe fixed, rather
than assuming the native prompt improves ranking. A native-prompt training
ablation remains secondary. Source evidence is
`jev-frozen-prompt-comparison.json` and both raw-score files in the session.
This is a control evaluation, not a training job or a new trained-model AUC.
Training ablations must use the
same admitted examples, history, abstracts and budgets; their runtime/resource
budget will be based on measured worker throughput, not the smoke's nominal
example count.

The historical full-budget eight-A100 C1 run's final attempt was recovered from
`user_logs/retry_005/std_log.txt`, not its initial-attempt log. It reports
174,900 training seconds (~48.6h), 8.05 samples/s, and all 11,001 steps over
three epochs. One-GPU full-budget replication is therefore not a reasonable
overnight promise. No larger training allocation is silently implied by the
completed one-GPU controls.

**Full loss-only run submitted (2026-10-21, UTC+8):** parent
`mighty_milk_0znks9vwg7`, worker `2c74fe4d-5b03-4df1-b2a3-502003a390eb`,
immutable source `ec86d37eb4b6c2894e8cd4af67ac71c2efcad986` delivered by AML Git
bundle. This is the full 469,264-pair, three-epoch
`decision` + `legacy` ablation, preserving seed42, global batch128/micro2,
lr1e-4, hist30/abstracts, cutoff4096, hard-neg0.5/neg1.0, bf16/ZeRO-2 and
no KL. This isolates the objective under the historical recipe rather than
mixing a shorter training budget or prompt change into the main comparison.
Following the renewed instruction to continue autonomously after this exact
resource plan was disclosed, the original one-node/eight-A100 type, Standard
ranking, was submitted with a 72-hour command timeout. Server readback verified
the node type, Standard tier, ranking VC and timeout; the worker was initially
Queued. Submission is not proof of optimizer progress or a completed result.
Afterward, serial one-A100 legacy-prompt evaluations will score both the old
full-vocabulary checkpoint and the new decision-loss checkpoint under the
current raw-score-preserving scorer. The old 0.5452 must not be treated as a
fresh matched rescore.

The private session controller `run_jev_full_campaign.py` monitors the recorded
parent rather than submitting another training job. It selects the highest
`user_logs/retry_NNN/std_log.txt`, verifies exact source/inputs/resources,
tracks all 11,001 planned optimizer steps, checks finite losses, and requests
cancellation on nonfinite training or the 72-hour wall-clock safety deadline.
After verified completion it checks saved checkpoint metadata/tensor extents
and serializes two 12-hour-timeout, one-A100 dev@5000 evaluations. An isolated
detached worktree pins their source even if the main research branch advances.
Each evaluation preserves raw scores and must pass exact input-hash, candidate
coverage, and four-metric reconstruction checks before paired comparisons are
written. The controller state is `jev-full-campaign-status.json`; final
comparisons, when actually available, go to `jev-full-adapted-comparison.json`.
Fourteen offline checks passed, including retry-log selection, resource/budget
guards, duplicate-submission recovery, and the full 5000-impression fixture flow.
Fixture metrics are not research measurements. C2/C3/C4 remain gated.

**Progress snapshot (2026-10-22 13:51 UTC+8):** the recorded parent/worker
remain Running, at 8,991/11,001 optimizer steps (81.7%), epoch2.452/3.
Recent logged losses are 0.6649, 0.6574 and 0.6743, below the earlier
approximately 0.693 plateau; no nonfinite values were detected. This is
improved training fit, not evidence of a ranking gain. Automatic supervision
is active. The historical full-vocabulary checkpoint was rechecked at 13:55:
config, tokenizer and chat-template files are present; the listed safetensors
file is 9,682,956,896 bytes; the prompt metadata/default is legacy.
`jev-old-checkpoint-readiness.json` records this bounded metadata/extent
readiness check, not a numerical reload or new AUC. No additional GPU was
allocated; matched old/new scoring still follows full training serially.

**Full training COMPLETED; matched scoring update (2026-10-23 10:44 UTC+8):**
both recorded training parent/worker are Completed. Exact-source, full-sample,
decision/legacy/no-KL, all 11,001 steps and epoch3 evidence passed.
Training runtime was 160,100 seconds (~44.47h); mean training loss 0.6914,
with no nonfinite values detected. The saved final checkpoint readback on
October 22 at 22:16 (UTC+8) verified config/tokenizer, A/B IDs 32/33,
427 tensor entries and a consistent 9,682,956,896-byte safetensors extent.
This is not a claim that every numerical weight was independently checked.

The controller then serialized the planned one-A100 evaluations. Fresh old
full-vocabulary rescore parent `purple_flag_yjyhnhtpyn`, worker
`b2dc2137-5884-4b2c-8d15-cbf02c3512a7`, is Completed. Every one of the 5000
impressions/187009 candidates, exact dev input hashes, and all four raw-score
metrics passed independent reconstruction:

| Matched legacy-prompt checkpoint | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Old full-vocabulary SFT | 0.54519484 | 0.23280850 | 0.24266650 | 0.31220688 |
| New decision-loss SFT (final verified October23) | 0.56365540 | 0.25667268 | 0.27457157 | 0.33463531 |

The new scorer parent `helpful_lizard_77ghb5kb5b`, worker
`8fb1b287-45c4-4e80-9176-38b6e78aeddc`, is Running, one ranking Standard
A100, immutable source `ec86d37`, same strict legacy dev@5000 protocol.
Its first attempt reached 1347 impressions; retry001 is at 644/5000 at the
snapshot time. These attempts are not combined and partial metrics are not
reported as final. The retry cause has not been diagnosed in this update.
No fresh training allocation, new decision-model AUC, improvement claim, or
downstream promotion is implied. `jev-full-campaign-status.json` tracks the
active pass; `jev-full-old_full_vocab.metrics.json` and `.scores.jsonl`
are the verified old-checkpoint evidence. The final paired comparison file
remains gated on complete new-model scoring.

**FINAL controlled loss-only outcome (2026-10-23):** training and both
matched evaluators are Completed; the controller finished successfully.
The earlier 10:44 progress snapshot is historical, not the current state.

| Same legacy prompt/dev@5000 protocol | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV-4B | 0.63623517 | 0.30691291 | 0.33506672 | 0.39611543 |
| Old full-vocabulary SFT | 0.54519484 | 0.23280850 | 0.24266650 | 0.31220688 |
| Candidate-only decision SFT | 0.56365540 | 0.25667268 | 0.27457157 | 0.33463531 |

| Paired AUC contrast | Mean difference | 95% impression-bootstrap interval |
|---|---:|---:|
| Decision minus full-vocabulary SFT | +0.01846056 | [+0.00735828, +0.02897442] |
| Decision minus frozen | -0.07257978 | [-0.08375560, -0.06140528] |
| Full-vocabulary SFT minus frozen | -0.09104033 | [-0.10272431, -0.07919353] |

All four ranking metrics improve relative to old SFT, but remain
below frozen. Full-precision values and all secondary intervals are preserved
in `jev-full-adapted-comparison.json`. Intervals use 2048 paired impression
resamples, seed42, without multiplicity adjustment. They quantify uncertainty
over this evaluated subset, not variability across training seeds. One
training seed and dev@5000 do not establish full-dev/test or SOTA performance.

**Interpretation:** candidate-only normalization helps relative to the old
full-vocabulary adaptation under this full-depth recipe, but does not restore
the matched frozen model's ranking quality. Neither the prompt change alone
nor this loss-only run establishes successful JEV adaptation. The experiment
does not identify the cause of the remaining quality loss and is not a
reproduction of native JEV low/high-exit CE plus cross-depth KL.
Do not promote this checkpoint to C2/C3/C4; no further GPU run is automatically
launched. The broader project SOTA objective remains unmet by this experiment.

**Independent final verification:** all three conditions' 5000 impression
IDs, 187009 candidate IDs/order/labels, unrounded scores and exact dev input
hashes passed. A separate verifier, importing no repository metric functions,
recomputed AUC as positive-negative pairwise win probability with half credit
for ties, implemented MIND MRR/DCG directly, and reproduced all twelve metric
values plus all three sets of bootstrap intervals within 1e-12.
`jev-final-independent-verification.json` records score/metric hashes and the
comparison SHA256:
`008f797cd1af96c93857a359546c8217b29ff1d890354472f89406de57cef470`.
`jev-full-campaign-status.json` is terminal
`full_adapted_comparison_verified`; all parent/worker statuses are Completed.
No GPU allocation from this experiment remains active.

```powershell
python pipeline\run_jev_train.py --bundle-source --data-root shares/users/wuc/data/MIND_small --output-root shares/users/wuc/output_dir/jev_decision_campaign_20261020/loss_only_legacy --loss-type decision --prompt-contract legacy --enable-kl 0 --sample -1 --num-epochs 3 --batch-size 128 --micro-batch-size 2 --max-history 30 --cutoff-len 4096 --use-abstract 1 --instance-type Singularity.ND96amrs_A100_v4 --timeout-seconds 259200 --display-name C1-decision-loss-only-full
```

**Publication:** the research-paper page and live feed were updated with the
verified paired-control table, intervals, full audit and negative result.
The full authenticated shared-site snapshot deployment succeeded on
2026-10-21 (UTC+8); other pages and authentication configuration were preserved.
Browser rendering was not independently rechecked in this background follow-up.
The subsequent full-run submission and its live parent/worker rows were also
published successfully through the complete shared-site snapshot on October21
(UTC+8), preserving authenticated routing. The new strict score verifier was
additionally exercised against the actual frozen legacy artifacts: all 5000
impressions, 187009 candidates, input hashes and four reconstructed metrics
passed. This validates the comparison machinery, not the still-pending training
outcome.

The October 23 interim report/feed publication also succeeded: completed
training, the verified fresh full-vocabulary rescore and both matched-evaluator
job IDs are now recorded on the authenticated shared site. The entire shared
snapshot was deployed; no partial decision-model metric was published.

**Final-results publication (2026-10-23):** the completed three-model table,
paired AUC intervals, negative frozen-to-adapted finding and downstream gate
were published with a freshly queried feed confirming all three parent jobs
Completed. Full authenticated shared-site deployment returned
`Status: Succeeded` and `Deployment Complete`.
`jev-final-results-publication-20261023.log` records the real deployment;
`jev-final-publication-verification.json` records report/feed hashes and
validated metric/interval bindings. Other shared pages and authenticated
routing were preserved. Signed-in browser rendering was not independently
rechecked.

#### JEV recovery follow-up: frozen residual readout pilot (2026-10-23)

The user requested continued improvement after the negative full-model result.
The actual score audit finds within-impression score correlation with frozen
JEV of 0.02179754 for decision SFT and 0.05134758 for full-vocabulary SFT.
Decision SFT's cross-class tie fraction is 0.01289701, below frozen's
0.01782339: extra ties alone do not explain its deficit. Its score mean moves
from frozen 0.10803005 to 0.49487821 after balanced-label training. These are
descriptive observations, not proof of catastrophic forgetting or of a bug.
Private evidence: `jev-recovery-v2-offline-diagnosis.json`.

The next bounded experiment preserves the frozen backbone and fits only a
small residual readout on its final normalized decision representation:
`margin = frozen_A_minus_B + standardized_hidden @ residual_weight + bias`.
This is an explicit new readout adapter, not successful full-model SFT or a
reproduction of native two-exit JEV training.

Predefined protocol:
- First compare the original conditional loader, training causal loader and
  actual collated candidate-loss path on four real training prompts. Record
  A/B head-row equality and logit margins; halt if rows differ or any margin
  discrepancy exceeds 0.25. This is a bounded parity diagnostic, not exhaustive
  proof that every training/scoring computation is equivalent.
- Exclude the final 256 training impressions entirely from fitting and use
  all their candidates for readout selection. Fit at most 16,384 deterministic
  sampled training pairs in complete mixed-label impression groups, using
  the existing seed42/history30/abstracts/legacy prompt contract.
- Fit binary and impression-local pairwise residual objectives on CPU with
  L2 strengths 100,10,1,0.1,0.01. Include unchanged frozen JEV as a candidate.
  Select by train-holdout AUC; frozen wins exact ties. Standardization is
  fitted exclusively on training features.
- After selection is saved, evaluate the selected readout and frozen JEV on
  dev rows 5001-6000, not the locked first5000 comparison. Preserve per-candidate
  margins, IDs, labels, feature arrays, readout parameters and input hashes.
- Allocate one Standard ranking A100 with a six-hour worker timeout. No new
  full-model training or downstream C2/C3/C4 is automatically authorized.

The probe is implemented in `src\probe_jev_readout.py` and submitted through
`pipeline\run_jev_readout_probe.py` with immutable bundled source.
No quality improvement is claimed before actual matched scoring completes.

**Submitted:** parent `ivory_star_h4z4j4tn84`, worker
`240ffc68-83e6-4289-985b-48ab9b966045`, source
`bceaab345e8e116810dc6292b7d8a189d7d1c452`, output
`shares/users/wuc/output_dir/jev_readout_recovery_20261023/run_02`.
Server-side readback verifies the ranking VC, one Standard ND12am_A100_v4,
21600-second timeout, exact source and output. At 17:57 UTC+8, the parent is
Running and the worker Queued; this is not a model-quality result.
The initial submission `teal_hamster_1nnbmw8xy9` inherited Recall VC from
the generic evaluation launcher. It was canceled after being observed Queued
and before replacement; the launcher now imports the verified ranking target
and rejects non-ranking targets before submission. The failed allocation
attempt remains in the job register rather than silently omitted.

`monitor_jev_readout_probe.py` monitors only the fixed replacement, enforces
the six-hour submission wall-clock deadline, saves retry-aware logs and checks
the actual server allocation. On completion it independently reconstructs
all eight matched metrics, full candidate coverage/labels/input hashes and
2048-resample paired intervals without submitting additional experiments.
Private status: `jev-readout-probe-status.json`; final verification target:
`jev-readout-independent-verification.json`.

By the 18:04 UTC+8 feed refresh, both corrected parent and worker are Running.
The research paper and fresh register were published with a successful full
authenticated shared-site deployment; real log:
`jev-readout-probe-publication-20261023.log`.
This confirms execution/publication, not a ranking improvement. Both the
train-holdout256 and fresh dev1000 windows were independently scanned locally:
every impression has positive and negative labels, so no degenerate group
needs to be silently skipped.

#### Frozen-readout pilot completed; fixed-head confirmation submitted (2026-10-23)

Both pilot parent `ivory_star_h4z4j4tn84` and its worker are Completed.
The final controller state is `readout_comparison_verified`: all 1000 new
dev impressions / 36802 candidates, input hashes, eight ranking metrics and
2048-resample paired intervals were independently reconstructed.
The conditional loader, causal loader and collated candidate-loss path gave
identical margins on the four real diagnostic prompts; A/B head rows matched.
This rules out a discrepancy in those four checks, not every possible
training failure.

The selected readout is `binary_l2_0.01`, chosen solely on the 256-impression
training holdout after fitting 16383 pairs from 3474 disjoint impressions.
Its train-holdout AUC is 0.69200943 versus frozen 0.59377260; these are
selection metrics, not dev performance.

| Matched new dev5001-6000 | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.61530684 | 0.29505517 | 0.32031180 | 0.38048552 |
| Frozen backbone + selected residual readout | 0.63367586 | 0.30265107 | 0.32476785 | 0.38781622 |

Paired AUC delta is +0.01836902, 95% impression-bootstrap interval
[+0.00029635,+0.03614217]. Secondary point metrics improve, but each of their
intervals includes zero. The AUC lower bound only narrowly exceeds zero;
this is a promising small pilot, not a confirmed robust adaptation or SOTA.
Do not compare its 0.6337 to frozen 0.6362 from the different first5000 window.
Evidence: `jev-readout-independent-verification.json`,
`jev-readout-result.json`, `jev-readout-train-selection.json`.

**Next confirmation, fixed before new scoring:** reuse exactly this readout,
without fitting, hyperparameter selection or dev-based changes, on fresh
dev rows6001-11000 (5000 impressions / 187881 candidates). Compare against
frozen JEV on the same candidates; do not pool the pilot into the confirmatory
interval. The head SHA256 is
`d7796a02baeed17a72acc42b23ae70f494654560dd3c2f39528cc54febe48f38`.
The confirmation streams representations instead of saving a multi-GB
feature matrix, and preserves raw margins plus 128 deterministic feature
spot checks for independent head-math readback. Allocation remains one
Standard ranking A100 with a six-hour cap; no full-model or downstream
training is launched.

Actual confirmation parent `great_oven_88h8j3jgjl`, worker
`ae73f51f-9d3c-496e-a402-8ffb37f0c5ed`, immutable bundled source
`a88c920ebc4c232a51df266bbb4dfc44ec477ab5`.
Output: `shares/users/wuc/output_dir/jev_readout_recovery_20261023/confirm_fresh5000_01`.
At 20:14 UTC+8 the parent is Running and the worker Queued; server-side
Ranking VC, one Standard ND12am_A100_v4, 21600-second timeout, source,
head path/checksum and output were verified. This is execution evidence,
not a confirmation quality result.
Attached controller `jev-readout-confirmation-controller`, PID65456,
uses `monitor_jev_readout_probe.py --confirmation` with these explicit
parent/source/output arguments. It enforces the six-hour submission
deadline and independently checks fresh5000 coverage, eight metrics,
2048 paired impression resamples and 128 residual-head feature spot checks.
Pilot and confirmation files have separate prefixes; do not launch duplicate
controllers or jobs. Three independent-verifier fixture tests pass, including
the complete 5000-impression fixed-head reconstruction.

The positive pilot table, uncertainty, clearly separated windows and submitted
confirmation are published in the research paper and fresh AML-backed feed.
Full authenticated shared-site deployment succeeded on October23; real log:
`jev-readout-positive-publication-20261023.log` (`Status: Succeeded`,
`Deployment Complete`). `jev-readout-publication-verification.json` binds
the deployed HTML/feed snapshot to the independently verified pilot artifacts
while retaining the previous full-model comparison. Browser rendering has not
been independently rechecked. GitHub push remains unavailable; the worker
received committed source through the verified AML bundle.

#### Fixed-head fresh5000 confirmation completed and verified (2026-10-23 UTC)

Parent `great_oven_88h8j3jgjl` and worker
`ae73f51f-9d3c-496e-a402-8ffb37f0c5ed` are Completed. At17:34 UTC
(October24,01:34 UTC+8), the controller exited successfully with stage
`fixed_readout_confirmation_verified`; no GPU job remains active in this
bounded readout experiment. Do not restart either completed controller or
resubmit these jobs.

The unchanged `binary_l2_0.01` head has the preregistered SHA256
`d7796a02baeed17a72acc42b23ae70f494654560dd3c2f39528cc54febe48f38`.
There was no fitting, hyperparameter reselection or retuning after the pilot.
All5000 fresh impressions /187881 candidates in dev rows6001-11000 were
checked against exact IDs/order/labels and audited input hashes. The verifier
independently reconstructed all eight metrics and2048-resample paired
intervals, and replayed128 deterministic feature/head mathematical spot checks.

| Matched fresh dev6001-11000 | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.62637000 | 0.29807593 | 0.32653138 | 0.38676991 |
| Frozen JEV + unchanged selected readout | 0.64790555 | 0.30990735 | 0.33898353 | 0.40229435 |

| Confirmation contrast | Mean difference | Paired95% impression-bootstrap CI |
|---|---:|---:|
| AUC | +0.02153556 | [+0.01380902, +0.02905105] |
| MRR | +0.01183142 | [+0.00431085, +0.01943781] |
| nDCG@5 | +0.01245215 | [+0.00359743, +0.02147156] |
| nDCG@10 | +0.01552445 | [+0.00792976, +0.02339710] |

All four unadjusted intervals exceed zero (2048 resamples,seed42).
This confirms a frozen-to-adapted improvement on the held-out confirmation
window, unlike the failed full-model recipe. It does not establish repeated
training-seed robustness, full-dev/test performance, MINDlarge transfer or
competition SOTA. Do not compare these absolute values directly with the
first5000 or pilot windows, and do not pool their intervals.

Evidence: `jev-readout-confirmation-independent-verification.json`,
`jev-readout-confirmation-result.json`,
`jev-readout-confirmation-dev.scores.jsonl` and
`jev-readout-confirmation-status.json`.
Result SHA256: `a3ad5872b4a41b572fd1eff3d97c932e94f9d9b0c96c2885ae2e54680486163e`.
Raw-score SHA256: `d5fc23355d899c2feaaeb68cbf50c2a9bae0d4601dbfcdb910050a92a39dcb95`.

**Next scientific gate:** preserve this exact head as the new adapted
reference; evaluate broader coverage and independent training seeds before
escalating to MINDlarge or downstream campaigns. No additional GPU allocation
or training is automatically launched by this completion.

The completed confirmation tables and all four intervals were published in
the research paper and fresh AML-backed feed. Full authenticated shared-site
deployment succeeded; real log `jev-readout-confirmed-publication-20261023.log`
records `Status: Succeeded` / `Deployment Complete`. Numeric/source bindings
are in `jev-readout-confirmation-publication-verification.json`; the earlier
pilot publication evidence and full-model comparison remain separate.
Browser rendering was not independently rechecked.

#### Continuing toward stronger adaptation: training-data scale control (2026-10-24 UTC+8)

The confirmed readout is a reference, not completion of the broader competitive
ranking objective. Test whether its small16383-pair fit is data-limited
before changing the backbone or launching downstream training.
This experiment changes only training data quantity: at most131072 sampled
pairs in complete mixed-label impression groups, with the same sampling seed42,
legacy prompt,hist30,abstracts,4095 prompt budget,binary objective and L2 0.01.
The original256 training-holdout impressions remain excluded from all fitting.

Preserve the confirmed head with SHA256
`d7796a02baeed17a72acc42b23ae70f494654560dd3c2f39528cc54febe48f38`.
Validate the original feature-cache row identities against the exact
deterministic larger sample prefix, input hashes and original pilot-result
checksum. Replay the first16 original training and16 holdout candidates in
identical batches against the frozen model; require exact feature/margin
equality before reusing cached arrays. Reproduce the original head's
training-holdout metrics before extracting any additional training features.
Save all larger-sample features/row metadata and their hashes for subsequent
CPU-only analysis rather than repeatedly paying for backbone extraction.

This stage reads only training inputs, fits one new head and records raw
training-holdout scores for frozen,confirmed-reference and scaled conditions.
It does not score dev, change hyperparameters based on dev, or reinterpret
training selection as independent quality evidence. The scaled head passes
the next gate only if its train-holdout AUC strictly exceeds the confirmed
reference; reference wins exact ties. A fresh dev comparison would be
registered separately before scoring, using new rows11001-16000 and retaining
both the confirmed readout and frozen backbone as controls.

Implementation: `src\scale_jev_readout.py`, shared deterministic group sampler
in `src\probe_jev_readout.py`, and `readout_scale.yaml` through
`run_jev_readout_probe.py --scale-training` with the checksum-pinned reference.
Allocation cap: one Standard Ranking A100,21600-second worker timeout and
six-hour submission wall-clock monitor. No additional GPU job or full-model
training is automatically submitted. Run IDs and actual worker state follow
only after real submission; no new quality result is claimed.

Actual preflight selects131071 pairs from30944 training impressions:
16383 original cached pairs plus114688 newly extracted pairs. All256
holdout IDs are excluded and the original cached metadata is an exact
prefix match (SHA256
`73d42dcebaeed02a5ccb5a56bb1f9710b311cafdbdc425260c3f6b17c55be4d7`).
All28 source JEV tests and5 independent-monitor fixture tests pass,
including altered head/metric/gate/row rejection. Component schema and
the training-only command are validated. Private preflight evidence:
`jev-readout-scale-preflight.json`.

Actual submitted parent `patient_nest_sjxf5vj85m`, source
`25b9a083d93a5b3f57c0ce2781fcab3bf9ba0d13`, output
`shares/users/wuc/output_dir/jev_readout_recovery_20261024/train_scale_01`.
Attached monitor `jev-readout-scale-controller` uses
`monitor_jev_readout_probe.py --scaling` with explicit parent/revision/output;
state and verification use the separate `jev-readout-scale-` prefix.
The monitor independently checks actual training metadata against the
preregistered sample, holdout exclusion, all12 raw holdout metrics, the
two head checksums,128 feature/head mathematical spot checks and the
strict train-only gate. The larger training feature-file digest is recorded
by the worker; the monitor does not independently download the entire file.
No new dev quality result is claimed.

At03:32 UTC+8, parent Running/worker
`6d032cc6-03de-4307-8eb3-378920c63172` Queued. The live controller
PID55044 verified actual Ranking VC,one Standard ND12am_A100_v4,
21600-second timeout,exact source/output and reference head path/checksum.
Keep these execution states distinct from a ranking-quality result.

The continuing protocol and actual AML states are published in the research
paper/feed while preserving both completed matched comparisons. Full
authenticated shared-site deployment succeeded; real log
`jev-readout-scale-publication-20261024.log` records `Status: Succeeded` /
`Deployment Complete`. Numeric/status binding evidence:
`jev-readout-scale-publication-verification.json`. Browser rendering was not
independently rechecked. This is an active experiment, not completed research
or a new quality result.

**Monitor interruption and recovery (October24,04:09 UTC+8):**
The original local controller exited1 without a Python traceback after its
04:03 heartbeat. Its PID55044 is no longer alive; no matching local
application-crash event was found, so the cause is not established.
Independent AML health readback at04:07 confirms both parent and worker
Running with no reported worker error. No GPU job was resubmitted/restarted.
The same source/output/job is now monitored by attached shell
`jev-readout-scale-controller-resume-01`, PID36444. Its fresh04:09 heartbeat
revalidates exact allocation/source/head and shows9488/114688 new features
processed. The original six-hour submission deadline remains09:31 UTC+8,
not six hours from the monitor restart. Private evidence:
`jev-readout-scale-monitor-interruption.json`; log
`jev-readout-scale-controller-resume-01.log`. Await this resumed shell's
completion notification, then read that known shell; do not duplicate it.

#### Larger-data fit completed; fresh three-arm test registered (2026-10-24 UTC+8)

Training-only parent `patient_nest_sjxf5vj85m` and worker
`6d032cc6-03de-4307-8eb3-378920c63172` are Completed. The resumed controller
exited0 at07:51 UTC+8, stage `training_scale_independently_verified`.
Actual131071 pairs from30944 training impressions were fitted with the
unchanged binary objective,L2 0.01,seed42 and frozen backbone.

**These are reused training-holdout selection metrics, not dev results:**

| Same256 training-holdout impressions /9526 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.59377260 | 0.26778246 | 0.28753004 | 0.34517504 |
| Confirmed16383-pair readout | 0.69200943 | 0.32886587 | 0.36049582 | 0.42302331 |
| New131071-pair readout | 0.71224853 | 0.36626940 | 0.40001304 | 0.46114360 |

The new head passes the preregistered strict train-holdout AUC gate.
Exact training row IDs/order/labels,holdout exclusion,all12 holdout metrics,
head checksums and128 two-head mathematical spot checks were independently
verified. The large training feature-file digest is worker-recorded, not
independently downloaded. Evidence:
`jev-readout-scale-independent-verification.json`.
Training result SHA256:
`5a4cf046889b40d5ce6ada7a5240f91a21919ab48d0161857fb559eca4a36dd7`.
New head SHA256:
`279ef6c47f09e7fe49b803810159cac6fba7ee23d612ea33318af6ec37f86457`.

**Fixed before new dev scoring:** compare three unchanged conditions on
untouched dev rows11001-16000,5000 impressions/183558 candidates.
Primary contrast is scaled-minus-confirmed readout, not merely scaled-minus-
frozen. Scaled-minus-frozen and confirmed-minus-frozen are supporting
contrasts. Report AUC,MRR,nDCG@5,nDCG@10 and2048 paired impression-bootstrap
resamples,seed42,unadjusted. Keep all prior windows separate; no pooling,
model selection,retraining or retuning on this dev window.

One frozen-backbone feature pass is reused for both readouts, preserving
raw three-arm margins and128 deterministic two-head feature spot checks.
The selected head and original confirmed head are both checksum-pinned;
the selected head's training-result provenance is pinned to the verified
training-only outcome. Exact-window preflight verifies all IDs/news coverage,
both classes per impression and both actual2560-dimensional heads.
All31 JEV source tests and6 independent-monitor fixture tests cover this
path, including executor mounting of both checksum-pinned heads.

Implementation extends `evaluate_jev_readout.py` with an optional comparison
head while preserving the earlier two-arm mode; component
`readout_comparison.yaml`, launcher `--comparison-head` and separate
`monitor_jev_readout_probe.py --comparison` mode. Budget remains one Standard
Ranking A100 with21600-second timeout and six-hour submission wall-clock cap.
No full-model training or additional automatic downstream job is launched.
The submission protocol below is retained; final dev results follow it.

Actual parent `khaki_muscle_yll119dfc7`, immutable source
`99d4ce25c26b0fd4f172eb154d1fc86605a30165`, output
`shares/users/wuc/output_dir/jev_readout_recovery_20261024/fresh5000_three_arm_01`.
Attached shell `jev-readout-scale-dev-controller`, separate status prefix
`jev-readout-scale-dev-`; monitor mode `--comparison`.
This is a separately registered evaluation allocation after the training-only
gate passed, not an automatic downstream training campaign.

At08:02 UTC+8 actual parent Running/worker
`8f10cf1e-c9d4-4953-8261-ae3d0442397e` Queued. Controller PID68876
has verified actual Ranking VC,one Standard ND12am_A100_v4,21600s,
exact source/output and BOTH head paths/checksums. Read only attached shell
`jev-readout-scale-dev-controller` after its completion notification;
do not restart completed training controllers or submit duplicate jobs.

The completed training-only table and newly submitted three-arm protocol
are published with fresh AML-backed states. Full authenticated shared-site
deployment succeeded; real log `jev-scale-three-arm-publication-20261024.log`
records `Status: Succeeded` / `Deployment Complete`. Binding evidence:
`jev-scale-three-arm-publication-verification.json` verifies all12 TRAINING
selection values separately from previous dev metrics and keeps the new
evaluator's quality-metric field empty. Browser rendering was not independently
rechecked. Previous pilot/confirmation publication evidence remains separate.

#### Three-arm fresh-dev result: scaling improves over confirmed readout

Parent `khaki_muscle_yll119dfc7` and worker
`8f10cf1e-c9d4-4953-8261-ae3d0442397e` are Completed. Controller exited0;
final stage `three_arm_comparison_verified` at2026-10-24 05:11 UTC
(13:11 UTC+8). Source checkout matched the pinned revision. Do not restart.

| Same untouched dev11001-16000 /5000 impressions /183558 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.62981627 | 0.30666348 | 0.33395060 | 0.39362614 |
| Confirmed16383-pair readout | 0.64560843 | 0.31194677 | 0.33804576 | 0.40073355 |
| Scaled131071-pair readout | 0.66443821 | 0.32573042 | 0.35819062 | 0.41857302 |

| Primary scaled-minus-confirmed | Mean difference | Paired impression-bootstrap95% CI |
|---|---:|---:|
| AUC | +0.01882978 | [+0.01445333,+0.02334820] |
| MRR | +0.01378365 | [+0.00877740,+0.01915956] |
| nDCG@5 | +0.02014485 | [+0.01465929,+0.02582601] |
| nDCG@10 | +0.01783947 | [+0.01305403,+0.02282135] |

All four primary intervals exclude zero (2048 paired impression resamples,
seed42,unadjusted). Supporting AUC scaled-minus-frozen is +0.03462194
[+0.02724089,+0.04217320]; confirmed-minus-frozen is +0.01579216
[+0.00789295,+0.02388755]. The confirmed-minus-frozen secondary-metric
intervals include zero; do not claim every contrast improves every metric.
All12 metrics and all12 contrast/metric intervals independently reconstructed;
exact coverage/order/labels,input hashes,both pinned heads,training provenance
and128 two-head mathematical spots verified. Evidence:
`jev-readout-scale-dev-independent-verification.json`.
Result SHA256 `74a4fbedb4c1b38940fe1942d0eaadbcebda448827c74e3ae65de225e7df3e2b`;
raw score SHA256 `cd3a2dea860de585314814ea9322c34ac5339fb454c5962db9bdf7a5029594e7`.

The preregistered data-scale gain generalized to a new dev window against
the confirmed adapter, not just frozen JEV. This is one training subset/seed,
not full-dev/test performance, native two-exit JEV reproduction or SOTA.
No cross-window absolute-score comparison or pooled interval is warranted.
The broader competitive objective remains open; retain the scaled head as
the current readout reference and preserve training caches for follow-up.

Final-result publication succeeded: complete authenticated shared-site
deployment log `jev-scale-three-arm-final-publication-20261024.log` records
`Status: Succeeded` / `Deployment Complete`. Binding proof
`jev-scale-three-arm-final-publication-verification.json` checks the result
and raw-score hashes,all12 published dev metrics,four primary paired intervals,
training-selection table kept separate,both heads,and Completed parent/worker
states from the freshly generated AML feed. Browser rendering not rechecked.

#### Cached readout refinement preregistration (2026-10-24)

After the three-arm scale result, reuse ONLY the131071 training-pair cache
and original256-impression excluded TRAINING holdout. No dev scores/features
enter this search. Independently download and hash the complete feature cache
against `39f2309b22c67db5d4245b905122b837dff6abd5f9ca98802638b341938a5aca`;
check exact independently audited training rows,holdout hashes/exclusion
and reproduce the current scaled head's training-holdout metrics.

Use the existing CPU float64 LBFGS helper,max100 iterations,at most4 threads.
Fixed candidates: binary L2=0.1,0.001 and impression-balanced pairwise
L2=0.1,0.01,0.001. Binary L2=0.01 is the current scaled reference,not a new
fit. Select maximum training-holdout AUC with reference winning ties;
candidate-order wins ties among new candidates. Preserve every head,all
candidate holdout scores,cache/provenance hashes and128 mathematical spots.
This is reused selection evidence,not independent robustness.

No additional backbone pass or GPU training is needed. If a new head
strictly passes the training-only gate,freeze it before registering a
comparison against the current131071-pair scaled head plus frozen JEV on
unused dev16001-21000. Primary refined-minus-scaled,2048 paired impression
resamples,seed42,unadjusted; no dev fitting,retuning or cross-window pooling.
If no candidate improves the training gate,retain the scaled head and
record the negative result without a redundant GPU comparison.

Actual metadata-only preflight passes5000 impressions/186527 candidates,
unique IDs,news/history coverage and both classes per impression; parameterized
component schema validates with historic default11000 and new start16000.
All18 directly related readout/refinement tests pass,including exact-reference
tie retention,cache tampering/leakage rejection,shared-feature evaluation and
the selected head's minimum fresh-window guard.

Committed source `20a8bd26abdb7ba0a84c2e2a2e651593b6f2ba43`.
Cache downloader `jev-refinement-cache-download` is active with bounded
16MB range reads; controller `jev-refinement-training-controller`,PID73664,
has a live `waiting_for_pinned_training_cache` heartbeat. It waits at most
30 minutes,then runs at mostone hour of4-thread CPU fitting and independent
raw-score/all-head verification. No new GPU job has been submitted.
Seven independent-monitor fixtures pass,including refined-head provenance,
both-head mathematics,all three contrasts and exact new5000-window coverage.
Active protocol and retained final-dev tables are published; full authenticated
snapshot deployment succeeded (`jev-refinement-active-publication-20261024.log`,
`Status: Succeeded` / `Deployment Complete`). No browser rendering recheck.
The current scaled head remains the only promoted reference; no refinement
selection or refined-head dev gain is claimed while this search is pending.
At13:30 UTC+8 the cache download completed successfully and the controller
advanced to `fitting_cpu_only`. The complete1.342GB training feature file
now independently matches its worker-recorded digest; training rows and
both original holdout artifacts also verify. This stronger cache readback
belongs to this follow-up,not the earlier training-only scale verification.
No dev inference or new GPU allocation has occurred.

#### Cached refinement completed; new head frozen before dev (2026-10-24)

CPU controller exited0 at13:33 UTC+8,stage
`training_refinement_independently_verified`. The predeclared training-only
AUC rule selects `binary_l2_0.001_refined`; no dev scoring entered selection.

| Reused training-holdout selection ONLY /256 impressions /9526 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Scaled reference,binary L2=0.01 | 0.71224853 | 0.36626940 | 0.40001304 | 0.46114360 |
| Binary L2=0.1 | 0.70212702 | 0.34273964 | 0.37794570 | 0.43546179 |
| Selected binary L2=0.001 | 0.71929553 | 0.36588080 | 0.40813435 | 0.45902339 |
| Pairwise L2=0.1 | 0.70619615 | 0.35914937 | 0.38719994 | 0.44838450 |
| Pairwise L2=0.01 | 0.71431441 | 0.36174241 | 0.39494351 | 0.45597102 |
| Pairwise L2=0.001 | 0.71780435 | 0.35147886 | 0.38811139 | 0.44807563 |

The training-only AUC gate passes,but selected MRR and nDCG@10 are slightly
below the reference on this reused holdout; do not claim all metrics improved.
Independent verification reconstructs all24 candidate metrics and every
candidate's scores over the full holdout feature cache,checks the full
training-cache digest,exact sample/exclusion and original reference bytes.
Evidence `jev-refinement-training-independent-verification.json`.
Frozen selected head SHA256:
`a8bcae18921897418822c8efa3b2aa12ff8a0dbdefbc8810eb45565865f19b6c`;
training result SHA256:
`83663c14594882d82bba3ba81402887ab5c2d8c6bebc1d793b6d63c629a28560`.
This is selection evidence,not dev or SOTA. The separately registered
newdev16001-21000 test remains primary refined-minus-scaled,with frozen
supporting control,5000 impressions/186527 candidates and no retuning.
Actual submission IDs and source must be recorded after real allocation.
Promotion rule fixed before scoring: retain the current scaled reference
unless the refined-minus-scaled AUC paired95% interval is wholly positive.
Report every secondary metric/interval,including regressions; do not select
another candidate or tune from this dev result. No automatic downstream
training follows either outcome.

Actual parent **`maroon_music_n00417p68k`**, worker
`907f0f26-3a75-4ca6-a2d0-ca153351814b`; immutable evaluation source
`6baf350d9533f8db60c803024f914214df2c0484` (CPU fit source remains20a8bd26).
Output `shares/users/wuc/output_dir/jev_readout_recovery_20261024/refinement_dev_01`.
Uploaded selected head,training result,selection,parity and protocol were
read back byte-for-byte before submission; no remote artifact overwrite.
At13:36 UTC+8 parent Running/worker Queued; monitor PID66036 verified one
Standard Ranking ND12am_A100_v4,21600s,exact source/output,both head
paths/checksums and start16000. Live attached shell
`jev-refinement-dev-controller`; profile `--refinement`,status
`jev-readout-refinement-dev-status.json`. Six-hour creation-based deadline
is enforced; no duplicate evaluator/controller. Two submission guard tests
also reject failed/tied gates,changed provenance and existing submission intent.
No refined-head dev quality result yet.
The completed CPU-selection table and real new-dev submission are published
in the full authenticated shared-site snapshot. Deployment receipt
`jev-refinement-submission-publication-20261024.log` records
`Status: Succeeded` / `Deployment Complete`. Binding proof
`jev-refinement-submission-publication-verification.json` checks all24
training-selection cells,head/result provenance,actual new parent state,
empty refined-head dev metric and all preserved prior dev tables/intervals.
Browser rendering not independently rechecked.

#### Refined-head new-dev result: no established gain; retain scaled reference

Parent `maroon_music_n00417p68k` and worker
`907f0f26-3a75-4ca6-a2d0-ca153351814b` Completed; controller exited0 at
2026-10-24 10:47 UTC (18:47 UTC+8),stage `three_arm_comparison_verified`.
Source checkout matched6baf350d. SAME untouched dev16001-21000:

| Same5000 impressions /186527 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.62556294 | 0.29615804 | 0.32245111 | 0.38317481 |
| Retained scaled131071-pair reference | 0.66751333 | 0.32407162 | 0.35662381 | 0.41742222 |
| Refined binary L2=0.001 | 0.66938687 | 0.32318974 | 0.35805351 | 0.41735250 |

| Primary refined-minus-scaled | Mean difference | Paired impression-bootstrap95% CI |
|---|---:|---:|
| AUC | +0.00187354 | [-0.00083710,+0.00461701] |
| MRR | -0.00088188 | [-0.00450070,+0.00267975] |
| nDCG@5 | +0.00142970 | [-0.00238980,+0.00529909] |
| nDCG@10 | -0.00006972 | [-0.00337412,+0.00309031] |

Every primary interval includes zero (2048 paired impression resamples,
seed42,unadjusted). The preregistered promotion rule fails: **do not promote
the refined head; retain binary L2=0.01 scaled head279ef6c4**. Slightly
higher AUC point value is not an established improvement. No dev retuning
or selection of another grid candidate follows this result.
Supporting scaled-minus-frozen AUC is +0.04195039
[+0.03454470,+0.04924813]; refined-minus-frozen is +0.04382393
[+0.03606471,+0.05124463]. Both adapters exceed frozen on this window,
but that does not answer the primary refinement question affirmatively.
All12 metrics,three contrasts,input hashes,exact coverage/order/labels,
both pinned heads,training-result provenance and128 two-head mathematical
spots independently verified. Evidence:
`jev-readout-refinement-dev-independent-verification.json`.
Result SHA256 `a3074f5af7acfdea5f431014ff34a809efaead61ebf685022d06476f27001f32`;
raw scores SHA256 `928e389203fe608b7edc348ea487f82bc81f57103b8c65b1053eafd2663a4425`.
One sample/seed,partial dev,not native two-exit JEV or SOTA. Do not pool with
the prior dev windows or interpret the reused0.71929553 selection AUC as dev.
Final negative-result publication succeeded in the full authenticated
shared-site snapshot; receipt `jev-refinement-final-publication-20261024.log`
records successful deployment. Binding proof
`jev-refinement-final-publication-verification.json` checks all12 dev
metrics,four primary intervals,result/raw hashes,Completed parent/worker
and the no-promotion decision,while retaining all prior training/dev tables.
Browser rendering not independently rechecked.

#### Next registered experiment: double data,not more dev-directed tuning

The failed refinement does not replace the retained scaled L2=0.01 head.
Test data quantity again with objective,L2=0.01,seed42,prompt and frozen
backbone fixed. Exact deterministic sample262143 pairs/69958 impressions:
reuse the verified131071-pair cache as exact prefix; extract only131072 new
training representations. All original256 training-holdout impressions
remain excluded. Pin reference head279ef6c4 and training result5a4cf046.
Original holdout features/rows remain in the original pilot cache,not in
the larger training output; the new path explicitly mounts that separate
root and checks its pinned digests. Replay16 train and16 holdout rows in
identical batches before reuse,and reproduce retained reference holdout
metrics. Hash the entire reused training feature file39f2309b.

Select only if new training-holdout AUC strictly exceeds the reference;
reference wins ties. Training stage never reads/scores dev. Save all
scores,head hashes,exact row metadata and128 two-head mathematical spots.
Independent monitor reconstructs12 holdout metrics and checks strict gate,
full expected sample/exclusion,reference cache/provenance and both heads.
Future evaluation must be separately registered on newdev21001-26000,
comparing doubled head,retained131071 head and frozen. Primary doubled-
minus-retained AUC,2048 paired impression resamples,seed42,unadjusted;
promote only if primary AUC95% interval wholly positive; report all
secondary results,no dev retuning or cross-window pooling.

One Standard Ranking A100,21600-second worker timeout and six-hour
creation-based wall-clock deadline. No full-model/downstream training.
Actual doubled-sample preflight checks all262143 IDs/order/labels,69958
training impressions,exact131071 prefix,holdout exclusion and new component
schema.20 related source tests and8 independent-monitor fixtures pass.
Run IDs/source must follow actual submission; no new quality claim yet.
Actual parent **`coral_turtle_b21f31930c`**, worker
`673f3dd0-32ae-4885-9ad8-f9ad0c778652`; immutable source
`ee17243f0d738c216b37998a780710ec7a5f10ba`.
Output `shares/users/wuc/output_dir/jev_readout_recovery_20261024/double_train_01`.
At19:01 UTC+8 parent Running/worker Queued; monitor PID26260 verified
actual Standard Ranking A100x1,21600s,source/output,retained reference
head path/checksum,262144 pair-cap tag and original holdout-cache root.
Attached shell **`jev-double-scale-controller`**; monitor profile
`--doubling`,status prefix `jev-readout-double-scale-`. Enforces original
creation-based six-hour deadline; do not duplicate or restart it.
No new train-selection or dev quality metric is available.
The retained-head decision and actual doubled-data submission are published
in the complete authenticated shared-site snapshot. Receipt
`jev-double-scale-submission-publication-20261024.log` records
`Status: Succeeded` / `Deployment Complete`; binding proof
`jev-double-scale-submission-publication-verification.json` verifies source,
262143/131071/131072 counts,actual fresh parent state and empty training-only
quality metric while preserving all prior training/dev tables and intervals.
Browser rendering not independently rechecked.

#### Doubling job completed; local metadata readback repaired

Independent AML readback confirms parent `coral_turtle_b21f31930c` and
worker `673f3dd0-32ae-4885-9ad8-f9ad0c778652` Completed,with no worker
error. Local controller exit1 is a verification failure,not a failed GPU
training run: the old20,000,000-byte request truncated the22,055,701-byte
`train.rows.json`. Remote file status independently matches the canonical
size derived from the exact262143-row audit.

The private verifier now derives its metadata bound from that audited
sample and requests one extra byte to reject unexpected oversize data.
Regression coverage exercises complete over20MB readback and excess-byte
rejection; no coverage,identity,provenance or metric check was removed.
Original failure log/status are preserved. Shell
`jev-double-scale-verifier-recovery-01` reads and verifies the SAME completed
job; no GPU resubmission or budget reset. Original creation was October24
11:00:29 UTC,deadline17:00:29 UTC (October25 01:00:29 UTC+8).
Independent training-outcome verification is pending; worker-reported
selection numbers are not yet treated as verified quality evidence.
Evidence `jev-double-scale-readback-recovery.json`.

#### Doubling training result independently verified (October25,2026 UTC+8)

Verification-only recovery exited0 at October24 16:13 UTC /October25
00:13 UTC+8,stage `training_scale_independently_verified`. Both cloud jobs
Completed; original metadata-readback failure remains documented above.
No GPU work or budget was restarted.

| Same reused TRAINING-holdout selection ONLY /256 impressions /9526 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.59377260 | 0.26778246 | 0.28753004 | 0.34517504 |
| Retained131071-pair reference | 0.71224853 | 0.36626940 | 0.40001304 | 0.46114360 |
| New262143-pair doubled head | 0.71305557 | 0.36538425 | 0.40591689 | 0.45882088 |

Strict train-only AUC gate passes by +0.00080704; MRR and nDCG@10 are
slightly lower. This is a small reused-training-holdout selection gain,
NOT independent dev performance or an all-metric improvement.
All12 metrics,exact262143-row sample/exclusion,both head hashes,reference
cache/provenance and128 two-head mathematical spots independently verified.
The new2.68GB training-feature digest8d4f803d is worker-recorded,not
independently downloaded in full. Evidence:
`jev-readout-double-scale-independent-verification.json`.
Selected `binary_l2_0.01_doubled`,head SHA256:
`2df5077c4e20ef115f6a379f23536450eefcb8a20bfc7e14534e20e49064ff17`.
Training result SHA256:
`16771883911bf4ddf695e363db8f1839f8ae35509041478e1586e434d1ee09d1`.

Freeze this exact head for the already registered newdev21001-26000
comparison against retained131071 head279ef6c4 and frozen. Primary
doubled-minus-retained,2048 paired impression resamples,seed42,unadjusted.
No fitting,reselection,retuning or pooling with earlier windows. Retain
the current reference unless the primary AUC95% interval is wholly positive;
report every secondary metric and regression. No new dev result yet.

#### Actual doubled-head new-dev submission (October25,2026 UTC+8)

Submitted parent `kind_snake_fkcl3465qw`,worker
`9f55bfbc-db7b-4439-9817-0303b17f64ec`;source
`8d412f4ea386f726fb355a3932a136571d7a3938`.
Output `shares/users/wuc/output_dir/jev_readout_recovery_20261025/double_fresh5000_01`.
Exact previously unused dev21001-26000:5000 impressions/186223 candidates,
news/history coverage,unique IDs,both classes and exact order audited.
Order SHA256:
`8a82e336ae41bd442da8b2e8d233104826be72af638a9d10c623a1451e3983a3`.
Both actual head checksums and dimension2560 independently preflighted;
selected training-result digest16771883 reconfirmed before allocation.

Primary frozen doubled2df5077c-minus-retained279ef6c4,plus frozen control,
from one feature pass. Same2048 paired impression resamples,seed42,
unadjusted;promoteonlywith wholly positive primary AUC95% interval,
report all secondary regressions,no fitting,reselection,retuning or pooling.
One Standard Ranking A100/21600s,original creation-based six-hour deadline.
Initial actual state at October24 16:19 UTC /October25 00:19 UTC+8:
parent Running,worker Queued,NOT yet extracting. Verifier PID49200
checks actual source/output,resources,both head paths/checksums and typed
window. Related9 independent mathematical/provenance/readback fixtures pass.
Controller attached as `jev-double-dev-controller`;await its completion
notification and read that known shell once,no duplicate job/budget reset.
No new dev quality metric yet;retain current279ef6c4 pending this test.

#### Final doubled-head new-dev result: no promotion (October25,2026 UTC+8)

Parent `kind_snake_fkcl3465qw` and worker
`9f55bfbc-db7b-4439-9817-0303b17f64ec` both Completed;controller
`jev-double-dev-controller` exited0. Completion notification consumed and
known shell read once. Actual evaluation source remains
`8d412f4ea386f726fb355a3932a136571d7a3938`. Final independent verification:
October24 21:43 UTC /October25 05:43 UTC+8,
stage `three_arm_comparison_verified`,source checkout observed.
Earlier submission/queued/no-result entries above are historical.

| Same untouched dev21001-26000 /5000 impressions /186223 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.62787974 | 0.30124496 | 0.32909718 | 0.39100421 |
| Retained131071-pair reference | 0.65561171 | 0.32347555 | 0.35483987 | 0.41735882 |
| Doubled262143-pair readout | 0.65448886 | 0.32171964 | 0.35290252 | 0.41657858 |

| Primary doubled-minus-retained metric | Mean difference | Paired impression-bootstrap95% CI |
|---|---:|---|
| AUC | -0.00112285 | [-0.00307466,+0.00088349] |
| MRR | -0.00175591 | [-0.00424986,+0.00062358] |
| nDCG@5 | -0.00193735 | [-0.00465904,+0.00056335] |
| nDCG@10 | -0.00078023 | [-0.00305400,+0.00152517] |

2048 paired impression resamples,seed42,unadjusted. All primary point
differences negative,but every interval includes zero;not a statistically
established regression and no incremental gain. Promotion gate fails:
**retain131071 head279ef6c4;do NOT promote doubled2df5077c**.
Supporting retained-minus-frozen AUC+0.02773197[+0.02035031,+0.03488561];
doubled-minus-frozen+0.02660912[+0.01938342,+0.03366962]. Beating frozen is
not the incremental promotion criterion. No dev retuning,reselection or
pooling with prior windows;no automatic further data-only scaling job.

All12 metrics,three contrasts,exact coverage/input hashes,both heads,
training provenance and128 two-head mathematical spots independently
verified. Proof `jev-readout-double-dev-independent-verification.json`.
Result SHA256:
`3711b20e2a7a26f23d497c844ce726371c1a5534d0315ca9bb6ad65c5d06d3e4`.
Raw-score SHA256:
`aa9bf431f12e84badc647c06d4a4a5c281c614ace4b0fa1ac4a55e389d5631b6`.
One training subset/seed,partial dev,not full-dev/test or SOTA.

#### Next controlled test: cached nonlinear correction (October25,2026)

Data-only doubling failed its independent-dev promotion gate. Keep retained
131071-pair head279ef6c4 unchanged;test a different function class rather
than allocating another data-extraction job.

`src/nonlinear_jev_readout.py` fits a small correction to that retained
margin:standardize the cached final-normalized JEV features with the
retained head's original mean/scale,then Linear(2560,16),ReLU,training-only
dropout0.1,and Linear(16,1). Add its scalar output to the unchanged
retained linear margin. Zero initialization of the final layer preserves
that reference before fitting. Both frozen backbone and retained linear
parameters stay unchanged;no extra model features or dev inputs.

Predeclared CPU-only protocol:reuse the independently downloaded131071
training pairs and original excluded256-impression/9526-candidate training
holdout. Verify full feature-cache/provenance hashes,exact sampled row
identity/exclusion,and reproduction of all four retained holdout metrics
before fitting. CPU AdamW,lr0.001,weight_decay0.01,batch2048,seed42,maximum
4threads/1800seconds;one16-epoch trajectory,only checkpoints4/8/16.
Best training-holdout AUC wins,reference wins ties. Open a future dev gate
only if that best checkpoint improves holdout AUC by at least0.005 AND
does not lower MRR,nDCG@5 or nDCG@10;otherwise retain the exact reference.
This is a conservative compute gate,not statistical evidence of quality.

Persist all three heads,16 candidate metrics,full holdout scores,128
feature spots and hashes;independently reconstruct every head's full
holdout mathematics and selection decision. No dev data or dev scores
are inputs,no GPU is automatically submitted,and any subsequent
fixed-head evaluation must use new dev-start>=26000 (rows26001 onward).
Reusing this training holdout and checking three epochs risks selection
overfit;one seed is not independent robustness. Existing four-field
linear-head format/evaluator are deliberately unchanged;the new strict
eight-field nonlinear head is CPU-tested separately before any future
evaluation integration. All28 targeted source tests pass,including XOR
learning,zero-residual parity,NumPy/Torch inference agreement,determinism,
invalid heads/data/nonfinite parity,deadline rejection,strict gate without
post-selection fallback and legacy behavior.

#### Cached nonlinear binary result: gate fails (October25 05:59 UTC+8)

CPU controller exited0;source
`af9eb9c0c584bfaf4638675898e7c195dba79af4`. All16 holdout metrics,
all four heads' full9526-row mathematics,128 spots,full training-cache
digest,sample identity/exclusion and selection independently verified.
No dev data read or GPU allocated.

| Reused TRAINING holdout only /256 impressions /9526 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Retained131071 linear reference | 0.71224853 | 0.36626940 | 0.40001304 | 0.46114360 |
| Binary MLP16 epoch4 | 0.71175598 | 0.36719867 | 0.39928874 | 0.46108529 |
| Binary MLP16 epoch8 | 0.70950922 | 0.35658260 | 0.38447309 | 0.45545861 |
| Binary MLP16 epoch16 | 0.71685153 | 0.35061941 | 0.38948949 | 0.44751655 |

Best AUC checkpoint16 gains only+0.00460300,below the preregistered0.005
gate,and lowers all three secondary metrics. Retain exact head279ef6c4,
`nonlinear_head_opens_dev_gate=false`. Do not evaluate this binary MLP
on dev or weaken its gate. Proof `jev-nonlinear-independent-verification.json`.
Result SHA256:
`6fe1bc2678aebd43759549b2760831a797967189fa844cd29d62d1524e000cfb`.
Raw holdout scores SHA256:
`f92c31105cf91d2a352f90240b5a1c377956424c3c86ae9cbb989a0cfebecfcc`.

#### Next CPU-only objective control: impression-balanced pairwise MLP

The binary MLP's training-holdout AUC/top-rank tradeoff motivates a ranking
objective control,not dev-based tuning. Keep the same architecture,
cache,reference,initialization,optimizer,seed42,16epochs,checkpoints4/8/16,
4threads/1800s and unchanged0.005/all-secondary gate. Only replace binary
CE by softplus(negative_margin-positive_margin) over within-impression
positive/negative pairs. Each impression has equal total pair weight,
as in the existing linear pairwise baseline. Uniformly shuffled pair
minibatches use importance weights so the full-epoch objective has that
same equal-impression normalization. Default binary behavior is preserved
behind `--objective pairwise`;separate output and candidate names.

This remains cached TRAINING-only selection on a reused holdout,not a
fresh independent validation result. No dev reads,no automatic GPU
submission,no fallback to a different epoch when the best-AUC checkpoint
fails secondary gates. Future dev-start>=26000 remains mandatory.
All29 targeted tests pass,including hand-reconstructed unequal-group
pairwise weighting and invalid impression-group rejection.

#### Cached pairwise nonlinear result: gate also fails (October25 06:06 UTC+8)

CPU-only source `de50e05c15eb9d7bf0ddebfc77142af6afa6b910`;
controller exited0. All16 metrics,all four full-holdout head mathematical
reconstructions,128 spots,full-cache hash,sample/exclusion and unchanged
selection rule independently verified. No dev reads or GPU allocation.

| Reused TRAINING holdout only /256 impressions /9526 candidates | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Retained131071 linear reference | 0.71224853 | 0.36626940 | 0.40001304 | 0.46114360 |
| Pairwise MLP16 epoch4 | 0.71410710 | 0.35689598 | 0.39043828 | 0.45345119 |
| Pairwise MLP16 epoch8 | 0.70789252 | 0.35024826 | 0.38635408 | 0.44265946 |
| Pairwise MLP16 epoch16 | 0.71069218 | 0.34618127 | 0.38197816 | 0.44096265 |

Best epoch4 AUC improves only+0.00185858,below0.005,and all secondary
metrics decrease. Strict gate fails;retain exact279ef6c4. No fallback
epoch reselection,no GPU test for either binary or pairwise nonlinear
head. This rejects these specified controls,not all nonlinear modeling.
Proof `jev-nonlinear-pairwise-independent-verification.json`.
Result SHA256:
`560adbb2a8f57842692993153e0cbce30bba21b6a092cf27db9286729c1d3c56`.
Raw holdout scores SHA256:
`99c6bda878e3b195380b14b363a7e3afbaec19eb15d07619517c780f4c8cf2b8`.
No new dev result or SOTA claim. Both CPU runs are complete.

#### Next signal-family control: cross-fitted popularity (October26,2026)

Dataset remains **MIND_small**,not MIND_large or competition/test scores.
The binary/pairwise nonlinear controls did not qualify for dev evaluation.
Keep retained131071 head279ef6c4 frozen,and add training-behavior signals
not present explicitly in its text features.

`src/behavior_jev_readout.py` uses only the checksum-pinned official
MIND_small training behaviors (SHA256a424547c) and verified feature cache.
Exclude ALL256 training-holdout impressions before counting any exposure
or click. Hash each remaining impression with SHA256("42:"+ID),first8
bytes big-endian modulo5. Every fitting row uses counts and global class
priors excluding its ENTIRE impression fold. Holdout rows use the full
non-holdout training population. Check every cached impression/candidate/
label against source behaviors;reject malformed/duplicate/missing rows.

Four fixed features:log1p(exposures/global_exposures*1e6),
log1p(clicks/global_clicks*1e6),smoothed item-CTR log-odds minus background
CTR log-odds,and an observed-item indicator. Beta smoothing strength20
is fixed before results. Unknown news maps to neutral features;do not
invent a click prior from evaluation labels. Persist full and per-fold
integer counts and validate sum/count identities on reload.

Fit only a four-feature correction to the unchanged retained margin,
using the existing float64 LBFGS/max100 iterations:two fixed candidates,
binary and equally impression-weighted pairwise,bothL2=0.01. Reference
wins AUC ties. Unchanged conservative compute gate:best reused-training-
holdout AUC must improve by>=0.005 AND MRR/nDCG5/nDCG10 must not decrease;
otherwise keep exact279ef6c4 with no alternative-candidate reselection.
Max4 CPUthreads/1800s,no dev inputs or automatic GPU submission. Only if
the gate passes may a separately frozen comparison use unused
dev26001-31000;previous windows remain excluded.

All35 targeted source tests pass,including own-fold target-label mutation,
entire-holdout label mutation,hand-calculated smoothing,unknown-item
neutrality,count/hash corruption,source/cache identity checks and legacy
readout behavior. This is a train-only compute gate on a reused holdout,
not independent robustness. Whole official training logs supply the
population,not only the131071 cached fitting pairs;fold exclusion is
not a claim of temporal/causal validation or user-disjoint validation.

#### Behavioral selection outcome and frozen fresh-dev protocol

Training-only source25b50700e63aa273e553328793c13c4d0bb81afb completed
successfully. Independently rebuilt every source integer count,the full
131071-row fit/9526-row holdout feature matrices,all three heads' complete
holdout score mathematics,and all12 metrics. No dev scores were used.

| MIND_small reused TRAINING holdout | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Retained131071 reference | 0.712248527659 | 0.366269403387 | 0.400013039175 | 0.461143604744 |
| Behavioral binary L2=0.01 | 0.735290951674 | 0.367851403727 | 0.407183903881 | 0.465065764585 |
| Behavioral pairwise L2=0.01,selected | 0.736222448598 | 0.367711763788 | 0.408061414881 | 0.465881764661 |

Pairwise passes the previously fixed compute gate:AUC+0.023973920939,
with all three secondary point estimates higher. Freeze this correction,
the population and the retained JEV head BEFORE any new-dev scoring:

- Correction32e8af31786d059d90b47c32ecd118cfe0633f9c7c8d4922ed52deb703d2ff68.
- Populationba4d7855aabe6894c6eecd9c9a0e69f1d326200891bc174dae048f9841ecc62c.
- Training result1670421674adeb8fb41f858d065edd5b276b97e3fcddb7fa0bdde23d761e63fe.
- Retained JEV head279ef6c47f09e7fe49b803810159cac6fba7ee23d612ea33318af6ec37f86457.

Use ONLY untouched MIND_small dev26001-31000,5000 impressions,without
pooling earlier windows. Metadata preflight verified188821 candidates,
unique impression IDs,news/history coverage and both classes in every
impression. Impression-order SHA256:
ca6be42a0bcf65aa61109254769fd9bf648d689551d4b5a1f27fc90c7d60d775.
One Standard Ranking A100/21600s frozen-backbone
pass produces retained279ef6c4,originald7796a02 and frozen margins.
Then `src/evaluate_jev_behavior.py` adds the selected frozen four-feature
correction to retained279ef6c4 margins on CPU;the correction is NOT a
2560-dimensional backbone head. No fitting,reselection,threshold search,
population updates or label-dependent feature construction on dev.

Final three arms are behavioral hybrid,retained279ef6c4 and frozen JEV.
Primary contrast is hybrid-minus-retained. Use2048 paired impression
bootstrap replicates,seed42,unadjusted95% intervals,for all four metrics
and all three contrasts. Promote ONLY if the primary AUC interval is
wholly positive;report every secondary regression. Otherwise retain
279ef6c4 regardless of its competitor's point estimate. Intermediate
baseline-only results cannot alter whether or how the frozen hybrid runs.
Freeze an immutable source worktree for the delayed CPU stage;verify the
complete dev candidate order/coverage,all features and correction math,
all12 metrics,input checksums,training provenance and bootstrap results.

The reused training holdout can overfit;the population adds official
training-log information beyond cached fit rows. This result is NOT
independent dev evidence,MIND_large,test/leaderboard or SOTA evidence.

Actual evaluation submitted October25,2026 at20:03 UTC /October26 04:03
UTC+8:parentkhaki_clock_p86ytlrfth,
workerc33d2b74-0cea-4b9b-b5b5-8029f31dd421.
Committed source7c6aa8b3948c7e759125570f732e74dabc148e25;immutable CPU
worktree `jev-behavior-fixed-source-7c6aa8b3`. Output
`jev_readout_recovery_20261025/behavior_fresh5000_01`.
Initial server readback:parentRunning,workerQueued,not yet feature extraction.
Actual Standard Ranking one-A100/21600s allocation,source,output,both
backbone-head paths/checksums and zero-based26000 window verified.
Attached controller `jev-behavior-dev-controller` PID73884;monitorPID78068.
Creation-based six-hour deadline,no duplicate submission or budget reset.
The controller independently verifies the baseline pass,then runs the
immutable CPU correction and independent full-feature/head/metric/paired
verifier automatically. Quality remains unavailable;279ef6c4 remains
retained until the predeclared independent-dev gate passes.

Behavioral training/actual-submission publication completed:full authenticated
shared-site snapshot returned `Status: Succeeded` and `Deployment Complete`.
Receipt `jev-behavior-submission-publication-20261025.log`;binding proof
`jev-behavior-submission-publication-verification.json` verifies Table21's
12 training cells,all frozen hashes,actual source/window/188821 candidates,
parent/worker identities and empty quality fields in one fresh AML feed.
Historical windows and prior rejection decisions remain separately verified.
No signed-in browser rendering verification is claimed;dev result pending.

Web status refresh,October26 00:41UTC /October25 17:41UTC-7:the same
parent/worker are nowRunning;baseline scoring162432/188821 candidates
(86.0%),no longer queued. Exact runtime source checkout is observed.
This is execution progress only;no independently verified behavioral dev
quality or promotion decision is available. The frozen recipe and
creation-based deadline are unchanged.

Refreshed method/dataset/results/status page and fresh AML execution feed
deployed successfully as a full authenticated shared-site snapshot.
Receipt `jev-behavior-web-refresh-publication-20261026.log`;
binding proof `jev-behavior-web-refresh-publication-verification.json`.
Training scores remain explicitly training-only;the running dev test has
no published quality metric. Earlier queued snapshots below are historical.

#### CPU-only completion recovery after successful fresh-dev scoring

The same GPU parent/worker Completed;baseline coverage188821 candidates,
all12 metrics/three contrasts/input hashes/both heads/128 spots independently
verified October26 01:29UTC. Subsequent CPU hybrid calculation hit its
180-second timeout before writing output;the GPU job did not fail.
Root cause:ranking aggregation converted the entire Python label list
to an array inside each impression loop,repeated for all three arms.
Move identical array conversion outside the loop;preserve grouping,
tie handling,metric formulas and averaging. Regression coverage checks
one conversion per full input and exact list/array/tied-score parity.
Record the CPU recovery source separately from the unchanged GPU source.
Recover only saved CPU scoring/verification;no GPU rerun,new training,
head/population/window change,reselection or deadline/budget reset.
Original controller error evidence remains preserved.

#### Verified official-dev result: behavioral hybrid passes promotion gate

CPU recovery source60cd6e4940f0f6924a1b7dfefcfb1012795010d3;unchanged GPU
source7c6aa8b3948c7e759125570f732e74dabc148e25. Recovery exited0,
independently verified October26 01:41UTC /09:41UTC+8. Same official
MIND_small dev26001-31000,5000 impressions/188821 candidates.

| Official MIND_small dev condition | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| Frozen JEV | 0.62893530 | 0.29733073 | 0.32481915 | 0.38229220 |
| Retained131071 text readout | 0.66104433 | 0.32175543 | 0.35096800 | 0.41470848 |
| Fixed behavioral pairwise hybrid | 0.67411689 | 0.32969184 | 0.36348107 | 0.42513257 |

| Primary hybrid-minus-retained | Mean difference | Paired impression-bootstrap95% CI |
|---|---:|---|
| AUC | +0.01307257 | [+0.00955989,+0.01672927] |
| MRR | +0.00793641 | [+0.00388142,+0.01193081] |
| nDCG@5 | +0.01251307 | [+0.00788235,+0.01694465] |
| nDCG@10 | +0.01042409 | [+0.00651074,+0.01421139] |

2048 paired impression resamples,seed42,unadjusted. All four primary
intervals positive,no secondary regression;predeclared promotion gate
passes. Retain the hybrid as the new research candidate:existing text
head279ef6c4 plus correction32e8af31 plus populationba4d7855. The4-feature
correction alone is not a2560-dimensional head replacement. Supporting
hybrid-minus-frozen AUC+0.04518160[+0.03809412,+0.05271782];
retained-minus-frozen+0.03210903[+0.02489327,+0.03933017].

Full features/correction math/exact coverage/all12 metrics/all three
paired contrasts independently reconstructed. No dev fitting,reselection,
count updates,window pooling,GPU rerun or budget reset. Original timeout
snapshot/log preserved. Proof `jev-behavior-dev-independent-verification.json`;
resultb5d6aba3be6c54047d3f288b71ca4ed580907869d3ee22e5f83af2bc99ea7d47;
raw19f1f6f07d3f4370c8953b1c9028bdbafea68ba3c66a489af87148117feacc77.
This remains partial official-dev evidence for a131071-pair subset-trained
model,not full-dev/test,MIND_large,SOTA or the requested full-training fit.
Full training audit:156965 impressions,236344 clicked/5607100 unclicked
candidates,5843444 total;existing negative sampling yields469264 fit rows.
Full-training coverage/refit was not done at this readout;the frozen follow-up below resolves its scope.

Verified-dev publication completed:the full authenticated shared-site
snapshot returned `Status: Succeeded` / `Deployment Complete`.
Receipt `jev-behavior-final-publication-20261026.log`;binding proof
`jev-behavior-final-publication-verification.json` validates all12
official-dev cells,four primary intervals,source/model/population hashes,
CPU-only recovery and completed-job feed. Training-selection Table21 is
collapsed and clearly separate from official-dev Tables22-23. The GPU
run's feed metric is its retained-head AUC0.66104433,not the CPU hybrid
AUC0.67411689;the hybrid result is explicitly reported separately.
No signed-in browser rendering verification is claimed.

#### Full official-training refit preregistration (2026-10-26)

The user requests full training and autonomous continuation. Use **all156965
official MIND-small training impressions**,retaining the existing seed42,
negative_ratio1/hard_negative_ratio0.5 sampler:469264 fitting rows,236344
clicked and232920 sampled unclicked candidates. Remove the131071-row cap
and all256 training-holdout exclusions. This is not a5.84M-all-candidate
negative expansion. All3212 single-class sampled impressions train the
binary text head and contribute population counts/feature normalization;
153753 mixed-class impressions supply the pairwise correction contrasts.
Five-fold impression-hash cross-fitting excludes each fitting impression's
entire fold from its own popularity features,not a training-data holdout.

Freeze one recipe,no training-quality selection:binary text residual L2=.01,
then pairwise four-feature correction L2=.01,normalized with every fitting
row;five folds,smoothing20,seed42,float64 LBFGS/max100iterations,four CPU
threads. Backbone,prompt/history30/abstracts/negative sampler stay unchanged.
Population includes every official training impression. Publish quality only
from official dev,never training loss or reused training-holdout scores.

Exact uncapped row SHA256:
`2771ba130ec47c6e27dea318641d25ea671529c03727b59b04e9f563319abcde`;
this is the original Windows CRLF audit file. The identical preregistered
row IDs/order/labels serialized with Linux LF have SHA256
`b4dba4f73cdf5878ba1d67582d8b81899240e13c3670f38b8b26ee04605af254`.
Both bindings are retained;no training identity or order changes.
full-population SHA256:
`993852a23416f436ed1dd0d7f6a8fde75dfb884b69da9c6b3242a1d00e67d06f`.
Reuse262143 checksum-pinned doubled-cache rows and765 matching former-holdout
rows. Hash whole caches and replay16 frozen hidden features/margins per cache
before reuse. Extract exactly206356 missing rows in four sequential stages:
65536/65536/65536/9748. These shard boundaries are storage/compute limits,
not training subset limits;assemble and fit all469264 rows only after
complete,duplicate-free coverage. Keep immutable committed source throughout.

Budget:at most four training-feature jobs plus one fresh-dev job,each one
Standard Ranking A100 with21600s worker timeout and a six-hour deadline
from its parent creation,including queue time. At most one allocation at
a time;maximum30hours aggregate per-parent wall-clock budget,no resets or
duplicate retries after uncertain submissions. CPU-only recovery may reuse
completed GPU artifacts without changing heads,features,windows or deadlines.

Fresh official dev**31001-36000**,5000 impressions/**189095 candidates**;
impression-order SHA256:
`dde64229686bfc8a93472750ca788a2c65d7e5de8b23ed5e2fa2bb63b79a01c7`.
One shared frozen-feature pass compares new full text head against retained
text279ef6c4 and frozen JEV. CPU then applies the new full correction/full
population to the former and the **old**correction32e8af31/populationba4d7855
to the latter. Primary is **full-data hybrid minus retained subset hybrid**,
not hybrid versus uncorrected text. Independently verify coverage,features,
both head mathematics,all12 metrics and all3 contrasts;2048 paired-impression
bootstrap resamples,seed42,unadjusted95% intervals. Promote only if the
primary AUC interval is strictly positive;report all secondary regressions.
No fitting,reselection or pooling earlier dev windows. Still partial
MIND-small dev,one seed,not full-dev/test,MIND-large or SOTA evidence.

Implementation and preflight are complete;source
`e44b2e1147b5b7d637dc37d354ede943ba84ec57`. Actual stage0 parent
**`shy_celery_p9fd7cc41x`** was created2026-10-26 02:11:24UTC;
fixed deadline08:11:24UTC. Server source,stage0,exact old-cache/former-holdout
paths,output root,one Standard Ranking A100 and21600s timeout verified.
Submission alone does not establish that a worker is running. Campaign
controller:`jev-full-refit-controller`,PID63568,persistent state
`jev-full-refit-campaign.json`;immutable source worktree
`jev-full-refit-source-e44b2e11`. Aggregate deadline2026-10-27 08:09:23UTC.
Do not resubmit/reset deadlines. Private preflight proof:
`jev-full-training-preflight.json`. Current published
quality remains the independently verified subset hybrid AUC0.67411689
on dev26001-31000 until this new full-training/dev campaign actually ends.

Submission publication complete2026-10-26 02:22UTC:parent
`shy_celery_p9fd7cc41x` and worker
`ca8a0e8f-4a36-413e-9f3e-4308b58a95e3` bothRunning;exact source checkout
seen,source/environment setup still in progress,no completed features or
full-training quality claimed. Full authenticated shared-site snapshot
returned`Status:Succeeded`/`Deployment Complete`;receipt
`jev-full-training-submission-publication-20261026.log`,binding proof
`jev-full-training-submission-publication-verification.json`.
Preserved the current shared-page layout and added official-dev12-metric
and full-training protocol sections without reverting other page changes.
All other shared assets/auth configuration remained unchanged. No signed-in
browser rendering check claimed. Feed training metric is empty.

Autonomous continuation confirmed2026-10-26:the existing controller chains
all four feature stages,full fitting,independent training verification and
the fresh official-dev comparison without another user approval. Stage0
was actively extracting21776/65536 rows at03:15:36UTC,not waiting idle.
An offline verification smoke covered the actual469264 training identities/
full population and rejected deliberately incomplete coverage,using
synthetic zero heads only;this is verifier validation,**not model quality**,
and created no GPU jobs or real campaign artifacts.
Proof:`jev-full-verifier-offline-smoke.json`.
Final-report generation covers gains,uncertainty and negative confidence
intervals;two fixtures passed. The duplicate-protected publication watcher
`jev-full-final-publication-watch` observes the existing campaign and will
recheck the independent final proof,then publish actual official-dev metrics/
intervals and the promotion decision. It cannot publish before all five
existing campaign jobs complete and official-dev verification succeeds.
Watch state:`jev-full-final-publication-watch.json`;final receipt target:
`jev-full-training-final-publication-20261026.log`. No new GPU allocation,
source/head/window change,deadline reset or full-training score claimed.

Stage0 GPU completed2026-10-26 04:47UTC:65536 new features,328444/469264
total fitting rows now covered. The controller stopped afterward at a
row-metadata digest assertion,not a GPU/model failure:Windows preregistration
used CRLF while the unchanged Linux worker wrote LF. CPU-only recovery
rehashes the original untouched local files,derives their exact LF JSON,
and requires the entire remote byte payload and worker digest to match;
all65536 IDs/order/labels verified,not just a semantic/count fallback.
Stage0 feature SHA256:
`17a6f84a9ccd6ef51d6935578b1d318dfccec55dbe473dda42db1cff0f5cf31c`.
Original state,traceback and private verifier sources preserved separately.
Proof:`jev-full-training-newline-recovery-verification.json`;all five
Windows/Linux row-file bindings retained. Seven targeted fixtures passed,
including rejection of changed IDs,labels,order,digests and other formatting.
The full469264-row offline verifier also passed with Linux remote bytes
and rejected incomplete coverage;synthetic zero heads,not model quality.
Proof:`jev-full-verifier-linux-smoke.json`.

The existing campaign resumed at planned stage1;stage0 is not rerun.
Recovery controller shell:`jev-full-refit-controller-recovery`.
Actual stage1 parent`musing_sprout_blpm5x03ms`,worker
`30894111-0833-43e8-a6c8-e8319cfee423`;server source,stage/cache paths
and one Standard Ranking A100/21600s verified. Created2026-10-26
05:00:57UTC,fixed deadline11:00:57UTC. Initial parentRunning,workerQueued;
that is not yet feature extraction. Replacement publisher PID63796
confirmed responsive,observing this same resumed campaign.
Original per-parent and aggregate deadlines,immutable source,heads,
training recipe and fresh-dev window are unchanged. Original publication
watcher correctly stopped on the controller error;its evidence is retained.
Replacement watcher uses a new duplicate-protected state:
`jev-full-final-publication-recovery-watch.json`,shell
`jev-full-final-publication-recovery-watch`;final log
`jev-full-training-final-publication-recovery.log`.
Recovery controller PID61768 and replacement publisher PID63796 confirmed
responsive. Recovery webpage refresh completed with feed05:05UTC:
stage0Completed/328444 rows covered,next actual stage1 parentRunning
(worker initiallyQueued),training quality empty. Full authenticated shared
snapshot returned`Status:Succeeded`/`Deployment Complete`;receipt
`jev-full-training-recovery-publication-20261026.log`,binding proof
`jev-full-training-recovery-publication-verification.json`.
Original submission publication proof and original failure evidence retained;
no signed-in browser rendering check claimed.

Stage1 completed2026-10-26 08:01UTC and its entire65536-row metadata
was independently verified. Coverage is393980/469264 rows;75284 remain.
Feature SHA256:
`5635bc0433fb0e2d9dff4fbae6e20c0f5aee473048143792cae23937f7cc518b`.
The controller automatically submitted the registered stage2:
parent`purple_dress_s2zryf6ck7`,worker
`5c616a1a-bf65-4e2b-bb8f-e6689746b5ca`,created08:11:59UTC,
fixed deadline14:11:59UTC. Both were Failed by08:24:42UTC;the monitor
obtained no worker log. The controller and guarded final publisher stopped
and their error evidence is preserved. Service-side failure diagnosis is
in progress;no replacement allocation or deadline/budget reset. Do not
blindly resume this terminal parent. Both completed feature stages remain
usable. No full-training fit or fresh full-training dev result exists yet.

Cloud diagnosis identifies a pre-execution infrastructure failure:
`SystemError`/`KnownSystemError`,container`pre-caching-job`,`ErrImagePull`,
missing registry layer
`sha256:998e2433689d901095a5164b1e2ab60c49a3a27281a1b5979560744cb989cbbb`.
Worker start/end are both08:23:20UTC,recorded compute/duration00:00:00,
no effective start or user logs. Diagnostic evidence:
`jev-full-refit-stage-02-cloud-diagnostics.json`.
Do not attribute this failure to the training code or claim zero charges.

Explicit infrastructure-only budget amendment:allow exactly one additional
parent submission for this confirmed terminal,zero-compute image-pull
failure,subject to a fresh service-side error check and HTTP404 for the
unused stage2 output directory. At most six parent submissions including
the failed pre-execution attempt;still four successful training stages and
one dev pass,one Standard Ranking A100 at a time. Preserve the original
stage2 deadline2026-10-26 14:11:59UTC and original aggregate deadline
2026-10-27 08:09:23UTC;no extra wall-clock allowance or automatic retry.
Model,source,environment version5,data,features,heads,recipe and dev window
are unchanged. A second infrastructure failure stops the campaign.
Original state,failed phase,submission receipt and watcher errors remain
archived. Eight controller fixtures pass,including exact failure eligibility,
absent-output checks,and deadline preservation;not model-quality evidence.

The infrastructure replacement was registered08:47UTC after fresh terminal
cloud checks and GETFILESTATUS_HTTP404 for the stage2 directory. Original
state is archived in
`jev-full-refit-campaign-before-stage2-infrastructure-replacement.json`.
Actual replacement parent`frosty_spoon_4xzk0mw81z`;source remains`e44b2e1`.
Receipt:`jev-full-refit-stage-02-infra-retry-01-submit.log`.
Controller:`jev-full-refit-controller-infrastructure-recovery`,log of the
same basename. No second retry is authorized by this amendment.
The failure-status webpage snapshot deployed successfully,with actual
393980-row coverage and the zero-compute failure;receipt
`jev-full-training-stage2-failure-status-only-publication-20261026.log`,
proof`jev-full-training-stage2-failure-publication-verification.json`.
An earlier feed-generation attempt was stopped before deployment because
SDK Studio-URL decoration repeatedly attempted the broken Windows CLI
credential path. Private publication now reads the same AML RunHistory
statuses directly with the existing working credential,avoiding URL
decoration;public authentication/deployment code is unchanged.
Three publication fixtures pass,including restoration after failure.

Server readback verifies replacement source/cache paths/stage and one
Standard Ranking A100/21600s. Created2026-10-26 08:48:06UTC;the replacement
deadline is still14:11:59UTC,not six new hours from this creation.
Worker`76c17e08-785c-43b4-a7d3-4d6073a6b024` initiallyQueued while the
parent isRunning;do not claim execution before worker evidence. Controller
PID83272 and guarded final publisher PID24928 confirmed responsive.
New watcher state:`jev-full-final-publication-infrastructure-watch.json`,
shell same basename,log`jev-full-training-final-publication-infrastructure.log`.
It cannot publish full-training quality before all actual stages and the
registered official-dev evaluation complete with independent verification.
Infrastructure-recovery status deployment returned`Status:Succeeded` and
`Deployment Complete`;receipt
`jev-full-training-infrastructure-recovery-publication-20261026.log`,
proof`jev-full-training-infrastructure-recovery-publication-verification.json`,
which also binds the original stage/aggregate deadlines and archived state.
No signed-in browser rendering verification claimed.

The one infrastructure replacement succeeded without changing the image
or model:stage2 Completed with65536 new rows,459516 total,exact row metadata
verified. Feature SHA256:
`a3c09341fcdef4f91e26aaef2ddee75b1c9adf0655796f79611eaa3fdefe6db6`.
Stage3 parent`tidy_night_p3t3smvnym`,worker
`5c53ab1c-d326-410d-92c3-ee1cc2080f25`,created11:30:32UTC,deadline17:30:32UTC,
Completed. All9748 final rows verified,469264/469264 covered.
Final feature SHA256:
`c4ddaf77fd6495bfe3c6f2667fb7b497cd2c0692be9dee8119fadda2cdc933f1`.
The worker fitted both full-data heads12:28UTC:
text`5392ad5738d09a9e0df9a502754ceb833f807f9f4c8b9fbc6a7ceded1adf13bc`,
behavior`48b04d256401751de412291bbc6e78265176049e1f62cb536c7ceab5fd22cd52`.
These are not yet independently accepted fitted artifacts or quality results.

The CPU verifier stopped at`population_sha256`12:35UTC. Full-file diagnosis
finds exactly three ZIP central-header creator-OS bytes,Windows0/Linux3,
at offsets2435227/2435285/2435341. Every NPY payload byte and every value in
item_ids/counts/totals is identical,including shapes/dtypes. Original
preregistered Windows population SHA993852a23416f436ed1dd0d7f6a8fde75dfb884b69da9c6b3242a1d00e67d06f
is preserved;the exact Linux archive SHA is
`b2bdeb1b7d4d1c090aee9f8b216a00275c89ba810e07b6fad093219e97ae9f95`.
Evidence:`jev-full-training-population-diagnostics.json`;original failed
state/log/verifier/watcher preserved underoriginal-population-error names.
CPU recovery derives Linux bytes only from the checksum-bound original,
changes only those three documented creator-OS fields,and requires the
entire remote file and worker digest to match. No arbitrary archive
normalization,semantic-only checksum fallback,head refit,source/data/recipe/
window change,training rerun or budget/deadline reset. Full independent
population counts,all behavioral features,scores,losses and128 cache/text
spots must still pass before the one registered fresh official-dev job.
The final publication watcher correctly stopped on this verifier error.
No full-training dev result exists.

Nine targeted controller fixtures pass,including the actual three observed
byte offsets,known Linux digest,unchanged original bytes,unsupported creator
codes and tampering rejection. The offline full469264-row verifier passes
with Linux population/row bytes and rejects incomplete coverage;synthetic
zero heads,not model quality. Proof:
`jev-full-verifier-population-linux-smoke.json`.
CPU-only recovery registered12:49UTC with all four successful stages reused.
Controller shell/log:`jev-full-refit-controller-population-recovery(.log)`;
recovery proof:`jev-full-training-population-recovery-verification.json`.
The full independent fitted-artifact check is running;the controller will
submit the single registered fresh-dev job only after that check succeeds.
Replacement guarded publisher shell/state:
`jev-full-final-publication-population-watch(.json)`,log
`jev-full-training-final-publication-population.log`. Prior controller/
watcher errors remain preserved. All-features webpage refresh deployed
`Status:Succeeded`/`Deployment Complete`;receipt
`jev-full-training-all-features-publication-20261026.log`,proof
`jev-full-training-all-features-publication-verification.json`.
All original deadlines and scientific settings remain unchanged.

The resumed verifier passed the complete population-count/behavioral-feature/
score/loss checks and reached independent source-cache spot reads. At
12:59UTC an Azure Data Lake HTTPS ReadTimeout interrupted one bounded range
request;this was not a numerical,coverage or model mismatch. No dev job
had been submitted. Original state/log/verifier/watcher error preserved
underoriginal-range-timeout names.

The verifier now retries only the identical read-only OPEN/GETFILESTATUS
request,at most four attempts with1/2/4-second backoff,for Timeout,
ConnectionError or HTTP408/429/500/502/503/504. Each failed attempt is logged;
exhaustion rethrows. Nontransient errors,untrusted redirects,checksums,
truncated ranges and all numerical tolerances remain strict;no alternate
bytes,skipped spots,success-shaped fallback or new training allocation.
Eleven targeted fixtures pass,including exact repeated arguments,bounded
exhaustion,read-only restriction and immediate nontransient failure.
CPU recovery registered13:05UTC with all four completed training stages
preserved;controller shell/log:
`jev-full-refit-controller-transport-recovery(.log)`.
All original parent and aggregate deadlines remain unchanged.

#### Full-training campaign final official-dev result (October27 UTC+8)

The full fitted-artifact verification passed,including all counts,behavioral
features,hybrid scores,both losses and128 cache/text-head spots. Training
used all156965 official MIND-small impressions/469264 sampled rows with no
training holdout. The single fresh official-dev pass,parent
`zen_oven_dxw716bx8c`,worker`110eab85-3df5-4b7a-947c-9d0b5ba0aa78`,
Completed before its creation-based six-hour deadline. Official dev
impressions31001-36000:5000 impressions/189095 candidates,not training
holdout,full-dev,test,MIND-large or leaderboard evaluation.

| Frozen arm | AUC | MRR | nDCG5 | nDCG10 |
|---|---:|---:|---:|---:|
| Frozen JEV |0.62488338|0.29519263|0.32302276|0.38354872|
| Retained subset hybrid |0.67598829|0.32733137|0.36660216|0.42598350|
| Full-training hybrid |0.67203545|0.32918194|0.36906438|0.42860420|

Primary full-training-minus-retained AUC delta**-0.00395284**,paired
impression-bootstrap95%CI**[-0.00675987,-0.00115384]**,entirely negative.
**Do not promote the full-training hybrid;retain the subset hybrid.**
Secondary deltas:MRR+0.00185057,CI[-0.00165696,+0.00514166];
nDCG5+0.00246222,CI[-0.00127959,+0.00609273];
nDCG10+0.00262070,CI[-0.00043716,+0.00562322]. All three intervals cross
zero;their positive point estimates do not establish improvements.
2048 paired-impression resamples,seed42,unadjusted intervals;one training
seed. No dev fitting,reselection,pooling windows or post-hoc promotion.
The retained hybrid's0.67598829 here is a same-window comparison;do not
compare the full model against its earlier0.67411689 ondev26001-31000.

All12 metric cells,both full correction/feature matrices,exact coverage
and three paired contrasts independently reconstructed. Controller exited0.
Quality proof is stored in`jev-full-refit-campaign.json`;final result
SHA256`b1add99e0b2a9179fb614e4ac67ed9b878bfa530e26dc0d2f342fe2b0ec4a43f`,
raw-score SHA256
`f470c7eab2001ae7ff66b2f366299690371e00201715a43faddecebb7877945d`.
Guarded final publication completed2026-10-27 03:25UTC+8 with
`Status:Succeeded`/`Deployment Complete`. Watch state:
`jev-full-final-publication-transport-watch.json` reports`final_result_published`.
Receipt:`jev-full-training-final-publication-transport.log`;binding proof:
`jev-full-training-final-publication-verification.json`.
HTML SHA256`e0402848f853f62a85a615828f8c3f42228d4c706b9b14dfd0f92fa674b6090e`;
feed SHA256`f80926082c343e11114c9818648b32592112b7e51b63858d33501917f30aee15`.
Other shared-site assets and authentication remained unchanged. No
signed-in browser rendering verification claimed. Campaign complete with
a negative primary result;no additional model/data/recipe tuning or GPU
job is automatically authorized by this result.

#### Post-hoc regression diagnosis:existing frozen arms only

CPU-only accounting on the same already-exposed dev31001-36000 replays five
existing arms from checksum-bound raw scores. All20 metric cells reproduce
their verified results,and the original hybrid contrast/intervals reproduce
exactly. No candidate fitting,score mixtures,tuning,GPU job or promotion.

| Existing component | Subset-trained | Full-trained | AUC change |
|---|---:|---:|---:|
| Text readout AUC,before behavior |0.66343769|0.66094967|-0.00248802|
| Behavioral AUC lift above own text baseline |+0.01255060|+0.01108578|-0.00146483|
| Hybrid AUC |0.67598829|0.67203545|-0.00395284|

Exploratory text-scaling95%CI[-0.00485889,-0.00009004];behavioral-lift
change95%CI[-0.00397179,+0.00097415]. Both frozen behavioral corrections
still improve their own text baselines,with positive AUC intervals.
The regression is already present before the behavioral correction;the
behavioral-lift reduction is not independently clear at this interval.
The sum is an accounting identity,not causal isolation:the populations,
corrections and text margins differ. These are post-hoc,unadjusted,
2048 paired-impression bootstrap intervals,seed42,not independent new
confirmation. They identify neither regularization,objective,data cohort
nor optimization as the cause. Do not tune/select a replacement using
these exposed dev labels. The retained candidate and primary rejection
remain unchanged. Artifact:`jev-full-regression-diagnosis.json`;
generator:`diagnose_jev_full_regression.py` in the session files.

### Phase 2: RL Fine-tuning

| Exp ID | Status | Base Model | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|------------|--------|-----|-----|--------|---------|-------|
| **R2.1** | ⬜ Pending | Best P1.x | REWARD_TYPE=asymmetric | - | - | - | - | Asymmetric reward |
| **R2.2** | ⬜ Pending | Best P1.x | KL_COEF=0.05 | - | - | - | - | Lower KL penalty |
| **R2.3** | ⬜ Pending | Best P1.x | TOTAL_EPOCHS=2 | - | - | - | - | More RL epochs |
| **R2.4** | ⬜ Pending | Best P1.6 | RL on 8B model | - | - | - | - | RL on larger model |

### Phase 3: Ensemble & Cascade

#### Multi-Pointwise Ensemble (MINDlarge, ready to run now)

> **Design principle**: use an **odd number of models** (3, 5, 7...) to enable clean majority voting (no ties). Score averaging is used for 2-model ensembles but majority vote is preferred for 3+.

| Exp ID | Status | Models | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|--------|--------|-----|-----|--------|---------|-------|
| **E3.A** | ✅ Completed | L1.3 + L1.7 | equal weights | **0.7152** (dev) / **0.7144** (test) 🏆 | 0.4032 | 0.3896 | 0.4519 | **LEADERBOARD BEST**; test≈dev (gap 0.0008 = no overfit) |
| **E3.B** | ⬜ Pending | L1.3 + L1.2 | equal weights | - | - | - | - | Abstract(0.7049) + EP7(0.6932) |
| **E3.C** | ✅ Completed | L1.3 + L1.7 + L1.2 | equal weights | 0.7105 | 0.4006 | 0.3878 | 0.4504 | Worse than E3.A (0.7152); L1.2 dilutes ensemble |
| **E3.D** | ⬜ Pending | L1.3 + L1.7 + L1.2 | weights 1.0 0.8 0.8 | - | - | - | - | L1.3-heavy weighting |
| **E3.E** | ✅ Completed | L1.3 + L1.7 + L1.9 | equal weights | **0.7071** | 0.4002 | 0.3863 | 0.4489 | +0.0022 AUC over L1.3 solo; L1.9=checkpoint-3584 |
| **E3.F** | ⬜ Pending | L1.3 + L1.6 + L1.7 | majority vote (3 models) | - | - | - | - | First majority-vote ensemble; odd number; run after L1.6 completes |
| **E3.G** | ⬜ Pending | L1.3 + L1.6 + L1.7 + L1.2 + L1.9 | majority vote (5 models) | - | - | - | - | 5-model majority vote; run after L1.6 completes |

#### Snapshot Ensemble (checkpoints from same training run, zero extra training cost)

> **Design principle**: pick checkpoints spaced across epochs (not consecutive saves) for maximum diversity. Use odd number of checkpoints.

| Exp ID | Status | Models | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|--------|--------|-----|-----|--------|---------|-------|
| **E3.S1** | ⬜ Pending | L1.3 ep3 + ep4 + final | majority vote (3 ckpts) | - | - | - | - | Snapshot ensemble from L1.3 run; free AUC gain |
| **E3.S2** | ⬜ Pending | L1.6 ep3 + ep4 + final | majority vote (3 ckpts) | - | - | - | - | Snapshot ensemble from L1.6 run; run after L1.6 completes |
| **E3.S3** | ⬜ Pending | L1.3 snapshots + L1.6 final | majority vote (3 models) | - | - | - | - | Cross-model + snapshot hybrid |

#### PW + Ranking Ensemble (requires ranking model training first)

| Exp ID | Status | Models | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|--------|--------|-----|-----|--------|---------|-------|
| **E3.1** | ⬜ Pending | PW + RK | ALPHA=0.7 | - | - | - | - | More pointwise weight |
| **E3.2** | ⬜ Pending | PW + RK | ALPHA=0.5 | - | - | - | - | Balanced ensemble |
| **E3.3** | ⬜ Pending | PW + RK | ALPHA=0.3 | - | - | - | - | More ranking weight |
| **E3.4** | ⬜ Pending | PW + RK | TOP_K=15 | - | - | - | - | Cascade top-15 |
| **E3.5** | ⬜ Pending | PW + RK | TOP_K=10 | - | - | - | - | Cascade top-10 |

### Phase 4: Multi-task Exploration

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **M4.1** | ⬜ Pending | Qwen3-1.7B | MINDsmall | POINTWISE_RATIO=0.7 | - | - | - | - | 70% pointwise |
| **M4.2** | ⬜ Pending | Qwen3-1.7B | MINDsmall | POINTWISE_RATIO=0.8 | - | - | - | - | 80% pointwise |
| **M4.3** | ⬜ Pending | Qwen3-1.7B | MINDsmall | Two-stage: PW(2ep)+RK(3ep) | - | - | - | - | Two-stage training |

### Phase 5: Advanced Experiments

| Exp ID | Status | Model | Approach | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|----------|-----|-----|--------|---------|-------|
| **A5.1** | 🔄 Running | Qwen3-1.7B | CoT RL | - | - | - | - | Crashed previously (W&B d00m4zay) due to missing UAI + wrong recall VC; fixed: UAI added, switched to ranking VC, SLA=Standard; resubmitted 2026-06-08 from L1.3 checkpoint; run ID: sad_gold_0hgz2zlrgm |
| **A5.2** | ⬜ Pending | Qwen3-14B/32B | Large model | - | - | - | - | Resource intensive |
| **A5.3** | ⬜ Pending | Qwen3-1.7B | Data augmentation | - | - | - | - | Research exploration |

---

## 📝 How to Update This Table

After completing an experiment:

1. **Update Status**: ⬜ Pending → 🔄 Running → ✅ Completed / ❌ Failed
2. **Fill in Results**: AUC, MRR, nDCG@5, nDCG@10 (in percentage)
3. **Add Notes**: Training time, observations, issues, checkpoint path
4. **Highlight Best**: Use 🏆 or **bold** for best results

### Example Update:

```markdown
| **P1.1** | ✅ Completed | Qwen3-1.7B | MINDsmall | NEG_RATIO=3.0 | **69.85%** 🏆 | 33.45% | 37.89% | 43.67% | Best AUC! 8h train, checkpoint: sft_mind_pointwise_large_neg3.0 |
```

### Quick Update Template:

Copy and paste this template when updating:

```markdown
| **[EXP_ID]** | ✅ Completed | [MODEL] | [DATASET] | [CONFIG] | [AUC]% | [MRR]% | [nDCG@5]% | [nDCG@10]% | [NOTES] |
```

---

## 🖥️ Manual Evaluation Commands

> Run these on the compute node after `conda activate MiniOneRec` and `cd /home/aiscuser/MiniOneRec`.
> Set the shared mount path first:
> ```bash
> SHARES="/scratch/azureml/cr/j/ef9a7f2099e947cab5fb282f38f685e3/cap/data-capability/wd/INPUT_msndni/shares"
> OUTPUT_DIR="$SHARES/users/wuc/output_dir"
> MIND_SMALL="$SHARES/users/wuc/data/MIND_small"
> MIND_LARGE="$SHARES/users/wuc/data/MIND_large"
>
> # Helper: picks latest checkpoint (final_checkpoint may be corrupted if job was interrupted)
> latest_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | tail -1; }
> ```

### Phase 1: MINDsmall

```bash
# P1.0 — ep5 default baseline
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.1 — NEG=3.0
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg3.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.2 — EP=7
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.3 — ABSTRACT
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.4 — HIST=50
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist50_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=50 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.5 — NEG=3.0, EP=7, HIST=50
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=50 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# P1.7 — SUBCATEGORY
D=$OUTPUT_DIR/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_subcat_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 USE_SUBCATEGORY=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev
```

### Phase 1L: MINDlarge

```bash
# L1.1 — NEG=3.0, HIST=30
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg3.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.2 — EP=7
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.3 — ABSTRACT
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# E1.1 — L1.3 checkpoint, eval WITHOUT abstract (cross-condition A/B test)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 USE_ABSTRACT=0 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_no_abstract_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# E1.2 — L1.3 checkpoint, eval WITH abstract (control, re-eval after pipeline fix)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_with_abstract_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# E1.3 — L1.9 (base, abstract-trained) checkpoint, eval WITHOUT abstract
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30_abstract
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=0 USE_ABSTRACT=0 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_no_abstract_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.4 — HIST=50
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist50_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 MAX_HISTORY=50 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.5 — NEG=3.0, EP=7, HIST=50
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 MAX_HISTORY=50 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.6 — 8B model (BATCH_SIZE=4 for VRAM)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-8B_bs256_ep5_neg2.0_hist30_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=4 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.7 — Base model (no chat template)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev

# L1.8 — Abstract + EP=7 (TRAINING) [micro-batch-size 4→2 to fix OOM with abstract+cutoff8192]
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_l1-8_abstract_ep7 `
  --display-name "L1.8: Abstract + EP=7 (large)" `
  --model-path Qwen/Qwen3-1.7B `
  --data-root shares/users/wuc/data/MIND_large `
  --output-root shares/users/wuc/output_dir `
  --batch-size 256 `
  --micro-batch-size 2 `
  --num-epochs 7 `
  --neg-ratio 2.0 `
  --max-history 30 `
  --use-chat-template 1 `
  --use-abstract 1 `
  --run-eval 1

# L1.8 — EVAL (after training)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $D/final_checkpoint dev

# L1.9 — Abstract + Base model (TRAINING)
python pipeline/run_pipeline.py `
  --experiment-name mind_sft_l1-9_abstract_base `
  --display-name "L1.9: Abstract + Base model (large)" `
  --model-path Qwen/Qwen3-1.7B-Base `
  --data-root shares/users/wuc/data/MIND_large `
  --output-root shares/users/wuc/output_dir `
  --batch-size 256 `
  --micro-batch-size 4 `
  --num-epochs 5 `
  --neg-ratio 2.0 `
  --max-history 30 `
  --use-chat-template 0 `
  --use-abstract 1 `
  --run-eval 1

# L1.9 — EVAL (after training)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30_abstract
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $D/final_checkpoint dev
```

### Phase 4M: Multitask

```bash
# M4.0 — multitask pw=0.7, ep3
D=$OUTPUT_DIR/sft_mind_multitask_small_Qwen3-1.7B_bs256_ep3_pw0.7_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=small DATA_ROOT=$MIND_SMALL USE_CHAT_TEMPLATE=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $(latest_ckpt $D) dev
```

```powershell
# M4.1 — multitask pw=0.5, ep5, MINDlarge + Abstract (AML Pipeline)
python pipeline/run_pipeline.py `
  --model-path Qwen/Qwen3-1.7B `
  --data-root shares/users/wuc/data/MIND_large `
  --output-root shares/users/wuc/output_dir `
  --batch-size 512 --micro-batch-size 4 `
  --num-epochs 5 `
  --use-abstract 1 `
  --pointwise-ratio 0.5 `
  --run-eval 1 `
  --experiment-name mind_sft_training `
  --display-name "M4.1 Multitask Large Abstract EP5 PW0.5"
```

---

## 🔧 Pipeline Support Matrix

| Experiment | Azure ML Pipeline | Local Scripts | Notes |
|------------|-------------------|---------------|-------|
| **P1.1** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.2** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.3** | ✅ Training + Eval | ✅ Available | Full pipeline support (use_abstract) |
| **P1.4** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.5** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.6** | ✅ Training + Eval | ✅ Available | Full pipeline support (4B model, micro_bs=2) |
| **R2.1-R2.4** | ❌ Training / ✅ Eval | ✅ Required | RL not yet in pipeline |
| **E3.1-E3.5** | ❌ Not applicable | ✅ Required | Ensemble/cascade local only |
| **M4.1** | ✅ Training + Eval | ✅ Available | Pipeline now supports multitask (--pointwise-ratio) |
| **M4.2-M4.3** | ✅ Training + Eval | ✅ Available | Use --pointwise-ratio <1.0 |
| **A5.1** | ✅ Training + Eval | ✅ Available | Full pipeline support (CoT RL) |
| **A5.2** | ✅ Training + Eval | ✅ Available | Full pipeline support (14B/32B) |
| **A5.3** | ❌ TBD | ❌ TBD | Research exploration |

**Legend:**
- ✅ = Supported
- ❌ = Not supported / Use local scripts
- TBD = To be determined

**Note**: All copy-paste ready commands are provided in the "Detailed Experiment Configurations" section below.

---

## Available Approaches

### Training Scripts Inventory

| Script | Approach | Best For |
|--------|----------|----------|
| `sft_mind_pointwise.sh` | Point-wise (Yes/No) | **Best AUC** |
| `sft_mind_pointwise_ds.sh` | Point-wise + DeepSpeed | Multi-node training |
| `sft_mind_ranking.sh` | Ranking (1/2/3...) | MRR/nDCG optimization |
| `sft_mind_multitask.sh` | Joint point-wise + ranking | Balanced metrics |
| `sft_mind_twostage.sh` | Point-wise → Ranking | Transfer learning |
| `rl_mind_pointwise.sh` | RL on point-wise | Polish best model |
| `rl_mind.sh` | RL on ranking | Ranking optimization |
| `rl_mind_cot.sh` | Chain-of-Thought RL | Interpretability |

### Evaluation Scripts Inventory

| Script | Format | Use Case |
|--------|--------|----------|
| `eval_mind_pointwise.sh` | Yes/No scoring | Point-wise models |
| `eval_mind_ranking.sh` | Multiple-choice | Ranking models |
| `eval_mind_ensemble.sh` | Score blending | Combine models |
| `eval_mind_cascade.sh` | Filter + rerank | Fast inference |
| `eval_mind_cot.sh` | CoT generation | CoT models |

---

## Experiment Phases

### Phase 1: Point-wise SFT Optimization (Priority: HIGH)

**Goal**: Push point-wise SFT from 69.39% → 70.5%+ AUC

| Exp ID | Description | Key Changes | Expected AUC |
|--------|-------------|-------------|--------------|
| P1.1 | More negatives | `NEG_RATIO=3.0` | 69.7% |
| P1.2 | Longer training | `NUM_EPOCHS=7` | 69.8% |
| P1.3 | Add abstracts | `USE_ABSTRACT=1` | 69.6% |
| P1.4 | More history | `MAX_HISTORY=50` | 69.7% |
| P1.5 | Combined best | Best of P1.1-P1.4 | 70.0%+ |
| P1.6 | Scale to 4B | `MODEL_PATH=Qwen3-4B-Instruct` | 70.5%+ |

### Phase 2: RL Fine-tuning (Priority: HIGH)

**Goal**: Gain +0.3-0.5% from RL optimization

| Exp ID | Description | Key Changes | Expected AUC |
|--------|-------------|-------------|--------------|
| R2.1 | Asymmetric reward | `REWARD_TYPE=pointwise_asymmetric` | +0.3% |
| R2.2 | Lower KL penalty | `KL_COEF=0.05` | +0.2% |
| R2.3 | More RL epochs | `TOTAL_EPOCHS=2` | +0.2% |
| R2.4 | RL on 8B model | From P1.6 checkpoint | +0.3% |

### Phase 3: Ensemble & Cascade (Priority: MEDIUM)

**Goal**: Gain +0.5-1.0% from model combination (NO training needed!)

| Exp ID | Description | Key Changes | Expected AUC |
|--------|-------------|-------------|--------------|
| E3.1 | Ensemble α=0.7 | More point-wise weight | 70.5%+ |
| E3.2 | Ensemble α=0.5 | Balanced | 70.3% |
| E3.3 | Ensemble α=0.3 | More ranking weight | 70.0% |
| E3.4 | Cascade K=15 | Top-15 reranking | 70.2% |
| E3.5 | Cascade K=10 | Top-10 reranking | 70.0% |

### Phase 4: Multi-task Exploration (Priority: LOW)

**Goal**: Explore alternative training strategies

| Exp ID | Description | Key Changes | Expected AUC |
|--------|-------------|-------------|--------------|
| M4.1 | Multi-task pw=0.7 | 70% point-wise | 68.5% |
| M4.2 | Multi-task pw=0.8 | 80% point-wise | 69.0% |
| M4.3 | Two-stage 2+3 | 2 pw epochs + 3 rank epochs | 68.0% |

### Phase 5: Advanced Experiments (Priority: EXPLORATORY)

| Exp ID | Description | Key Changes | Notes |
|--------|-------------|-------------|-------|
| A5.1 | CoT RL | `COT_STYLE=category` | Interpretability |
| A5.2 | Larger model | Qwen3-14B or 32B | Resource intensive |
| A5.3 | Data augmentation | Synthetic negatives | Research |

---

## Detailed Experiment Configurations

### Phase 1: Point-wise SFT Commands

> **Note**: Training pipeline now includes **integrated evaluation** by default (`--run-eval 1`).
> Results are saved to `{output_root}/sft_mind_pointwise_*/eval_results/`.
>
> **Debug Mode**: Add `--debug` to any command to keep the container running after errors for debugging.

```bash
# ============================================
# P1.1: More negatives (NEG_RATIO=3.0)
# ============================================
python3 pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-1_neg3-0 \
  --display-name "P1.1: More negatives (neg=3.0)" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 3.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug  # Uncomment to enable debug mode

# [Optional] Re-evaluate with standalone pipeline (if needed)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_sft_p1-1_neg3-0 \
  --display-name "P1.1: More negatives (neg=3.0)" \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg3.0_hist30_chat/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev

# ============================================
# P1.2: Longer training (NUM_EPOCHS=7)
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-2_ep7 \
  --display-name "P1.2: Longer training (ep=7)" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 7 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# [Optional] Re-evaluate with standalone pipeline (if needed)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_sft_p1-2_ep7 \
  --display-name "P1.2: Longer training (ep=7)" \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev

# ============================================
# P1.3: Add abstracts (USE_ABSTRACT=1)
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-3_abstract \
  --display-name "P1.3: With abstracts" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --use-abstract 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# [Optional] Re-evaluate with standalone pipeline (if needed)
# Note: checkpoint folder includes "_abstract" suffix due to USE_ABSTRACT=1
# python pipeline/run_eval_pipeline.py \
#   --experiment-name mind_sft_p1-3_abstract \
#   --display-name "P1.3: With abstracts" \
#   --model-path shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint \
#   --data-root shares/users/wuc/data/MIND_small \
#   --eval-type pointwise \
#   --split dev

# ============================================
# P1.4: More history (MAX_HISTORY=50)
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-4_hist50 \
  --display-name "P1.4: More history (hist=50)" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 50 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# [Optional] Re-evaluate with standalone pipeline (if needed)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_sft_p1-4_hist50 \
  --display-name "P1.4: More history (hist=50)" \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist50_chat/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev

# ============================================
# P1.5: Combined best settings
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-5_combined \
  --display-name "P1.5: Combined best settings" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 7 \
  --neg-ratio 3.0 \
  --max-history 50 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# [Optional] Re-evaluate with standalone pipeline (if needed)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_sft_p1-5_combined \
  --display-name "P1.5: Combined best settings" \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev

# ============================================
# P1.6: Scale to 4B model (first try before 8B)
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-6_4b \
  --display-name "P1.6: Scale to 4B model" \
  --model-path Qwen/Qwen3-4B-Instruct \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 2 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# ============================================
# P1.7: Add subcategory to prompt (USE_SUBCATEGORY=1) ✅ AUC=0.6767
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-7_subcat \
  --display-name "P1.7: With subcategory" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --use-subcategory 1 \
  --run-eval 0
```

### Phase 1L: MINDlarge Commands

> **Note**: Use `--run-eval 0` — integrated eval is unreliable; run manually after training.
> Set up shared path variables before running eval commands:
> ```bash
> SHARES="/scratch/azureml/cr/j/ef9a7f2099e947cab5fb282f38f685e3/cap/data-capability/wd/INPUT_msndni/shares"
> OUTPUT_DIR="$SHARES/users/wuc/output_dir"
> MIND_LARGE="$SHARES/users/wuc/data/MIND_large"
> ```

```bash
# ============================================
# L1.1: NEG=3.0 on large ✅ AUC=0.6883
# ============================================
# [Completed — checkpoint exists]

# ============================================
# L1.2: EP=7 on large ✅ AUC=0.6932
# ============================================
# [Completed — checkpoint-70144]

# ============================================
# L1.3: Abstract on large ✅ AUC=0.7049 🏆 BEST
# ============================================
# [Completed — final_checkpoint]

# ============================================
# L1.7: Base model on large ✅ AUC=0.6946
# ============================================
# [Completed — no chat template]

# ============================================
# L1.8: Abstract + EP=7 on large ⬜ Pending (retrain — prev run OOM'd mid-training)
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_l1-8_abstract_ep7 \
  --display-name "L1.8: Abstract + EP=7 (large)" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_large \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 2 \
  --num-epochs 7 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --use-abstract 1 \
  --run-eval 0

# L1.8 — Manual eval (after training completes)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_abstract_chat
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_CHAT_TEMPLATE=1 USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $D/final_checkpoint dev

# ============================================
# L1.9: Abstract + Base model on large ✅ AUC=0.6880
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_l1-9_abstract_base \
  --display-name "L1.9: Abstract + Base model (large)" \
  --model-path Qwen/Qwen3-1.7B-Base \
  --data-root shares/users/wuc/data/MIND_large \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-abstract 1 \
  --run-eval 0

# L1.9 — Manual eval (after training completes)
D=$OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30_abstract
mkdir -p $D/eval_results
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MIND_SIZE=large DATA_ROOT=$MIND_LARGE USE_ABSTRACT=1 MAX_HISTORY=30 BATCH_SIZE=8 \
  OUTPUT_FILE=$D/eval_results/dev_pointwise_predictions.txt \
  bash scripts/eval_mind_pointwise.sh $D/final_checkpoint dev
```

### Phase 2: RL Commands

```bash
# ============================================
# R2.1-R2.3: RL Fine-tuning
# ============================================
# Note: RL training is not yet integrated into Azure ML pipeline
# Use local scripts for RL experiments:

# R2.1: Asymmetric reward
MODEL_PATH=<best_phase1_checkpoint> \
MIND_SIZE=small \
REWARD_TYPE=pointwise_asymmetric \
bash scripts/rl_mind_pointwise.sh

# R2.2: Lower KL penalty
MODEL_PATH=<best_phase1_checkpoint> \
MIND_SIZE=small \
REWARD_TYPE=pointwise_asymmetric \
KL_COEF=0.05 \
bash scripts/rl_mind_pointwise.sh

# R2.3: More RL epochs
MODEL_PATH=<best_phase1_checkpoint> \
MIND_SIZE=small \
REWARD_TYPE=pointwise_asymmetric \
TOTAL_EPOCHS=2 \
bash scripts/rl_mind_pointwise.sh

# Evaluate RL models
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_eval_r2-x \
  --display-name "R2.x Eval: RL model" \
  --model-path shares/users/wuc/output_dir/<rl_checkpoint>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug
```

### Phase 3: Ensemble & Cascade Commands

```bash
# Per-model flags are AUTO-DETECTED from checkpoint directory name:
#   _chat     → use_chat_template=1
#   _abstract → use_abstract=1
# Override with CHAT_TEMPLATES="1 0 1" and ABSTRACTS="1 0 0" if needed.

SHARES=/scratch/azureml/cr/j/ef9a7f2099e947cab5fb282f38f685e3/cap/data-capability/wd/INPUT_msndni/shares
OUTPUT_DIR=$SHARES/users/wuc/output_dir
MIND_LARGE=$SHARES/users/wuc/data/MIND_large

# ============================================
# E3.A — L1.3 + L1.7 (Abstract+chat vs Base, auto-detect: chat=1,abstract=1 | chat=0,abstract=0)
# ============================================
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIND_SIZE=large MIND_ROOT=$MIND_LARGE MAX_HISTORY=30 BATCH_SIZE=8 \
OUTPUT_FILE=$OUTPUT_DIR/ensemble_results/e3a_l13_l17.txt \
bash scripts/eval_mind_pointwise_ensemble.sh \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30/final_checkpoint

# ============================================
# E3.B — L1.3 + L1.2 (Abstract+chat vs EP7+chat, auto-detect: chat=1,abstract=1 | chat=1,abstract=0)
# ============================================
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIND_SIZE=large MIND_ROOT=$MIND_LARGE MAX_HISTORY=30 BATCH_SIZE=8 \
OUTPUT_FILE=$OUTPUT_DIR/ensemble_results/e3b_l13_l12.txt \
bash scripts/eval_mind_pointwise_ensemble.sh \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat/final_checkpoint

# ============================================
# E3.C — L1.3 + L1.7 + L1.2 (Top-3, equal weights, all auto-detect)
# ============================================
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIND_SIZE=large MIND_ROOT=$MIND_LARGE MAX_HISTORY=30 BATCH_SIZE=8 \
OUTPUT_FILE=$OUTPUT_DIR/ensemble_results/e3c_l13_l17_l12.txt \
bash scripts/eval_mind_pointwise_ensemble.sh \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat/final_checkpoint

# ============================================
# E3.D — L1.3 + L1.7 + L1.2 (L1.3-heavy: weights 1.0 0.8 0.8)
# ============================================
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MIND_SIZE=large MIND_ROOT=$MIND_LARGE MAX_HISTORY=30 BATCH_SIZE=8 \
WEIGHTS="1.0 0.8 0.8" \
OUTPUT_FILE=$OUTPUT_DIR/ensemble_results/e3d_l13_l17_l12_weighted.txt \
bash scripts/eval_mind_pointwise_ensemble.sh \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30/final_checkpoint \
  $OUTPUT_DIR/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat/final_checkpoint
```

```bash
# ============================================
# E3.1-E3.5: Ensemble & Cascade Experiments
# ============================================
# Note: Ensemble and cascade evaluation are not yet integrated into Azure ML pipeline
# Use local scripts for these experiments:

# E3.1: Ensemble with α=0.7 (more point-wise)
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
ALPHA=0.7 \
MIND_SIZE=small \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ensemble.sh dev

# E3.2: Ensemble with α=0.5 (balanced)
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
ALPHA=0.5 \
MIND_SIZE=small \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ensemble.sh dev

# E3.3: Ensemble with α=0.3 (more ranking)
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
ALPHA=0.3 \
MIND_SIZE=small \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ensemble.sh dev

# E3.4: Cascade with TOP_K=15
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
TOP_K=15 \
MIND_SIZE=small \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_cascade.sh dev

# E3.5: Cascade with TOP_K=10
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
TOP_K=10 \
MIND_SIZE=small \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_cascade.sh dev
```

### Phase 4: Multi-task Commands

```bash
# ============================================
# M4.1-M4.3: Multi-task Experiments
# ============================================
# Note: Multi-task training is not yet integrated into Azure ML pipeline
# Use local scripts for these experiments:

# M4.1: Multi-task with 70% point-wise
POINTWISE_RATIO=0.7 \
MIND_SIZE=small \
DATA_ROOT=../data/MIND_small \
USE_CHAT_TEMPLATE=1 \
MAX_HISTORY=30 \
bash scripts/sft_mind_multitask.sh

# M4.2: Multi-task with 80% point-wise
POINTWISE_RATIO=0.8 \
MIND_SIZE=small \
DATA_ROOT=../data/MIND_small \
USE_CHAT_TEMPLATE=1 \
MAX_HISTORY=30 \
bash scripts/sft_mind_multitask.sh

# M4.3: Two-stage training
# Stage 1: Pointwise (2 epochs)
MIND_SIZE=small \
NUM_EPOCHS=2 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# Stage 2: Ranking (3 epochs)
MODEL_PATH=<stage1_checkpoint> \
MIND_SIZE=small \
NUM_EPOCHS=3 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_ranking.sh

# Evaluate multi-task/two-stage models (pointwise)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_eval_m4-x_pointwise \
  --display-name "M4.x Eval: multi-task (pointwise)" \
  --model-path shares/users/wuc/output_dir/<multitask_checkpoint>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug

# Evaluate multi-task/two-stage models (ranking)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_eval_m4-x_ranking \
  --display-name "M4.x Eval: multi-task (ranking)" \
  --model-path shares/users/wuc/output_dir/<multitask_checkpoint>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type ranking \
  --split dev
  # --debug
```

### Phase 5: Advanced Commands (CoT RL)

> **Note**: CoT RL pipeline now available via `pipeline/run_cot_rl_pipeline.py`.
> Training includes **integrated evaluation** by default (`--run-eval 1`).

```bash
# ============================================
# A5.1a: Standard CoT style (Azure ML Pipeline)
# ============================================
python pipeline/run_cot_rl_pipeline.py \
  --experiment-name mind_cot_rl_a5-1a_standard \
  --display-name "A5.1a: CoT RL (standard)" \
  --sft-model shares/users/wuc/output_dir/sft_mind_pointwise_*/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --cot-style standard \
  --reward-type mind_cot_binary \
  --max-response-length 256 \
  --train-batch-size 32 \
  --learning-rate 1e-7 \
  --kl-loss-coef 0.5 \
  --total-epochs 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# ============================================
# A5.1b: Category-based CoT style (Azure ML Pipeline)
# ============================================
python pipeline/run_cot_rl_pipeline.py \
  --experiment-name mind_cot_rl_a5-1b_category \
  --display-name "A5.1b: CoT RL (category)" \
  --sft-model shares/users/wuc/output_dir/sft_mind_pointwise_*/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --cot-style category \
  --reward-type mind_cot_binary \
  --max-response-length 256 \
  --train-batch-size 32 \
  --learning-rate 1e-7 \
  --kl-loss-coef 0.5 \
  --total-epochs 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# ============================================
# A5.1c: Detailed CoT style (Azure ML Pipeline)
# ============================================
python pipeline/run_cot_rl_pipeline.py \
  --experiment-name mind_cot_rl_a5-1c_detailed \
  --display-name "A5.1c: CoT RL (detailed)" \
  --sft-model shares/users/wuc/output_dir/sft_mind_pointwise_*/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --cot-style detailed \
  --reward-type mind_cot_ndcg \
  --max-response-length 256 \
  --train-batch-size 32 \
  --learning-rate 1e-7 \
  --kl-loss-coef 0.5 \
  --total-epochs 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug

# CoT RL Hyperparameters:
# - --cot-style: standard | category | detailed
# - --reward-type: mind_cot_binary | mind_cot_ndcg | mind_cot_auc | mind_cot_margin
# - --max-response-length: 256 (longer for CoT reasoning)
# - --train-batch-size: 32 (reduced for longer sequences)
# - --kl-loss-coef: 0.5

# [Optional] Local scripts for quick iteration:
# SFT_MODEL=<checkpoint> COT_STYLE=category REWARD_TYPE=mind_cot_binary bash scripts/rl_mind_cot.sh
```

---

## Execution Schedule

### Week 1: SFT Optimization

| Experiments | GPU Hours | Notes |
|-------------|-----------|-------|
| P1.1, P1.2 | 16h | Run in parallel |
| P1.3, P1.4 | 16h | Run in parallel |
| Evaluate P1.1-P1.4 | 4h | Identify best |
| P1.5 (combined) | 8h | Best settings |
| P1.6 (4B model) | 12h | If resources allow |

### Week 2: Ensemble & RL

| Experiments | GPU Hours | Notes |
|-------------|-----------|-------|
| E3.1, E3.2, E3.3 | 4h | No training! |
| E3.4, E3.5 | 4h | Cascade variants |
| R2.1 | 8h | RL on best SFT |
| R2.2, R2.3 | 16h | RL tuning |
| Final evaluation | 4h | Compare all |

### Week 3: Advanced (Optional)

| Experiments | GPU Hours | Notes |
|-------------|-----------|-------|
| M4.1, M4.2 | 16h | Multi-task |
| A5.1 (CoT) | 16h | If time permits |
| Final report | - | Document results |

---

## Tracking & Evaluation

### Experiment Log Template

```markdown
## Experiment: [EXP_ID]

**Date**: YYYY-MM-DD
**Status**: [ ] Pending / [ ] Running / [ ] Completed / [ ] Failed

### Configuration
- Model:
- Dataset: MINDlarge / MINDsmall
- Key settings:
  - NEG_RATIO:
  - NUM_EPOCHS:
  - MAX_HISTORY:
  - USE_CHAT_TEMPLATE: 1

### Command
```bash
<command here>
```

### Results
| Metric | Value |
|--------|-------|
| AUC | |
| MRR | |
| nDCG@5 | |
| nDCG@10 | |
| Training time | |

### Notes
-
```

### Finding the Best Checkpoint

Training creates multiple checkpoint folders. Here's how to find and use the best one:

**Checkpoint Types:**
| Folder | Description | When Created |
|--------|-------------|--------------|
| `checkpoint-512`, `checkpoint-1024`, ... | Intermediate checkpoints | Every N steps during training |
| `final_checkpoint` | Best model (if `load_best_model_at_end=True`) | End of training (new scripts only) |

**Option 1: Check `best_model_checkpoint` in trainer_state.json** (Recommended)

```bash
# Find the last checkpoint
LAST_CKPT=$(ls -d output_dir/sft_mind_pointwise_xxx/checkpoint-* | sort -t- -k2 -n | tail -1)

# Get the best checkpoint path
grep "best_model_checkpoint" "$LAST_CKPT/trainer_state.json"
# Output: "best_model_checkpoint": "/path/to/checkpoint-1536"
```

**Option 2: Compare eval_loss across checkpoints**

```bash
for ckpt in output_dir/sft_mind_pointwise_xxx/checkpoint-*/trainer_state.json; do
  loss=$(grep -o '"eval_loss": [0-9.]*' "$ckpt" | tail -1 | grep -o '[0-9.]*')
  echo "$(dirname $ckpt): eval_loss=$loss"
done | sort -t= -k2 -n | head -5
```

**Note:** Both `checkpoint-xxxx` and `final_checkpoint` folders work for evaluation. Just pass the checkpoint path directly:

```bash
# Using intermediate checkpoint (works fine!)
python pipeline/run_eval_pipeline.py \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_xxx/checkpoint-1024 \
  --use-chat-template 1
```

---

### Quick Evaluation Commands

```bash
# ============================================
# Azure ML Pipeline Evaluation (Recommended)
# ============================================

# Point-wise evaluation on dev set
python pipeline/run_eval_pipeline.py \
  --experiment-name quick_eval_pointwise \
  --display-name "Quick Eval: pointwise (dev)" \
  --model-path shares/users/wuc/output_dir/<checkpoint_name>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug

# Ranking evaluation on dev set
python pipeline/run_eval_pipeline.py \
  --experiment-name quick_eval_ranking \
  --display-name "Quick Eval: ranking (dev)" \
  --model-path shares/users/wuc/output_dir/<checkpoint_name>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type ranking \
  --split dev
  # --debug

# Test set evaluation (for submission)
python pipeline/run_eval_pipeline.py \
  --experiment-name test_eval \
  --display-name "Test Eval: submission" \
  --model-path shares/users/wuc/output_dir/<checkpoint_name>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split test
  # --debug

# ============================================
# Local Evaluation (Alternative)
# ============================================

# Point-wise evaluation (multi-GPU)
MIND_SIZE=small USE_CHAT_TEMPLATE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_pointwise.sh <checkpoint> dev

# Ranking evaluation (multi-GPU)
MIND_SIZE=small USE_CHAT_TEMPLATE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ranking.sh <checkpoint> dev

# Quick test (100 impressions)
MIND_SIZE=small USE_CHAT_TEMPLATE=1 \
bash scripts/eval_mind_pointwise.sh <checkpoint> dev 100
```

---

## Key Hyperparameter Reference

### By Model Size

| Model | Learning Rate | Micro Batch | Recommended |
|-------|---------------|-------------|-------------|
| 1.7B | 3e-4 (base) / 2e-5 (instruct) | 4 | Quick iteration |
| 4B | 2e-5 | 2 | Balanced |
| 8B | 1e-5 - 2e-5 | 1-2 | Best quality |

### By Task

| Task | NEG_RATIO | MAX_HISTORY | CUTOFF_LEN |
|------|-----------|-------------|------------|
| Point-wise | 2.0-3.0 | 30-50 | 2048 |
| Ranking | 4.0-8.0 | 30 | 4096 |
| Multi-task | pw:1.0, rk:4.0 | 30 | 4096 |
| RL | 3.0 | 30-50 | 2048 |

### Critical Settings

- **USE_CHAT_TEMPLATE=1**: Essential for instruct models (+5% AUC!)
- **MIND_SIZE=small**: Use large dataset for SOTA attempts
- **Flash Attention**: Enable for faster training/eval

---

## Success Criteria

| Milestone | AUC Target | Status |
|-----------|------------|--------|
| Baseline | 69.69% | ✅ Achieved |
| Submitted best (E3.A) | 71.44% | ✅ On leaderboard (#14) |
| +0.8% → top-5 | 72.26% | ⬜ Pending |
| +1.1% → top-3 | 72.58% | ⬜ Pending |
| +1.8% → SOTA | 73.26% | ⬜ Target (beat zetik) |

---

### Phase 7: Architecture Comparisons (Literature Baselines)

> **Survey / lit review:** see `docs/MIND_SURVEY.md` for the full deep-dive grounding each baseline
> (PBNR, ONCE, RecRanker/GLIMPSE pairwise, frozen-LLM+MLP) and the broader MIND landscape (NRMS/NAML/DKN
> dual-encoders, LLaRA/GLIMPSE LLM rankers, SFT/RLCD/RL/Teacher-OPD paradigms, and the "distillation needs a
> real teacher" lesson from the JEV R-Drop collapse).

> **Motivation**: The deep research survey identified three peer-reviewed paradigms that have been benchmarked on MIND. Implementing each one cleanly against our L1.3 baseline (0.7049 AUC) will show whether alternative architectures can close the gap to SOTA — and which combination is worth ensembling.

---

#### 7A: PBNR-style — Constrained Yes/No Decoding (arXiv:2304.07862)

**What changes vs. current approach**: Current scoring uses raw token logits `P(Yes)/(P(Yes)+P(No))`. PBNR enforces `P(yes)+P(no)=1` via constrained decoding at inference, yielding a properly calibrated binary probability. Training is identical; only inference scoring changes.

**Expected gain**: Small but clean. Better calibration → better ranking especially at the tails of the score distribution.

| Exp ID | Status | Model | Dataset | Config | AUC | Notes |
|--------|--------|-------|---------|--------|-----|-------|
| **A7.1** | ⬜ Pending | Qwen3-1.7B | MINDlarge | L1.3 + constrained decoding at eval | - | No retraining needed — eval-only change on L1.3 checkpoint |

**Implementation**: Modify `evaluate_mind_pointwise.py` to restrict vocabulary to `["Yes", "No"]` tokens during generation (set all other logits to `-inf` before softmax). Then score = `P(Yes)`.

---

#### 7B: ONCE-style — LLM as Content Encoder (arXiv:2305.06566)

**What changes vs. current approach**: Instead of generative Yes/No scoring, use Qwen3-1.7B as a **semantic encoder** — extract the last-layer hidden state of each news article, then score as `dot_product(user_embedding, candidate_embedding)` where user embedding = mean pool of clicked article embeddings.

**Expected gain**: Faster inference (precompute news embeddings offline), potentially different error profile → good ensemble partner.

| Exp ID | Status | Model | Dataset | Config | AUC | Notes |
|--------|--------|-------|---------|--------|-----|-------|
| **A7.2a** | ⬜ Pending | Qwen3-1.7B | MINDlarge | Frozen encoder + mean-pool user rep | - | Frozen baseline — no training |
| **A7.2b** | ⬜ Pending | Qwen3-1.7B | MINDlarge | Fine-tuned encoder + contrastive loss (InfoNCE) | - | Train with in-batch negatives |

**Implementation notes**:
- A7.2a: Zero-shot encoder — just extract CLS/mean-pool hidden states, no training
- A7.2b: Fine-tune with contrastive loss using same train split as L1.3; batch size = 256, same negatives

---

#### 7C: RecRanker-style — Hybrid Pointwise + Pairwise Training (arXiv:2312.16018)

**What changes vs. current approach**: Add a **pairwise Bradley-Terry ranking loss** alongside the existing pointwise cross-entropy. For each impression, construct positive-negative pairs and train the model so `P(Yes|positive) > P(Yes|negative)` using the BT objective. This is also the GLIMPSE approach (arXiv:2409.17711).

**Expected gain**: +0.2–0.5% AUC based on GLIMPSE results. Directly compatible with current architecture — just an additional loss term.

| Exp ID | Status | Model | Dataset | Config | AUC | Notes |
|--------|--------|-------|---------|--------|-----|-------|
| **A7.3** | ⬜ Pending | Qwen3-1.7B | MINDlarge | L1.3 config + Bradley-Terry pairwise loss (λ=0.5) | - | Joint loss = CE + λ·BT; same epochs/lr as L1.3 |
| **A7.4** | ⬜ Pending | Qwen3-1.7B | MINDlarge | A7.3 + λ sweep (0.1, 0.3, 0.5, 0.7) | - | Find optimal pairwise loss weight |

> **MIND_small pilot (2026-10-13, dev evals `auto_eval_a73.py` + `collect_eval_aucs.py`)**:
> pairwise BT on the small dataset gives a clear but content-sensitive signal.
>
> | Variant | train | dev AUC |
> |---------|-------|---------|
> | pw0.0 abs0 (control = pointwise CE) | `silver_ring_l47fyrmkx0` | 0.6562 |
> | **pw0.5 abs0 (title-only + BT)** | `serene_bell_nmnm20cn4v` | **0.6669** (+0.0107) |
> | pw0.5 abs1 (abstract + BT) | `bubbly_screw_j68j6vw7rs` | 0.6563 (≈control) |
>
> **Read**: adding BT pairwise on **title-only** lifts dev AUC +0.0107 over pure pointwise CE;
> adding **abstract cancels the pairwise gain** (0.6563 ≈ control). Suggests the pairwise term
> needs short, discriminative candidates (title), and the longer abstract prompt dilutes the
> relative ordering signal. Promotable config for MINDlarge: **pw0.5, title-only, no abstract**
> (λ=0.5). A7.4 λ-sweep should focus on the title-only setting.
>
> **MINDlarge submission (2026-10-13)**: `patient_brick_b829btzn4v` — A7.3L pw0.5 abs0
> title-only, Qwen3-1.7B, MIND_large (train_sample=0), matching L1.3 otherwise. COMPLETED.
>
> **MINDlarge eval (2026-10-13, `purple_malanga_09mg6l4ncq`)**: **dev AUC 0.6422**
> (MRR 0.2998, nDCG@5 0.3303, nDCG@10 0.3935), verified on **MIND_large dev
> (376,471 impressions)** with checkpoint `..._pw0.5_chat/final_checkpoint`.
>
> **Read**: the pairwise title-only gain **does NOT transfer to MIND_large**. 0.6422 is
> well below the L1.3 pointwise reference (0.7049) and even the L1.3 AML-pipeline repro
> (0.6622). The +0.0107 from small was noise/small-scale artifact. **Deprioritize A7.3/A7.4
> pairwise** — no clear signal at production scale; λ-sweep (A7.4) not justified.

**Implementation**: Add `pairwise_loss_weight` parameter to `sft_mind_pointwise_ds.py`. For each batch, sample random pos-neg pairs and add BT loss: `L_BT = -log(sigmoid(score_pos - score_neg))`.

---

#### Summary: Phase 7 Expected Complexity

| Exp | Training Required | Estimated GPU Hours | Risk |
|-----|-------------------|---------------------|------|
| A7.1 | ❌ Eval only | ~2h | Low — eval change only |
| A7.2a | ❌ Eval only | ~2h | Low — frozen encoder |
| A7.2b | ✅ Full train | ~24h | Medium — new architecture |
| A7.3 | ✅ Full train | ~24h | Low — additive to L1.3 |
| A7.4 | ✅ Full train ×4 | ~96h | Medium — sweep |

**Recommended order**: A7.1 → A7.3 → A7.2a → A7.2b (cheapest first, most likely gains first)

---

### A7.5: Frozen LLM + MLP Classifier (Ablation Baseline)

**Paradigm**: Extract the last-token hidden state from a frozen Qwen3-1.7B backbone, then train a 2-layer MLP binary classifier head (`hidden_size → 256 → 1`) on top. BCE loss. No gradient flows through the backbone — only the MLP (~660K params) trains.

**Motivation**: BERT-era methods (NRMS, NAML, PLM4NewsRec) used frozen/fine-tuned PLM encoders + MLP heads and achieved ~67-68% AUC. This experiment asks: does a much stronger frozen LLM backbone close the gap to fine-tuned Qwen3-1.7B, or does task-specific fine-tuning remain essential? Serves as an ablation baseline — if frozen LLM + MLP ≈ fine-tuned LLM, it suggests the LLM's prior is doing most of the work.

**Architecture**:
- Backbone: Qwen3-1.7B, fully frozen (`requires_grad=False`)
- Feature: `outputs.hidden_states[-1][:, -1, :]` (last transformer layer, last token)
- Head: `Linear(2048→256) + GELU + Dropout(0.1) + Linear(256→1)`
- Loss: `binary_cross_entropy_with_logits`
- Optimizer: AdamW on MLP only, lr=1e-3, CosineAnnealingLR

**Expected result**: 67–70% AUC — likely below L1.3 (70.49%) because no task-specific fine-tuning. The frozen representation may not optimally encode the "would a user click this?" signal. Main value is as a fast, cheap ablation (single GPU, ~2-4h).

**Novelty assessment**: The pattern itself (frozen encoder + MLP) is known from BERT era. In the decoder-LLM era on MIND specifically, no peer-reviewed paper found doing this exact setup — but PRECTR-V2 (arXiv:2602.20676) explicitly calls out frozen Emb+MLP as suboptimal due to "misalignment between representation learning and CTR fine-tuning." Run as reference baseline, not primary contribution.

| Exp ID | Status | Model | Dataset | Config | AUC | Notes |
|--------|--------|-------|---------|--------|-----|-------|
| **A7.5** | ⬜ Pending | Qwen3-1.7B (frozen) | MINDlarge | Frozen backbone + MLP(256), lr=1e-3, BS=64, 10 epochs | - | Reference ablation baseline; single GPU, no DeepSpeed needed |

**Implementation**: `src/train_mind_mlp_classifier.py` + `src/evaluate_mind_mlp_classifier.py` + launch scripts already created.

```bash
# Train (single GPU, ~2-4h)
MODEL_PATH=/path/to/Qwen3-1.7B \
DATA_ROOT=data/MIND \
bash scripts/train_mind_mlp_classifier.sh

# Eval
bash scripts/eval_mind_mlp_classifier.sh /path/to/Qwen3-1.7B output_dir/mlp_classifier/best_mlp.pt dev
```

| Exp | Training Required | Estimated GPU Hours | Risk |
|-----|-------------------|---------------------|------|
| A7.5 | ✅ Full train (MLP only) | ~3h | Low — single GPU, only MLP trains |

---

## 💡 Future Work / Ideas

### F1: LLM-Generated Narrative User Profiles

**Idea**: Instead of feeding raw reading history (a numbered list of article titles), use an LLM to generate a concise narrative profile summarizing the user's interests, then use that summary as the user representation.

**Example**:

Current input:
```
1. [Sports] LeBron James scores 40 points in Lakers win
2. [Sports] NBA playoffs preview
3. [Technology] Apple iPhone review
...
```

Proposed input:
```
"This user is primarily interested in NBA basketball and follows player
performances closely. They also have moderate interest in consumer technology."
```

**Why it may help**:
- Compresses very long histories into compact, generalized representations
- Captures higher-level interests that generalize across similar articles
- Related work: PALR (arXiv:2305.07622), UP5 (arXiv:2304.14399) show gains

**Why it may not help (for current setup)**:
- History is already capped at 30 short items — token cost is manageable
- The fine-tuned model already learns to implicitly summarize user interests
- Adds offline preprocessing step + error propagation from profile generation
- More useful when histories are very long (500+) or very sparse (1–2 clicks)

**When to revisit**: After Phase 1–2 ablations are complete and diminishing returns are observed. Worth trying as a separate ablation (e.g., A5.4) if the gap to SOTA remains >1.5%.

---

## Appendix: File Locations

```
MiniOneRec/
├── pipeline/                       # 🆕 Azure ML Pipelines
│   ├── run_pipeline.py            # Training pipeline submission
│   ├── run_eval_pipeline.py       # Evaluation pipeline submission
│   ├── README.md                  # Pipeline documentation
│   └── components/
│       ├── mind_train/            # Training component
│       │   ├── component.yaml
│       │   ├── config_mind_train.json
│       │   └── pipeline_executor.py
│       └── mind_eval/             # Evaluation component
│           ├── component.yaml
│           ├── config_mind_eval.json
│           └── pipeline_executor.py
├── scripts/                        # Local execution scripts
│   ├── sft_mind_pointwise.sh      # Point-wise SFT
│   ├── sft_mind_pointwise_ds.sh   # Point-wise + DeepSpeed
│   ├── sft_mind_ranking.sh        # Ranking SFT
│   ├── sft_mind_multitask.sh      # Multi-task SFT
│   ├── sft_mind_twostage.sh       # Two-stage SFT
│   ├── rl_mind_pointwise.sh       # Point-wise RL
│   ├── rl_mind.sh                 # Ranking RL
│   ├── rl_mind_cot.sh             # CoT RL
│   ├── eval_mind_pointwise.sh     # Point-wise eval
│   ├── eval_mind_ranking.sh       # Ranking eval
│   ├── eval_mind_ensemble.sh      # Ensemble eval
│   └── eval_mind_cascade.sh       # Cascade eval
├── src/
│   ├── mind_utils.py              # Shared utilities
│   ├── sft_mind_pointwise.py      # Point-wise training
│   ├── sft_mind_multitask.py      # Multi-task training
│   └── evaluate_mind_*.py         # Evaluation scripts
├── output_dir/                    # Training outputs
├── results_mind/                  # Evaluation results
└── EXPERIMENT_PLAN.md             # This file
```

## Experiment Execution Modes

### Azure ML Pipeline (Recommended for Production)
- **Pros**: Scalable, tracked, reproducible, 8x A100 GPUs
- **Cons**: Requires Azure setup, slightly slower iteration
- **Use for**: Main experiments, SOTA attempts, final evaluations

### Local Scripts (For Quick Iteration)
- **Pros**: Fast iteration, full control, easy debugging
- **Cons**: Limited to local GPUs, manual tracking
- **Use for**: Development, debugging, quick tests

---

---

## 🚀 Quick Reference: Experiment Commands

### Update Experiment Status

```bash
# Mark experiment as running
# In EXPERIMENT_PLAN.md, change: ⬜ Pending → 🔄 Running

# Mark experiment as completed with results
# In EXPERIMENT_PLAN.md, change: 🔄 Running → ✅ Completed
# Fill in: AUC | MRR | nDCG@5 | nDCG@10 | Notes
```

### Submit Experiment (Phase 1)

```bash
# Example: P1.1 (More negatives) - with integrated evaluation
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-1_neg3-0 \
  --display-name "P1.1: More negatives (neg=3.0)" \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 5 \
  --neg-ratio 3.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug
```

### Re-evaluate Experiment (Optional)

```bash
# Use standalone eval pipeline only if re-evaluation is needed
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_eval_p1-1 \
  --display-name "P1.1 Eval: neg=3.0" \
  --model-path shares/users/wuc/output_dir/sft_mind_pointwise_*/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug
```

### Update Progress Summary

After each experiment, update the Progress Summary table:
1. Increment "Completed" or "Failed" count
2. Decrement "Pending" count
3. Update "Best AUC" if improved
4. Update overall progress percentage

### Copy Results to Archive

When an experiment is completed, copy the row from active table to "Completed Experiments Archive" section.

---

## 🏆 SOTA Pursuit Roadmap (2026-09-27)

**Goal**: Beat **zetik @ 0.7326 AUC** on Codabench MIND (comp 13967). Currently **#14 @ 0.7144** → need **+0.0182 AUC**.

**Principle**: SOTA gap is too large for any single +1% experiment. Stack **independent, additive** signals:
ready-to-use ensembles (no training) → CF signal (inference-time blend) → architecture ablations (PBNR constrained decode = eval-only) → scaling (4B/8B). Each contributes a small independent gain; combined they close the gap.

### Tier 0 — Already have (no new training)
| Lever | ETA | Est. gain | Status |
|-------|-----|-----------|--------|
| E3.A ensemble (0.7144) | submitted | baseline | ✅ on board |
| **E3.C** L1.3+L1.7+L1.2 | ready | +? (was 0.7105 dev) | try `--weights 1.0 0.8 0.8` (E3.D) |
| **Snapshot ensemble** L1.3 ep3+ep4+final | free | +0.2-0.5% | run E3.S1 |

### Tier 1 — Inference-time (fast, cheap, high ROI — do FIRST)
| Lever | Where | Est. gain | Effort |
|-------|-------|-----------|--------|
| **PBNR constrained decoding** (P(Yes)+P(No)=1) | `evaluate_mind_pointwise.py` (eval-only) | +0.1-0.3% | ~2h |
| **CF score ensemble** (`--cf_alpha` sweep 0.1-0.5) | `evaluate_mind_pointwise.py` + `compute_cf_scores.py` | +0.1-0.4% | ~2h |
| **Weighted multi-ensemble** E3.D/E3.F | `eval_mind_pointwise_ensemble.sh` | +0.1-0.3% | ~1h |

### Tier 2 — Training ablations (additive, ~24h GPU each)
| Lever | Where | Est. gain |
|-------|-------|-----------|
| **A7.3 / A7.4** RecRanker pairwise (GLIMPSE) Bradley-Terry loss | `sft_mind_pointwise_ds.py` + λ sweep | +0.2-0.5% |
| **L1.6** 4B abstract (checkpoint-37376=0.6819 mid; finish + vs train) | `pipeline/run_pipeline.py` | +0.5-1.0% |
| **6D CF-augmented prompt** "similar users also clicked X,Y,Z" | `sft_mind_pointwise_ds.py` | +0.1-0.3% |

### Tier 3 — Ensemble the new models
After Tier 2 adds divergent models (4B, pairwise, CF-prompt), fold them into **majority-vote ensembles** (3/5/7 models):
E3.F → E3.G (5-model). This is where the compounding gain lands.

### Recommended order (cheapest → compound)
1. **Now**: E3.D weighted + E3.S1 snapshot + PBNR constrained eval + CF α sweep — all inference-time, no GPU training.
2. If inference gains stall <~0.5-0.8%, launch **A7.3 pairwise training** (additive to L1.3) in parallel with **finishing L1.6 4B**.
3. Fold winners into **E3.F (3-model) / E3.G (5-model)** majority vote → submit.
4. Validate on **dev** first; only submit to Codabench when dev shows a clear edge over 0.7152.

---

*Last updated: 2026-09-27*
