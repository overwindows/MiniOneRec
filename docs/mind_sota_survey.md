# MIND SOTA Survey: Literature + Code Scan for an LLM Pointwise-Scoring Stack

*Compiled 2026-10-30. Context: our stack = Qwen3-1.7B/8B pointwise SFT ("Will this user read this article? Yes/No", score = logit(Yes) - logit(No)), prompt-feature ablations, DPO/RLCD contrastive stage on same-impression pairs, CF-score blending, GBDT re-ranker. Current MIND-small dev AUC ~0.695-0.696.*

**Split-labeling convention used throughout:** every number is tagged with the split it was reported on. The three common protocols are **not comparable**:
- `small-dev` = MINDsmall_dev (73,152 impressions; the de-facto "test" of MIND-small since small has no test set). This is our protocol.
- `large-dev` = MINDlarge validation (often used as test by papers, sometimes with a re-split of train).
- `large-test` = official hidden-label test set, scored via leaderboard submission (2.37M impressions).
Typical offset: models score roughly +1.5 to +2.5 AUC points higher on large-test than on small-dev (e.g., UNBERT 67.62 small-dev vs 70.68 large-test; DIGAT ~68.8 small vs ~70.1 large).

---

## 1. The MIND leaderboard and the 2020 competition

### 1.1 Leaderboard snapshot (msnews.github.io, fetched 2026-10-30; all numbers large-test)

| Rank | Entry | AUC | MRR | nDCG@5 | nDCG@10 | Date |
|---|---|---|---|---|---|---|
| 1 | zetik | 0.7326 | 0.3780 | 0.4175 | 0.4735 | Jun 2026 |
| 2 | newsbang | 0.7316 | 0.3768 | 0.4162 | 0.4722 | May 2026 |
| 3 | UniUM-Fastformer-Pretrain | 0.7304 | 0.3770 | 0.4180 | 0.4718 | Oct 2021 |
| 4 | MINER | 0.7275 | 0.3724 | 0.4102 | 0.4661 | Sep 2021 |
| 5 | _dctr | 0.7272 | 0.3727 | 0.4112 | 0.4673 | - |
| 6 | UniUM-Fastformer | 0.7268 | 0.3745 | 0.4151 | 0.4684 | Aug 2021 |
| 9-10 | tuyendv / quynhnl | 0.7256 | 0.3720 | 0.4101 | 0.4660 | 2022 |

Source: https://msnews.github.io/ . Notes:
- The two **2026 entries (zetik 0.7326, newsbang 0.7316) have no public method descriptions** — the 2021-era UniUM-Fastformer-Pretrain (PLM-NR line from MSRA: Fastformer user encoder + pretrained LM news encoder, unified user-modeling pretraining) stood at #1 for ~4.5 years until then.
- UNBERT ensemble ("UNBERT-en") reported 0.7183 large-test and claimed leaderboard #1 in Sept 2021 (IJCAI'21 paper, https://www.ijcai.org/proceedings/2021/0462.pdf); single UNBERT: **0.7068 large-test / 0.6762 small-dev-as-test**. UNBERT = single BERT cross-encoder over concatenated [user history ; candidate] with word-level + news-level matching signals — i.e., architecturally the closest published ancestor of our LLM pointwise cross-encoder.
- MINER (ACL Findings 2022): multi-interest user modeling with poly attention, BERT news encoder — 0.7275 large-test.

### 1.2 MIND News Recommendation Competition (Jul-Sep 2020)

Winners listed at https://msnews.github.io/competition.html . **Grand prize: Huige Cheng (Sogou), handle `chenghuige`** — report: https://msnews.github.io/assets/doc/1.pdf , code: https://github.com/chenghuige/mind . Read the full report; key facts (all verified from the PDF):

- **Final: 0.7133 large-test (ensemble), 0.707 best single model.** Milestones table: ID-only (uid+docid+history docids) 0.5272 test → +cat/entity 0.6763 → +title/abstract 0.6979 → tuned base 0.7036 → **+body 0.707** → +dev data for training 0.7104 → 5-negative-subsample-ensemble 0.7133.
- **Pointwise training** (binary click), **shuffle between impressions** ("shuffle between impressions gets better per-impression AUC; totally random shuffle gets better global AUC").
- **Negative downsampling to 1/8** of negatives per impression → ~16.5% positive rate; generates 5 datasets with disjoint negative samples; **mean-ensemble of the 5 models** = +0.3 pts.
- **Model:** DLRM-style field-wise concat + pairwise dot interactions over uid, docid, cat, subcat, entities, title, abstract, body; attention pooling for candidate text; **DIN attention pooling** (candidate-conditioned) over history titles/abstracts/bodies; MLP (1024,512,256). GloVe init, tiny-BERT (L2-H128) title encoder later.
- **Body text matters:** latest 10 history bodies (max 100 words each); post-competition: raising history abstracts/bodies from 10 → 30-50 gave +0.3-0.4 dev AUC.
- **Continue-pretraining tiny-BERT on clicked-title sequences** (train+dev+test corpus) beat stock BERT: title-only dev AUC 0.6916 (GloVe attn) → 0.6987 (bert) → 0.7034 (bert continue-pretrained); new-doc AUC 0.6996 → 0.7273.
- **What did NOT work on test:** direct AUC/pairwise losses, dense features (history length, doc freshness), impression-context features (+0.4 dev AUC but hurt test — test impressions are shuffled), day-by-day training, >1 epoch.
- **Dev/test shift warning:** 87.5% of test docs are new vs 32.3% in dev; he masked 92% of dev docids to make dev predictive of test; tracked **"global AUC of new docs"** as a generalization metric.

Second-place reports: U Kang (SNU) https://msnews.github.io/assets/doc/2.pdf , Yichao Lu (Layer 6) doc/3.pdf — both heavy ensembles of neural rankers (not fetched in detail).

**Orthogonal to our LLM stack:** negative-subsample bagging (5x), ID/CTR memorization features (dev-only caution), body text, impression-shuffle vs random-shuffle choice, candidate-conditioned (DIN-style) history selection, new-doc AUC as a diagnostic.

---

## 2. Classic neural + PLM baselines: what AUC, which split

| Model | small-dev | large-dev | large-test | Source |
|---|---|---|---|---|
| NRMS (as rerun by Prompt4NR, own small re-split) | 65.63 | - | - | arXiv:2304.05263 Table 3 |
| UNBERT | 67.62 | - | 70.68 (70.68=0.7068) | IJCAI'21, ijcai.org/proceedings/2021/0462.pdf |
| HieRec | ~67.8-69.0 (split labels unclear in snippet; HieRec paper: 67.95 small / 69.03 large per common citation) | - | - | arXiv:2106.04408; DIGAT table arXiv:2210.05196 |
| DIGAT | ~68.8 | - | ~70.1 (large) | arXiv:2210.05196 (per search snippets; verify table before quoting) |
| GLORY (title only, GloVe) | 68.16 | 69.54 | - | Transformers4NewsRec arXiv:2410.13125 |
| GLORY (title+body) | 69.07 | 70.65 | - | arXiv:2410.13125 |
| GLORY (NewsReX rerun, 3 seeds) | 66.24 | - | - | newsrex-readme.static.hf.space — **conflicts with above; reruns are consistently lower than original papers** |
| MINER (original) | - | - | 72.75 | leaderboard; (71.51 figure circulating is secondhand) |
| MINER (NewsReX / MANNeR reruns) | 65.79 / 63.1±1.2 (large re-split) | - | - | NewsReX; arXiv:2307.16089 |
| Fastformer+PLM-NR (UniUM) | - | - | 73.04 | leaderboard; Fastformer arXiv:2108.09084, PLM-NR arXiv:2104.07413, SpeedyFeed arXiv:2102.09268 |
| MANNeR (CR-module, RoBERTa-base) | - | **69.7±0.9** (large, dev-as-test, temporally re-split train) | - | arXiv:2307.16089 v3 Table 1 (EMNLP-Findings 2024) |
| CAUM-PLM (MANNeR's best baseline) | - | 66.4±3.1 (same protocol) | - | arXiv:2307.16089 |

Takeaways:
- **Reproduction gap is pervasive**: independent reruns (NewsReX, MANNeR, the 2025 survey arXiv:2502.09797) place NRMS/NAML/LSTUR/MINER at 50-66 AUC small-dev, far below original claims. Original-paper numbers on small-dev cluster at 66-69; nothing published clearly exceeds **~69.7 on a dev-as-test protocol**, and that one (MANNeR) is on large-dev.
- **Our 69.5-69.6 small-dev is therefore at or above every published small-dev number we found** (Prompt4NR ensemble 69.64 being the nearest, on its own re-split). Published headroom evidence: large-dev/test numbers suggest 1.5-2.5 pts of split offset, and the leaderboard top is 73.3 large-test — consistent with "low 70s on small-dev" being the practical ceiling area.
- **MANNeR** (github.com/andreeaiana/manner): supervised-contrastive news encoder (SCL, temp 0.36, 4 negatives), **no learned user encoder — late fusion = mean of candidate-vs-history dot products**, last-4 RoBERTa layers tuned. Plus plug-in category/sentiment modules linearly blended at score level: `score = sim_content + λ·sim_category + ...`.

---

## 3. LLM-based news / sequential recommendation (2023-2026)

### 3.1 Scoring/ranking with (P)LLMs on MIND

- **Prompt4NR** (SIGIR'23, arXiv:2304.05263, github.com/resistzzz/Prompt4NR). BERT-base cloze-prompting ("[MASK]: related/unrelated") on **MIND-small** (own temporal re-split: penultimate week train, last week test). 12 templates (discrete/continuous/hybrid × relevance/emotion/action/utility). Best single template 69.37 AUC; **cross-type ensemble 69.64 / MRR 34.26 / nDCG@5 38.30 / nDCG@10 44.33**; BERT-NRMS baseline 68.60; NRMS 65.63. History = 50 recent clicks, titles capped at 10 words, candidate 20 words. **Template-ensembling gave +0.3-0.4 over best single template.**
- **GLIMPSE** (arXiv:2409.17711, Amazon). Flan-T5-base (~200M) **multi-task SFT: pointwise Yes/No relevance + pairwise "Candidate A/B" preference** on MIND-small, evaluated on the 73,152-impression set (= small-dev protocol, same as ours). Inference: rank by pointwise score, then **m right-to-left bubble passes over top-k using the pairwise head** (swap adjacent if Pref says so). Results (their Table 1): pointwise 68.18 AUC / 33.56 MRR → +1 RTL pass top-5: 68.97 / 33.73 → **+2 RTL passes top-5: 69.14 / 33.88 / 37.66 / 43.88**. Cost: ~8 extra pairwise calls per impression. Bradley-Terry on target-token probabilities. Theory section gives conditions for expected-MRR improvement.
- **Can LLMs Outshine Conventional Recommenders?** (arXiv:2503.05493). LoRA-tuned Llama-3/Mistral 7-8B, click prob = softmax over YES/NO logits at final token (same trick as ours). On a **MIND 20k-sample / 1.5k-user subset (GAUC)**: Llama-3 0.7345, Mistral 0.7353, BERT-base 0.7175, DCNv2 0.7167, zero-shot Llama-3 0.4904. **Not comparable to full small-dev AUC** (subset + GAUC), but confirms: (a) fine-tuned 7B ≫ zero-shot, (b) the margin over a tuned BERT cross-encoder is ~1.7 pts, (c) conventional models retain ~95% of quality at 10^4x speed.
- **RecPrompt** (arXiv:2312.10463): GPT-3.5/4 prompt-optimization loop, evaluated on only 400 MIND test users — AUC 69.39 (GPT-4, optimized) vs 65.06 (unoptimized). Tiny sample; directional evidence for automated prompt search only.
- **PNR-LLM** (arXiv:2504.20452): hierarchical LLM prompting (Gemini 1.0 Pro) to enrich news + user representations feeding a CNN recommender: 68.88 small / 69.32 large (dev). LLM-as-enricher, not scorer.
- **ONCE/GENRE+DIRE** (WSDM'24, arXiv:2305.06566, github.com/Jyonn/ONCE): closed-source LLM (GPT-3.5) for **data augmentation** (content summarizer, user profiler — regenerating titles, generating category/interest profiles) + open-source LLM (LLaMA) as trainable content encoder (DIRE). Up to +19.3% relative nDCG@5 over non-LLM baselines on MIND-small-scale experiments; the 2025 survey's rerun lists ONCE at 65.06 AUC small-dev (survey protocol), i.e., below MANNeR. Augmentation ideas transfer (LLM-written user-profile line in prompt); encoder part doesn't (we already are the LLM).
- **LKPNR** (arXiv:2308.12028): LLM + knowledge-graph personalized news rec; survey rerun: 67.32 small-dev AUC, best Recall@5/10 in that table.
- **UniTRec** (ACL'23 short, arXiv:2305.15756, github.com/Veason-silverbullet/UniTRec): BART text-to-text; **ranks by averaging a discriminative matching score with a target-text perplexity score**. MIND-small; reports MRR/nDCG/HR only (no AUC). The score-fusion idea (discriminative + generative perplexity of the candidate title given user context) is cheap to try on our model.
- **LLM-TRSR** (WWW'24, arXiv:2403.13325, github.com/zhengzhi-1997/LLM-TRSR): segment long history into blocks → LLM **hierarchical summarization** (summarize blocks, then summarize summaries) → preference summary + recent items + candidate into LoRA-SFT'd LLM recommender. On MIND their hierarchical variant beat recurrent; beats all baselines on their MIND subsets (paper tables use HitRate/nDCG on sampled candidates — not AUC-comparable).
- **Survey of LLM-based news recsys** (arXiv:2502.09797): own controlled rerun on small-dev puts **MANNeR top (68.44 AUC)**, LKPNR 67.32, ONCE 65.06, and finds naive "DLLM-empowered" variants often *underperform* their originals. Confirms: encoder-style contrastive fine-tuning + modular score fusion is the strongest published recipe class on small-dev.

### 3.2 Preference / contrastive fine-tuning (DPO family)

- **S-DPO** (NeurIPS'24, arXiv:2406.09215, "On Softmax DPO for Recommendation"): extends DPO's Bradley-Terry to a **softmax/Plackett-Luce ranking over one positive vs multiple negatives**: loss ≈ -log σ(... ) generalized to -log [exp(βΔ_pos) / Σ_negs exp(βΔ_neg)] on policy/reference log-ratio deltas. Shown to connect to softmax-sampled contrastive loss with inherent hard-negative mining; beats pairwise DPO and SFT on three sequential-rec datasets (LastFM, Goodreads, MovieLens — not MIND). Directly applicable: our RLCD/DPO stage currently uses pairs from the same impression; S-DPO consumes the *whole impression slate* (1 pos, N displayed negatives) per step.
- **SPRec** (WWW'25, arXiv:2412.09243, github.com/RegionCh/SPRec): alternating SFT/DPO where **the model's own previous top predictions serve as DPO negatives** (self-play), aimed at popularity-bias/filter-bubble debiasing; maintains or improves accuracy while improving fairness (MovieLens etc.). Relevant if our DPO stage over-concentrates on popular items.
- **CoLLM** (arXiv:2310.19488, github.com/zyang1580/CoLLM): **inject CF embeddings from a conventional model (e.g., MF/LightGCN/SASRec) into the LLM's input as mapped soft tokens**; two-stage tuning (LoRA on text first, then the mapping MLP). Beats text-only LLM rec (TALLRec-style) in warm-start and the CF teacher in cold-start. This is the principled upgrade of our *post-hoc* CF-score blending: give the LLM the CF signal *before* the Yes/No decision.

### 3.3 Calibration of Yes/No scores

- No news-rec-specific paper found. General findings: raw yes/no token probabilities are overconfident and cluster near 1.0; **Platt scaling is the most robust low-data default; beta calibration with elastic-net when ≥~30% calibration data** (judge-calibration study, arXiv:2605.09702); isotonic overfits when score support is spiky.
- Key structural fact for us: **per-impression AUC/MRR/nDCG are invariant to any monotone transform of the score** — calibration cannot move dev AUC by itself. It matters only where scores are *combined*: CF blending, GBDT feature construction, and cross-model ensembling. Calibrate (Platt/isotonic on a dev fold) each score source before blending; or blend on *per-impression ranks/z-scores* instead of raw logits.

### 3.4 Efficiency for scoring many candidates

- **Shared-prefix KV caching**: put the (long) user-history + instructions prefix first and the (short) candidate last; exact-prefix caching (vLLM automatic prefix caching, SGLang RadixAttention) then amortizes prefill across the ~37 candidates per MIND impression. Verified speedups with no quality change when prefixes match exactly (arXiv:2512.11851). Caveat from ACL'26 judge work (arXiv:2601.08343): *cross-candidate* (position-stitched) KV reuse is unsafe without correction — but that doesn't apply to independent pointwise calls sharing a true common prefix.
- GLIMPSE shows a 200M encoder-decoder gets 68.2 pointwise small-dev — model scale is not the bottleneck below ~69; data/objective/ensembling are.
- SpeedyFeed (arXiv:2102.09268, KDD'21): cache news encodings, bus-LM batching, >100x PLM-recommender training speedup — relevant only if we ever move to bi-encoder distillation.

---

## 4. Ensembling practice: competition evidence

- **MIND 2020 winner**: 5x negative-subsample bagging (+0.3 pts), dev-data-in-train (+0.34 pts), mean score ensemble. UNBERT-en: +1.15 pts over single (0.7068 → 0.7183 large-test).
- **RecSys Challenge 2024 (EB-NeRD, Danish news, same impression/AUC framing)** — overview arXiv:2409.20483:
  | Rank | Team | AUC | Method |
  |---|---|---|---|
  | 1 | :D | 89.24 | 3-stage: transformers + **GBDT + ensembling**, time-aware feature engineering |
  | 2 | BlackPearl | 88.15 | **GBDT ensemble** + hierarchical long/short-term user interest model |
  | 3 | Tom3TK | 87.07 | **GBDT ensemble** w/ article-timeliness features + content-based CF |
  | top academic | FeatureSalad | 85.13 | stacking GBDT + multiple NNs (code: github.com/recsyspolimi/recsys-challenge-2024-ekstrabladet — CatBoost/LightGBM classifiers *and rankers*) |
  Ablation: team :D fell 88.64 → 76.99 AUC without "future-information" features — **timeliness/recency features carry enormous weight in news AUC**. Every top team = GBDT-centric stack over neural scores + handcrafted time features. Our GBDT re-ranker is the right architecture; the leverage is in its feature set (time-to-impression, article age at impression, CTR-so-far, position/popularity dynamics) and in using a *ranking* objective (LambdaRank/YetiRank per-impression groups), not plain logloss.

---

## 5. GitHub repos worth mining

| Repo | What it offers |
|---|---|
| github.com/chenghuige/mind | MIND-2020 winner's code: negative-subsample bagging, DIN pooling, field-interaction MLP, docid-masking eval protocol, new-doc AUC metric |
| github.com/andreeaiana/newsreclib (+ docs newsreclib.readthedocs.io) | PyTorch-Lightning framework, ~15 NNR models incl. MANNeR components, standardized small/large benchmarks (EMNLP'23 demo, arXiv:2310.01146) |
| github.com/Jyonn/Legommenders (arXiv:2412.15973, WWW'25) | Modular content-rec library (1,000+ model combos, 15 datasets); hosts ONCE; LLM-as-encoder pipelines, caching of item encodings |
| github.com/Jyonn/ONCE | GPT-based data augmentation prompts (summarizer/user profiler) + LLaMA encoder recipe |
| github.com/resistzzz/Prompt4NR | 12 prompt templates + ensembling scripts on MIND-small |
| github.com/zhengzhi-1997/LLM-TRSR | Hierarchical/recurrent LLM history-summarization prompts + LoRA SFT recommender |
| github.com/zyang1580/CoLLM | CF-embedding-to-token-space mapping, two-stage tuning |
| github.com/RegionCh/SPRec | Self-play DPO loop code |
| github.com/Veason-silverbullet/UniTRec | Discriminative + perplexity score fusion |
| github.com/recsyspolimi/recsys-challenge-2024-ekstrabladet | Full GBDT-stacking news-rec pipeline (CatBoost/LGBM rankers, time features) |
| github.com/recommenders-team/recommenders | Reference NRMS/NAML/LSTUR/… implementations and MIND utilities |
| github.com/msnews/MIND | Official dataset tools; leaderboard submission format |

---

## 6. Ranked actionable ideas for our stack

Ranked by (expected gain on small-dev AUC/MRR) x (transfer confidence) / (effort).

### 1. Pairwise head + right-to-left top-k refinement (GLIMPSE)
- **Source:** arXiv:2409.17711 (Flan-T5-base, same small-dev protocol as ours): pointwise 68.18 → 69.14 AUC (+0.96), MRR +0.32, with 2 RTL passes over top-5 (~8 extra calls/impression).
- **Why it transfers:** drop-in for our SFT: add a second task template ("Which will the user read, A or B?" → logit(A) vs logit(B), Bradley-Terry) mixed into the same training run; inference keeps pointwise ranking and only locally re-sorts the head of the list. AUC gain shrinks as pointwise gets stronger, but AUC is dominated by the full list while the pairwise fix concentrates exactly where MRR/nDCG live; at our 69.6 baseline expect +0.2-0.5 AUC and larger MRR/nDCG@5 moves.
- **Effort:** low-medium. New prompt template + data mixing in `sft_mind_pointwise_ds.py`; small inference-side bubble pass in eval. Also doubles as extra training signal for the pointwise head (multi-task effect).

### 2. S-DPO: slate-level softmax DPO instead of pairwise DPO
- **Source:** arXiv:2406.09215 (NeurIPS'24). Softmax/Plackett-Luce extension of DPO over 1 positive + N negatives; consistently beats pairwise DPO/SFT on three datasets; hard-negative-mining interpretation.
- **Why it transfers:** our DPO/RLCD stage already builds same-impression pos/neg pairs; S-DPO just sums all displayed negatives of that impression into one loss term — exactly matching the per-impression AUC objective (one-vs-rest within slate). Negligible data-pipeline change.
- **Effort:** low. Loss-function change in the contrastive stage; keep β schedule, add N-negative sampling (use the whole impression or cap at ~8 hard negatives by current model score).

### 3. Competition-grade feature set + ranking objective in the GBDT
- **Source:** RecSys'24 winners (arXiv:2409.20483; all top teams GBDT ensembles, timeliness features worth ~12 AUC pts in :D's ablation) + Sogou MIND winner (doc/1.pdf: subsample bagging +0.3, dev-data +0.34).
- **Why it transfers:** our GBDT sits exactly where theirs did. Concrete additions: (a) article age at impression time + time-of-day/weekday; (b) running CTR / impression-count of the candidate *up to the impression timestamp* (within dev-legal data); (c) category/subcategory match rates vs user history, history-recency-weighted; (d) per-impression normalized LLM score (z-score, rank, softmax margin) rather than raw logit; (e) train LightGBM/CatBoost with **per-impression group ranking loss** (lambdarank) instead of binary logloss; (f) bag over negative subsamples and seeds.
- **Effort:** medium, zero GPU. All in `build_ranker_features.py`/`train_ranker.py`. Caution: docid-level CTR features inflate small-dev but would not survive a large-test submission (87.5% new docs) — fine if the target metric is small-dev, but track a "new-doc AUC" diagnostic like Sogou did.

### 4. Abstract/body text in the prompt (if still title-dominant)
- **Source:** Sogou report: +title/abstract 0.6979 → +body 0.707 large-test; +0.3-0.4 dev AUC again from more history bodies (10→30-50). Transformers4NewsRec (arXiv:2410.13125): GLORY small-dev 68.16 → 69.07 adding body.
- **Why it transfers:** content signal is model-agnostic; LLMs read abstracts natively. Add candidate abstract (+first ~50-100 body words) and abstracts for the ~5 most recent/most-similar history items (DIN-style: select history items by similarity to the candidate — we already ablate category-history selection; similarity-to-candidate selection is the stronger variant and is what DIN/UNBERT exploit).
- **Effort:** low-medium; main cost is context length → pair with idea 7 (prefix caching) to keep eval cheap.

### 5. Multi-template / multi-checkpoint score ensembling
- **Source:** Prompt4NR (arXiv:2304.05263): best single template 69.37 → cross-type ensemble 69.64 small (own split); UNBERT-en +1.15 large-test; Sogou +0.3 from negative-subsample bagging.
- **Why it transfers:** cheapest reliable +0.2-0.4. Ensemble (mean of per-impression z-scores) across: 2-3 prompt phrasings (relevance/action/utility framing), the 1.7B and 8B models, SFT and DPO checkpoints, and (free) the existing CF and GBDT scores. Our GBDT can absorb these as features instead of simple averaging.
- **Effort:** low; inference cost x number of variants (mitigated by prefix caching).

### 6. Calibrate every score source before blending; blend in rank space
- **Source:** judge-calibration literature (arXiv:2605.09702: Platt most robust default; beta-calibration with enough data); monotone-invariance argument (AUC unchanged by calibration *per source*, changed by blending).
- **Why it transfers:** our CF-blend and GBDT consume raw logit differences whose scale drifts across checkpoints/users/history-lengths. Platt-scale (or isotonic with CV) the LLM score and CF score on a held-out fold, and/or feed per-impression ranks/z-scores to the blender. Costs nothing, typically worth ~0.1-0.3 when mixing heterogeneous scores.
- **Effort:** trivial (sklearn in `compute_cf_scores.py`/`apply_ranker.py`).

### 7. Shared-prefix KV caching for evaluation throughput (enabler, not AUC)
- **Source:** exact-prefix reuse is lossless and fast (arXiv:2512.11851; vLLM APC / SGLang RadixAttention); avoid cross-candidate KV stitching (arXiv:2601.08343).
- **Why it transfers:** order the prompt as [instructions + user history]-prefix, candidate last; ~37 candidates/impression share one prefill. Enables ideas 1, 4, 5 (longer prompts, more templates, pairwise passes) at ~constant wall-clock. If our 6x pointwise-eval speedup isn't already this, it stacks.
- **Effort:** low-medium (prompt reordering + serving flag).

### 8. CF signal *inside* the prompt (CoLLM-style) / LLM-written user profile (ONCE-style)
- **Source:** CoLLM arXiv:2310.19488 (CF embeddings as mapped soft tokens beat post-hoc text-only + CF baselines); ONCE arXiv:2305.06566 (GPT-generated user-profile/interest lines, up to +19% relative nDCG@5 for its encoder models); LLM-TRSR arXiv:2403.13325 (hierarchical history summary in prompt, best on MIND in its comparison).
- **Why it transfers:** moves the CF/global-popularity signal before the Yes/No decision instead of after. Cheap textual variant first: inject "this article has been clicked by k% of similar users / popularity percentile p / user's top interests: ..." as prompt fields (we already engineered recency tags, so the plumbing exists); soft-token CoLLM variant second if textual helps.
- **Effort:** textual medium; soft-token medium-high (mapping MLP + stage-2 tuning).

**Not recommended / low priority:** zero/few-shot GPT prompting lines (RecPrompt, PNR-LLM — below our baseline); LLM-as-encoder (DIRE/PLM-NR — we already outperform that class on small-dev); graph models (GLORY/DIGAT — gains subsumed by content+history modeling at our operating point); impression-context features if a large-test submission is ever planned (Sogou: +0.4 dev, hurts test).

---

## 7. Sanity anchors for our current number

- Published small-dev(ish) bests: Prompt4NR ensemble 69.64 (own re-split), GLIMPSE 69.14, GLORY+body 69.07, DIGAT ~68.8, PNR-LLM 68.88, UNBERT 67.62. **Our 69.5-69.6 single-stack is already at the published frontier for this split**; the remaining published evidence of headroom is (a) ensembling (+0.3-1.1 wherever tried), (b) pairwise refinement (+0.3-1.0), (c) body text (+0.3-0.9), (d) slate-level preference objectives (unquantified on MIND but consistent across datasets).
- Numbers to distrust: anything quoted "on MIND" without a split (e.g., "MINER 71.51", MDPI "71.72" — could not verify split, MDPI page 403'd); reruns vs originals differ by up to 5-15 AUC pts (survey arXiv:2502.09797, NewsReX).

## Source index

- MIND leaderboard: https://msnews.github.io/ ; competition: https://msnews.github.io/competition.html ; winner report: https://msnews.github.io/assets/doc/1.pdf
- UNBERT: https://www.ijcai.org/proceedings/2021/0462.pdf
- PLM-NR: arXiv:2104.07413 · SpeedyFeed: arXiv:2102.09268 · Fastformer: arXiv:2108.09084
- HieRec: arXiv:2106.04408 · DIGAT: arXiv:2210.05196 · GLoCIM: arXiv:2408.00859 · GLORY: arXiv:2307.06576
- MANNeR: arXiv:2307.16089 · NewsRecLib: arXiv:2310.01146 · Transformers4NewsRec: arXiv:2410.13125 · Legommenders: arXiv:2412.15973
- Prompt4NR: arXiv:2304.05263 · UniTRec: arXiv:2305.15756 · ONCE: arXiv:2305.06566 · LKPNR: arXiv:2308.12028 · RecPrompt: arXiv:2312.10463 · PNR-LLM: arXiv:2504.20452 · LLM-TRSR: arXiv:2403.13325 · PUNR: arXiv:2304.12633
- GLIMPSE: arXiv:2409.17711 · S-DPO: arXiv:2406.09215 · SPRec: arXiv:2412.09243 · CoLLM: arXiv:2310.19488 · LLM vs conventional: arXiv:2503.05493
- Survey: arXiv:2502.09797 · Judge calibration: arXiv:2605.09702 · Prefix caching: arXiv:2512.11851 · KV-reuse caution: arXiv:2601.08343
- RecSys Challenge 2024: arXiv:2409.20483 · PoliMi solution: github.com/recsyspolimi/recsys-challenge-2024-ekstrabladet
