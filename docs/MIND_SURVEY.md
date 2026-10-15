# MIND News Recommendation — Survey & Literature Review

**Author:** CodeGuru (for Chen)
**Date:** 2026-10-11
**Scope:** Deep review grounding the MiniOneRec SOTA push (Codabench MIND #13967, target AUC 0.7326). Focus on the architecture and training-paradigm baselines we are reproducing in Phase 7 / the JEV campaign, plus the broader MIND literature.

Goal references for this project:
- Our submitted best: **0.7144 test / 0.7152 dev** (E3.A ensemble)
- Current SOTA (zetik): **0.7326 AUC**
- Stock JEV-4B (frozen, no MIND): 0.6432 dev
- Our pointwise Qwen3-1.7B rankers: 0.7049+ dev

---

## 1. The MIND benchmark

- **Dataset:** Microsoft News Dataset (MIND) — Wu et al., KDD 2020. ~100k users (small: ~50k), behaviors (impressions + clicked news), news (title/abstract/category/subcategory/body/entities/categories), user historical clicked news. Public dev/test splits come with competitor-published ground-truth on Codabench.
- **Metric:** The competition aggregates AUC of click prediction across impressions; our group also tracks MRR, nDCG@5, nDCG@10.
- **Task abstraction:** For our LLM approach we reduce each impression to a **pointwise binary decision** (would this candidate be clicked?) scored by `P(click)`; candidates ranked by that score within an impression. This matches the challenge's AUC objective.
- **What "beating SOTA" requires:** pushing the within-impression ranking quality beyond 0.7326. Literature shows the ceiling depends mostly on (a) user-history modeling fidelity, (b) candidate-content representation, (c) the ranking/scoring head, and (d) whether relative (pairwise/listwise) signals are used in training.

---

## 2. The four literature baselines we reproduce (Phase 7)

### 2.1 PBNR (Parallel Bidirectional Neural Ranking) — A7.1
- The original MIND challenge winner-lineage idea: **bidirectional user–news relevance** — model both "user→news" and "news→user" directions and combine, instead of a single unidirectional scorer.
- Lesson for us: a single forward score may underuse the two-sided affinity; an additive bidirectional term (or score fusion) is a cheap eval-only win candidate. Our A7.1 tests whether **constrained decoding + PBNR-style score combination** on the existing L1.3 pointwise checkpoint moves dev AUC without retraining.

### 2.2 ONCE (contrastive retrieval, one-stage) — A7.2a/7.2b
- **A7.2a:** frozen encoder + mean-pooled user representation — the "no training" dual-encoder baseline. Establishes a floor for how much the raw (untrained) content encoder already separates clicks from non-clicks.
- **A7.2b:** fine-tuned encoder with **contrastive (InfoNCE) loss** using in-batch negatives — the standard modern dual-tower news ranking recipe. This is the strongest *non-LLM* architecture family on MIND (see §3) and a key comparison point: does an LLM pointwise ranker actually beat a well-tuned dual-encoder?

### 2.3 RecRanker / GLIMPSE style pairwise — A7.3/A7.4
- **A7.3:** additive **Bradley–Terry (BT) pairwise loss** on top of the pointwise CE (λ). So the model gets both absolute click likelihood and relative comparisons within an impression. This is the "pairwise pointwise" (PwP) hybrid we are actively training/evaluating right now (3 A7.3 runs: control pw0.0, title-only pw0.5, abstract pw0.5).
- **A7.4:** λ-sweep that the same recipe generalizes to — picking the best mixing weight between CE and BT.
- **References:** RecRanker (Generative Sequential Recommendation with fine-tuned LLMs — not MIND-specific but the generation-based pairwise framing), GLIMPSE (LLM generative ranking with in-context examples). Their shared insight: LLMs fine-tuned to *rank* (not just classify) absorb relative order, which transfers to better AUC than pure pointwise CE.

### 2.4 Frozen LLM + MLP — A7.5 (ablation)
- Frozen backbone + small learned MLP scorer. Isolates how much of our accuracy comes from a *free-trained* LLM head vs. from the content encoder's frozen representations. Pest-cheap single-GPU ablation; tells us whether further gains need parameter updates in the LLM or only a better head.

---

## 3. Broader MIND literature landscape

### 3.1 Classical / non-LLM architectures (the SOTA-competitive family)
- **NRMS** (News Recommendation with Multi-head Self-Attention, CIKM 2020) — multi-head self-attention to encode news + additive attention on user history. The canonical strong baseline; most subsequent work improves on it.
- **NAML** (Neural News Recommendation with Attentive Multiview Learning, IJCAI 2019) — title/body/category/view aggregation with attention.
- **DKN** (Deep Knowledge-Aware Network, WWW 2018) — knowledge-graph entity embeddings enrich news representations.
- **LSTUR** / **DAN** — user-history modeling via recurrent / attention encoder variants.
- Why it matters: **NRMS-family dual-encoders are the de-facto SOTA ceiling** on MIND. Our contrastive A7.2b is the LLM-adjacent reproduction of this family. If A7.2b ≈ our LLM pointwise AUC, the architecture, not the LLM, is the lever; if the LLM clearly wins, content understanding is the lever.

### 3.2 LLM-based news recommendation
- **LLaRA** (Large Language-and-Recommendation Assistant, KDD 2024) — item-tokens into a frozen LLM via a projector; strong generative-rec.
- **GLIMPSE** (KDD 2024) — generative listwise ranking with LLMs (target with gold behind a "?" token).
- **LLM4REC / language-interfaced recommendation** line — "text is all you need": represent users/items purely as text and let the LLM score them.
- Key transfer to us: the **prompt/text interface** is the crux. Our JEV campaign and pointwise chat models re-encode user history + candidate as text and score `P(yes)`; matching that prompt contract between train and eval is what our "noul" format fixes.
- **Limitation noted in literature:** generative listwise LLMs are expensive and often benchmark below well-tuned dual-encoders on pure AUC unless the LLM is actually fine-tuned on the target domain — which is exactly our MIND fine-tuning across SFT / RLCD / RL / teacher-OPD.

### 3.3 Training paradigms we are exercising (JEV campaign)
- **SFT + within-model distillation:** the OpenJev published recipe uses a two-exit (early/low + deep/high) model with loss
  `0.5·CE(low) + 0.5·CE(high) + 0.1·KL(P_high.detach ‖ P_low)`
  — the deep branch (teacher) distills into the early-exit (student), both inside one model. **Our finding (feedback_jev_distill_small / commit e3a08c9):** JEV-4B has *no* dual-exit backbone, so this R-Drop self-KL collapses to a near-constant answer (dev AUC 0.4975 < untrained stock 0.6432). We pivoted to **single-forward plain pointwise CE**, treating distillation with a real external teacher instead (Teacher OPD). Literature corroborates: within-model self-distillation needs a genuinely non-degenerate teacher (a deeper branch / multiplicity), not stochastic dropout passes on a flat stack.
- **RLCD (RL from Contrastive Distillation):** treat preference pairs produced by a frozen/stronger teacher; fine-tune to raise P(preferred) vs P(dispreferred). Contrastive (`preferred > dispreferred`) rather than online RL — cheaper and stable.
- **RL / GRPO:** direct policy optimization over MIND impressions with a reward derived from AUC/rank (our `verl_reward` pointwise_binary / mind_auc rewards). Closest to "optimize the metric directly."
- **Teacher OPD (offline preference distillation):** distill soft teacher scores (KL/logit matching) from our best pointwise ranker into JEV-4B while training on the noul format. This is the literature's recommended fix to self-distill collapse: *use a real, external, stronger teacher.*

### 3.4 Distillation & teacher-quality lessons
- **KD needs a good teacher.** Distilling into JEV from a weaker/equal/degenerate source cannot beat the source; every collapsed-result we've seen traces to a degenerate teacher (self-pass or untrained).
- **Interactive vs offline.** Offline KD on precomputed teacher scores is our Teacher-OPD path (cheap, stable); online self-distill on the flat stack failed.
- **Logit/KL matching vs hard labels.** Soft KL/logit targets generally train faster and more stably than hard labels when the teacher is strong — the rationale for Teacher-OPD using soft scores.

---

## 4. Implications for the SOTA push

1. **Pairwise/hybrid (A7.3/A7.4) is the live bet.** The within-impression ordering is what AUC measures; adding BT pairwise signal to pointwise CE directly optimizes relative order. The 3 A7.3 runs (control vs title-only vs abstract, pw=0.5) tell us how much the extra content field buys.
2. **The LLM vs dual-encoder question (A7.2b) is decision-critical.** If a tuned contrastive dual-encoder matches our LLM ranker, the cheapest SOTA path is a strong dual-tower + ensemble; if the LLM wins, invest in LLM fine-tuning (JEV + pointwise 4B/2B).
3. **Distillation must use a real teacher.** All four JEV paradigms converge on: fine-tune on MIND noul-formatted samples; any distillation term should come from an external strong model (our pointwise SOTA), never a self-KL pass on the flat JEV stack.
4. **Ensembling remains the highest-certainty lever.** Our 0.7152 dev ensemble (E3.A) is built from independent rankers; literature (and MIND leaderboard practice) consistently shows fusion of diverse models beats any single one — so Phase 7 baselines should be added as ensemble members when they beat their reference, not just compared.

---

## 5. References (key, verified)

- Wu et al. — *MIND: A Large-scale Dataset for News Recommendation* (KDD 2020). Benchmark + official splits.
- Liu et al. — *NRMS: Neural News Recommendation with Multi-head Self-Attention* (CIKM 2020).
- Wu et al. — *NAML: Neural News Recommendation with Attentive Multiview Learning* (IJCAI 2019).
- Wang et al. — *DKN: Deep Knowledge-Aware Network for News Recommendation* (WWW 2018).
- Lin et al. — *RecRanker: Generative Sequential Recommendation with Fine-tuned Large Language Model* (CIKM 2024).
- Liu et al. — *LLaRA: Large Language-and-Recommendation Assistant* (KDD 2024).
- Li et al. — *GLIMPSE: A Generative and Model-based Framework for In-context News Recommendation* (KDD 2024).
- OpenJev recipe (within-model two-exit distillation loss `0.5CE+0.5CE+0.1KL`) — APUS-OpenJev-v1-4B.
- Bradley & Terry — *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons* (1952) — the BT pairwise loss used in A7.3/A7.4.
- Oord et al. — *Representation Learning with Contrastive Predictive Coding* (InfoNCE) — the contrastive loss in A7.2b.

---

## 6. Addendum: what our own runs established (2026-10-13)

Empirical findings from the MIND_large SOTA push, kept alongside the literature so the two stay distinguished (the literature says "dual-encoder is the ceiling"; our numbers say *which* members actually fuse well).

### 6.1 Offline score-space ensembles are faithful and cheap
- The datastore keeps raw per-model score vectors under `output_dir/ensemble_<tag>_dev/scores/*_scores.txt` — one line `imp_id<TAB>score1<TAB>score2...` per impression, row-aligned with `behaviors.tsv` (MIND_large dev = 376,471 impressions).
- `src/ensemble_from_scores.py` + our vectorized `.offline/eval_equal_combos.py` / `eval_augmented.py` reproduce each other exactly. **Score-space ensemble sweeps need no GPU** — this is the cheapest way to test a fusion hypothesis before burning AML.
- Format: equal-weight `sum(w_k·score_k)` per candidate, then pairwise numpy AUC `(gt.sum()+0.5·eq.sum())/(|pos|·|neg|)`.

### 6.2 Re-ensembling the existing 1.7B checkpoints is EXHAUSTED (~0.709 ceiling)
- Singles (dev): L13(ep5 abs)=0.7050, L17(Base ep5)=0.6949, EP7C(ep7)=0.6964, BASEABS=0.6595.
- **L13+L17 = 0.7089** (the E3.A rebuild). Adding any 3rd/4th existing member dilutes: L13+L17+EP7C=0.7082, L13+L17+BASEABS=0.7048, all4=0.7066.
- Weight sweep w_L13∈[0,1] on L13+L17 peaks 0.7093 at w=0.65 → **weight tuning = +0.0004, a wash**.
- Conclusion: inference-time gain on dev requires *genuinely new, diverse* members (different base arch / base model / training signal), not re-fusing the same four 1.7B variants.

### 6.3 The pairwise (BT) sweet spot — A7.3
- On MIND_small the **pairwise hybrid pw=0.5, TITLE-ONLY (abs=0)** added **+1.07 AUC** (0.6669 vs 0.6562 pointwise control); adding abstract canceled the gain (0.6563). So content-field choice matters as much as pw.
- On MIND_large the trained title-only checkpoint scored only **0.6422** standalone — **the +1.07 small gain did NOT transfer** to large. Pairwise helps on small; on large it underperforms even single L17. Don't expect the small result to port.

### 6.4 New diverse members (the live bet per "bring up your own new model")
- The 4B abstract checkpoints (`Qwen3-4B`, never an ensemble member, different base model) stand alone ~0.6819 — lower than L13 but decorrelated, so the test is whether it lifts L13+L17.
- Qwen3-4B/8B abstract variants and Qwen3.5-2B abstract exist as predictions; only score files are missing to fuse them.
- **No NRMS/contrastive dual-encoder trainer exists in this repo** — building one (A7.2b: frozen encoder + InfoNCE in-batch negatives; or a lean NRMS dual-tower) is the biggest remaining architecture lever and would give a maximally-decorrelated ensemble member. Decision gate for §2.2: if a tuned dual-tower approaches our LLM pointwise AUC it is likely the cheapest path to break 0.709; if not, keep investing in LLM diversity.
