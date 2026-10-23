# MIND Experiment Plan: Path to SOTA

> Generated: 2026-02-25
> Updated: 2026-09-27
> Current Best (submitted): **71.44% AUC on test** / 71.52% dev (E3.A: L1.3 + L1.7 ensemble)
> **Rank: #14 / 101 on Codabench (MIND News Recommendation, competition 13967)**
>
> **Live leaderboard (2026-09-27, private/test):**
> - #1 SOTA: **zetik 0.7326** AUC (2026-06-23) — target to beat
> - #2: newsbang 0.7316 · #3: i4never 0.7258 · #4: qinzixuan 0.7249 · #5: jianchunyang 0.7226
> - **overwindows: 0.7144 AUC** (MRR 0.3621, nDCG@5 0.3975, nDCG@10 0.4539), 2026-04-21
>
> **Gap to #1 zetik (SOTA): +0.0182 AUC (~1.8%)** · to top-5 (+0.7226): +0.0082 · to top-3: +0.0114
> Need ~1.8% AUC gain over E3.A to become SOTA → stack multiple independent gains (ensemble + CF + pairwise/constrained-decoding + scaling).

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

**Progress**: 0/21 (0%) | **Current Best**: 69.69% AUC (Baseline) | **Target**: 72.72% AUC | **Gap**: 3.03%

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
