# MIND Experiment Plan: Path to SOTA

> Generated: 2026-02-25
> Updated: 2026-02-26 (Azure ML Pipeline Integration)
> Current Best: 69.69% AUC (Point-wise + RL)
> Target: 72.72% AUC (MIND Leaderboard SOTA)
> Gap: ~3%

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
| **P1.1** | ⬜ Pending | Qwen3-1.7B | MINDsmall | NEG_RATIO=3.0 | - | - | - | - | More negatives |
| **P1.2** | ⬜ Pending | Qwen3-1.7B | MINDsmall | NUM_EPOCHS=7 | - | - | - | - | Longer training |
| **P1.3** | ⬜ Pending | Qwen3-1.7B | MINDsmall | USE_ABSTRACT=1 | - | - | - | - | With abstracts (local only) |
| **P1.4** | ⬜ Pending | Qwen3-1.7B | MINDsmall | MAX_HISTORY=50 | - | - | - | - | More history |
| **P1.5** | ⬜ Pending | Qwen3-1.7B | MINDsmall | NEG=3.0, EP=7, HIST=50 | - | - | - | - | Combined best |
| **P1.6** | ⬜ Pending | Qwen3-8B-Instruct | MINDsmall | Default + 8B model | - | - | - | - | Scale to 8B |

**Status Legend**: ⬜ Pending | 🔄 Running | ✅ Completed | ❌ Failed

### Phase 2: RL Fine-tuning

| Exp ID | Status | Base Model | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|------------|--------|-----|-----|--------|---------|-------|
| **R2.1** | ⬜ Pending | Best P1.x | REWARD_TYPE=asymmetric | - | - | - | - | Asymmetric reward |
| **R2.2** | ⬜ Pending | Best P1.x | KL_COEF=0.05 | - | - | - | - | Lower KL penalty |
| **R2.3** | ⬜ Pending | Best P1.x | TOTAL_EPOCHS=2 | - | - | - | - | More RL epochs |
| **R2.4** | ⬜ Pending | Best P1.6 | RL on 8B model | - | - | - | - | RL on larger model |

### Phase 3: Ensemble & Cascade

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
| **A5.1** | ⬜ Pending | Qwen3-1.7B | CoT RL | - | - | - | - | Chain-of-Thought |
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

## 🔧 Pipeline Support Matrix

| Experiment | Azure ML Pipeline | Local Scripts | Notes |
|------------|-------------------|---------------|-------|
| **P1.1** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.2** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.3** | ❌ Training / ✅ Eval | ✅ Required | Abstract parameter not in pipeline |
| **P1.4** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.5** | ✅ Training + Eval | ✅ Available | Full pipeline support |
| **P1.6** | ✅ Training + Eval | ✅ Available | Full pipeline support (8B model) |
| **R2.1-R2.4** | ❌ Training / ✅ Eval | ✅ Required | RL not yet in pipeline |
| **E3.1-E3.5** | ❌ Not applicable | ✅ Required | Ensemble/cascade local only |
| **M4.1-M4.3** | ❌ Training / ✅ Eval | ✅ Required | Multi-task not yet in pipeline |
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
| P1.6 | Scale to 8B | `MODEL_PATH=Qwen3-8B-Instruct` | 70.5%+ |

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
# python pipeline/run_eval_pipeline.py \
#   --experiment-name mind_eval_p1-1 \
#   --display-name "P1.1 Eval: neg=3.0" \
#   --model-path shares/users/wuc/output_dir/sft_mind_pointwise_*/final_checkpoint \
#   --data-root shares/users/wuc/data/MIND_small \
#   --eval-type pointwise \
#   --split dev

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

# ============================================
# P1.3: Add abstracts (Note: requires pipeline update)
# ============================================
# TODO: Add USE_ABSTRACT parameter to pipeline
# For now, use local script:
# MIND_SIZE=small USE_ABSTRACT=True USE_CHAT_TEMPLATE=1 bash scripts/sft_mind_pointwise_ds.sh

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

# ============================================
# P1.6: Scale to 8B model
# ============================================
python pipeline/run_pipeline.py \
  --experiment-name mind_sft_p1-6_8b \
  --display-name "P1.6: Scale to 8B model" \
  --model-path Qwen/Qwen3-8B-Instruct \
  --data-root shares/users/wuc/data/MIND_small \
  --output-root shares/users/wuc/output_dir \
  --batch-size 256 \
  --micro-batch-size 1 \
  --num-epochs 5 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1 \
  --run-eval 1 \
  --eval-split dev
  # --debug
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
  --model-path shares/users/wuc/models/<rl_checkpoint>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug
```

### Phase 3: Ensemble & Cascade Commands

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
  --model-path shares/users/wuc/models/<multitask_checkpoint>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug

# Evaluate multi-task/two-stage models (ranking)
python pipeline/run_eval_pipeline.py \
  --experiment-name mind_eval_m4-x_ranking \
  --display-name "M4.x Eval: multi-task (ranking)" \
  --model-path shares/users/wuc/models/<multitask_checkpoint>/final_checkpoint \
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
| P1.6 (8B model) | 16h | If resources allow |

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

### Quick Evaluation Commands

```bash
# ============================================
# Azure ML Pipeline Evaluation (Recommended)
# ============================================

# Point-wise evaluation on dev set
python pipeline/run_eval_pipeline.py \
  --experiment-name quick_eval_pointwise \
  --display-name "Quick Eval: pointwise (dev)" \
  --model-path shares/users/wuc/models/<checkpoint_name>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type pointwise \
  --split dev
  # --debug

# Ranking evaluation on dev set
python pipeline/run_eval_pipeline.py \
  --experiment-name quick_eval_ranking \
  --display-name "Quick Eval: ranking (dev)" \
  --model-path shares/users/wuc/models/<checkpoint_name>/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_small \
  --eval-type ranking \
  --split dev
  # --debug

# Test set evaluation (for submission)
python pipeline/run_eval_pipeline.py \
  --experiment-name test_eval \
  --display-name "Test Eval: submission" \
  --model-path shares/users/wuc/models/<checkpoint_name>/final_checkpoint \
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
| +0.5% | 70.2% | ⬜ Pending |
| +1.0% | 70.7% | ⬜ Pending |
| +1.5% | 71.2% | ⬜ Pending |
| +2.0% | 71.7% | ⬜ Pending |
| SOTA | 72.7% | ⬜ Target |

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

*Last updated: 2026-02-27*
