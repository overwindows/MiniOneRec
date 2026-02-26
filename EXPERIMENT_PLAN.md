# MIND Experiment Plan: Path to SOTA

> Generated: 2026-02-25
> Current Best: 69.69% AUC (Point-wise + RL)
> Target: 72.72% AUC (MIND Leaderboard SOTA)
> Gap: ~3%

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
| Zero-shot baseline | Qwen3-4B-Instruct-2507 | MINDlarge | 58.11% | 27.44% | 28.83% | 35.01% |
| Point-wise SFT | Qwen3-4B-Instruct | MINDlarge | 66.30% | 31.28% | 34.77% | 41.10% |
| Point-wise SFT | Qwen3-1.7B | MINDsmall | 67.68% | 33.62% | 37.38% | 43.32% |
| Point-wise SFT | Qwen3-1.7B-Base | MINDlarge | 69.39% | 33.70% | 37.61% | 43.82% |
| Point-wise + RL | Qwen3-1.7B-Base | MINDlarge | **69.69%** | 34.02% | 38.02% | 44.23% |
| Ranking SFT | Qwen3-1.7B-Base | MINDsmall | 66.17% | 45.20% | 50.09% | 56.45% |

### Reference Baselines

| Model | AUC | Source |
|-------|-----|--------|
| NRMS | 67.76% | Published baseline |
| MIND Leaderboard SOTA | ~72.72% | Competition |

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

```bash
# ============================================
# P1.1: More negatives (NEG_RATIO=3.0)
# ============================================
MIND_SIZE=large \
NEG_RATIO=3.0 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# Evaluate
MIND_SIZE=large USE_CHAT_TEMPLATE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_pointwise.sh \
  output_dir/sft_mind_pointwise_large_*_neg3.0_*/final_checkpoint dev

# ============================================
# P1.2: Longer training (NUM_EPOCHS=7)
# ============================================
MIND_SIZE=large \
NUM_EPOCHS=7 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# ============================================
# P1.3: Add abstracts
# ============================================
MIND_SIZE=large \
USE_ABSTRACT=True \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# ============================================
# P1.4: More history (MAX_HISTORY=50)
# ============================================
MIND_SIZE=large \
MAX_HISTORY=50 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# ============================================
# P1.5: Combined best settings
# ============================================
MIND_SIZE=large \
NEG_RATIO=3.0 \
NUM_EPOCHS=7 \
MAX_HISTORY=50 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh

# ============================================
# P1.6: Scale to 8B model
# ============================================
MIND_SIZE=large \
MODEL_PATH=Qwen/Qwen3-8B-Instruct \
MICRO_BATCH_SIZE=1 \
LEARNING_RATE=1e-5 \
USE_CHAT_TEMPLATE=1 \
bash scripts/sft_mind_pointwise_ds.sh
```

### Phase 2: RL Commands

```bash
# ============================================
# R2.1: Asymmetric reward
# ============================================
MODEL_PATH=<best_phase1_checkpoint> \
REWARD_TYPE=pointwise_asymmetric \
bash scripts/rl_mind_pointwise.sh

# ============================================
# R2.2: Lower KL penalty
# ============================================
MODEL_PATH=<best_phase1_checkpoint> \
REWARD_TYPE=pointwise_asymmetric \
KL_COEF=0.05 \
bash scripts/rl_mind_pointwise.sh

# ============================================
# R2.3: More RL epochs
# ============================================
MODEL_PATH=<best_phase1_checkpoint> \
REWARD_TYPE=pointwise_asymmetric \
TOTAL_EPOCHS=2 \
bash scripts/rl_mind_pointwise.sh
```

### Phase 3: Ensemble & Cascade Commands

```bash
# ============================================
# E3.1: Ensemble with α=0.7 (more point-wise)
# ============================================
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
ALPHA=0.7 \
MIND_SIZE=large \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ensemble.sh dev

# ============================================
# E3.2: Ensemble with α=0.5 (balanced)
# ============================================
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
ALPHA=0.5 \
MIND_SIZE=large \
bash scripts/eval_mind_ensemble.sh dev

# ============================================
# E3.4: Cascade with TOP_K=15
# ============================================
POINTWISE_MODEL=<best_pointwise_checkpoint> \
RANKING_MODEL=<best_ranking_checkpoint> \
TOP_K=15 \
MIND_SIZE=large \
bash scripts/eval_mind_cascade.sh dev
```

### Phase 4: Multi-task Commands

```bash
# ============================================
# M4.1: Multi-task with 70% point-wise
# ============================================
MIND_SIZE=large \
POINTWISE_RATIO=0.7 \
USE_CHAT_TEMPLATE=1 \
MAX_HISTORY=30 \
bash scripts/sft_mind_multitask.sh

# ============================================
# M4.2: Multi-task with 80% point-wise
# ============================================
MIND_SIZE=large \
POINTWISE_RATIO=0.8 \
USE_CHAT_TEMPLATE=1 \
MAX_HISTORY=30 \
bash scripts/sft_mind_multitask.sh
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
# Point-wise evaluation (multi-GPU)
MIND_SIZE=large USE_CHAT_TEMPLATE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_pointwise.sh <checkpoint> dev

# Ranking evaluation (multi-GPU)
MIND_SIZE=large USE_CHAT_TEMPLATE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/eval_mind_ranking.sh <checkpoint> dev

# Quick test (100 impressions)
MIND_SIZE=large USE_CHAT_TEMPLATE=1 \
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
- **MIND_SIZE=large**: Use large dataset for SOTA attempts
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
├── scripts/
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

---

*Last updated: 2026-02-25*
