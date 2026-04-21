# MIND Experiment Plan: Path to SOTA

> Generated: 2026-02-25
> Updated: 2026-04-21
> Current Best: **71.44% AUC on test** / 71.52% dev (E3.A: L1.3 + L1.7 ensemble)
> Target: 72.72% AUC (MIND Leaderboard SOTA)
> Gap: ~1.28%

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

> **Training with abstracts but evaluating without abstracts can yield higher scores.**
>
> Models trained with `USE_ABSTRACT=1` appear to achieve better evaluation metrics when
> evaluated with `USE_ABSTRACT=0` (title-only). Hypothesis: abstracts provide richer
> training signal — the model learns deeper content/category understanding — but at eval
> time, shorter title-only prompts reduce noise and let the model focus on the learned
> topic signals. This means L1.3's reported 0.7049 AUC (evaluated with abstracts) may
> understate the model's true capability; re-evaluating L1.3 without abstracts is recommended.

---

### Phase 1L: Point-wise SFT on MINDlarge

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **L1.1** | ✅ Completed | Qwen3-1.7B | MINDlarge | NEG_RATIO=3.0, HIST=30 | 0.6883 | 0.3316 | 0.3678 | 0.4310 | NEG=3.0 hurts on large too; [W&B](https://wandb.ai/wuchen/huggingface/runs/49ovq991) |
| **L1.2** | ✅ Completed | Qwen3-1.7B | MINDlarge | NUM_EPOCHS=7 | 0.6932 | 0.3359 | 0.3738 | 0.4373 | checkpoint-70144; [W&B](https://wandb.ai/wuchen/huggingface/runs/h70c32sr) |
| **L1.3** | ✅ Completed | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1 | **0.7049** 🏆 | 0.3461 | 0.3847 | 0.4479 | **NEW BEST**; [W&B](https://wandb.ai/wuchen/huggingface/runs/d3dr5t8d) |
| **L1.4** | ✅ Completed | Qwen3-1.7B | MINDlarge | MAX_HISTORY=50 | 0.6863 | 0.3310 | 0.3675 | 0.4307 | HIST=50 hurts on large; [W&B](https://wandb.ai/wuchen/huggingface/runs/gskwfdbe) |
| **L1.5** | ✅ Completed | Qwen3-1.7B | MINDlarge | NEG=3.0, EP=7, HIST=50 | 0.6863 | 0.3310 | 0.3675 | 0.4307 | HIST=50 bottleneck; [W&B](https://wandb.ai/wuchen/huggingface/runs/36ur6v25) |
| **L1.6** | 🔄 Retrying | Qwen3-4B | MINDlarge | USE_ABSTRACT=1, micro_bs=4 | - | - | - | - | Replacing failed 8B run; same best config as L1.3 |
| **L1.7** | ✅ Completed | Qwen3-1.7B-Base | MINDlarge | Base model (non-instruct) | 0.6946 | 0.3373 | 0.3767 | 0.4392 | No chat template; [W&B](https://wandb.ai/wuchen/MIND/runs/i9h0gzwr) |
| **L1.8** | ❌ Failed | Qwen3-1.7B | MINDlarge | USE_ABSTRACT=1, EP=7, micro_bs=2 | 0.6622 | 0.3259 | 0.3621 | 0.4222 | Worse than L1.3 (0.7049); output path shows Base/ep5/no-abstract — likely misconfigured run or job ran wrong parameters; 8d total runtime vs 31d ETA is suspicious; verify AML job params; [W&B](https://wandb.ai/wuchen/huggingface/runs/p03ej16f) |
| **L1.9** | ✅ Completed | Qwen3-1.7B-Base | MINDlarge | USE_ABSTRACT=1, Base model | 0.6880 | 0.3368 | 0.3769 | 0.4382 | Abstract didn't boost base; [W&B](https://wandb.ai/wuchen/huggingface/runs/gzijusub) |

---

### Phase 1E: Abstract Train → Title-Only Eval (Cross-Condition)

> **Rationale**: Abstracts act as training-time augmentation — the model learns richer content
> representations — but at eval time, title-only prompts are more compact and signal-dense
> for a small (1.7B) model. Re-evaluate all abstract-trained checkpoints with `USE_ABSTRACT=0`.

| Exp ID | Status | Source | Eval Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|--------|-------------|-----|-----|--------|---------|-------|
| **E1.1** | ⬜ Pending | L1.3 (abstract-trained) | eval USE_ABSTRACT=0 | - | - | - | - | A/B test: compare vs L1.3's 0.7049 (eval w/ abstract) |
| **E1.2** | ⬜ Pending | L1.3 (abstract-trained) | eval USE_ABSTRACT=1 | - | - | - | - | A/B control: re-eval w/ abstract after eval pipeline fix |
| **E1.3** | ⬜ Pending | L1.9 (base, abstract-trained) | eval USE_ABSTRACT=0 | - | - | - | - | Same test on base model; compare vs L1.9's 0.6880 |
| **E1.4** | ⬜ Pending | L1.6 (4B, abstract-trained) | eval USE_ABSTRACT=0 | - | - | - | - | Run after L1.6 training completes |

**If E1.1 > L1.3**: adopt "abstract train + title eval" as default for all future abstract-trained models.

---

### Phase 4M: Multi-task (Extra Checkpoints)

| Exp ID | Status | Model | Dataset | Config | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|--------|--------|-------|---------|--------|-----|-----|--------|---------|-------|
| **M4.0** | ❌ Failed | Qwen3-1.7B | MINDsmall | POINTWISE_RATIO=0.7, ep3 | 0.6213 | 0.2833 | 0.3124 | 0.3765 | Multitask hurts badly; [W&B](https://wandb.ai/wuchen/MIND/runs/5nb9n11h) |
| **M4.1** | ⬜ Pending | Qwen3-1.7B | MINDlarge | POINTWISE_RATIO=0.5, EP=5, Abstract | - | - | - | - | Proper test: large data + abstract + 5 epochs + pipeline |

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
| **A5.1** | ❌ Crashed | Qwen3-1.7B | CoT RL | - | - | - | - | Chain-of-Thought; [W&B](https://wandb.ai/wuchen/MiniOneRec_MIND/runs/d00m4zay) |
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
| +0.5% | 70.2% | ⬜ Pending |
| +1.0% | 70.7% | ⬜ Pending |
| +1.5% | 71.2% | ⬜ Pending |
| +2.0% | 71.7% | ⬜ Pending |
| SOTA | 72.7% | ⬜ Target |

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

*Last updated: 2026-03-01*
