<div align="center">


<img src="./assets/logo.png" width="500em" ></img> 

**An Open-Source Framework for
Scaling Generative Recommendation**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)
<a href="https://arxiv.org/abs/2510.24431"><img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red"></a>

<a href="https://arxiv.org/abs/2510.24431">📄 Technical Report</a> | <a href="https://huggingface.co/kkknight/MiniOneRec">🤗 Huggingface</a> | <a href="https://modelscope.cn/models/k925238839/MiniOneRec">🤖  Modelscope</a>
</div>

**MiniOneRec** is the first fully open-source **generative recommendation** framework, which provides an end-to-end workflow spanning **SID construction**, **supervised fine-tuning (SFT)**, and recommendation-oriented **reinforcement learning (RL)**. 

---

## 📢 Announcement

- 2025-12-04 — We update new scripts to support processing the Amazon23 dataset.

- 2025-12-01 — We fix a bug in data.py that could cause the SID–item alignment task to see the answers in advance. This was because we had previously attempted to use partial trajectories to guide the full SID–item generation and does not affect the model performance.

- 2025-11-20 — The SID construction method in **RQ-Kmeans+** has been updated (first proposed in **GPR** and this is the first open-source reproduction).

- 2025-11-19 — We implemented a multi-GPU parallel text-to-embedding method based on Accelerate, which is significantly more efficient than the original version: rq/text2emb/amazon_text2emb.py

- 2025-11-19 — The SID construction method in **constrained-RQ-Kmeans** has been updated.

- 2025-11-07 — Thank you for submitting issues! Based on your feedback, we have released a new implementation. If you encounter any problems while running the code, please update to and consult the **latest version** first.
  
- 2025-11-07 — You can now choose to freeze the LLM parameters during the SFT stage and train only the embeddings for the newly added SID vocabulary.

- 2025-10-31 — You can now directly download the implementation **checkpoints** of our MiniOnRec model.

- 2025-10-31 — The SID construction method in **RQ-Kmeans** has been updated.

---

## 🛠️ Key Techniques 
<div align="center">
<img src="./assets/minionerec_framework.png" width=100% ></img> 
</div>

- **SID Construction: MiniOneRec begins by transforming every product into a compact, semantically meaningful token.** It concatenates an item’s title and description, feeds this sentence through a frozen text encoder, and then quantises the resulting embedding with a three-level RQ-VAE.

- **SFT: With all items rewritten as SIDs, the model is first trained in a supervised fashion.** It views the chronologically ordered user history as a token sequence and learns, via next-token prediction, to generate the SID of the next product the user is likely to consume. Crucially, this stage is co-trained with a set of language-alignment objectives that map back and forth between natural language and SID space, allowing the recommender to inherit the world knowledge embedded in large language models while grounding that knowledge in discrete item codes.

- **Recommendation-Oriented RL: After SFT, MiniOneRec is further polished with a recommendation-oriented RL phase based on GRPO.** Multiple candidate recommendations are generated for each prompt, their rewards are normalised within the group to stabilise gradients, and a KL penalty keeps the updated policy close to its reference. Because the action space is a closed list of item SIDs, the system switches to constrained beam search, which guarantees that every beam is unique and valid, greatly improving sampling efficiency and diversity. The reward signal itself blends a binary correctness term with a rank-aware component that penalises high-probability yet incorrect items more heavily, and can be augmented with collaborative-filtering scores. Together, this pipeline enables MiniOneRec to couple dense linguistic knowledge, achieving a high-performance, lightweight generative recommendation system.

---

## 📊 Evaluation

<div align="center">
<img src="./assets/minionerec_main_result.png" width=100% ></img> 
</div>

---

## 🧪 LLM Capability Evaluation (Optional)

If you want to track general LLM capability during SFT/RL (e.g., MMLU, GSM8K), this repo includes a lightweight wrapper around [lm-eval-harness](https://github.com/EleutherAI/lm-eval-harness).

### Evaluation Results

| Model | MMLU | HellaSwag | ARC-Challenge | Winogrande | GSM8K | IFEval |
|-------|------|-----------|---------------|------------|-------|--------|
| Qwen3-8B | 73.01% | 74.97% | 56.40% | 67.80% | 60.27% | 24.95% |
| Qwen3-4B-Instruct-2507 | 70.65% | 69.06% | 58.45% | 68.03% | 71.27% | 57.67% |
| Qwen3-1.7B | 55.48% | 60.40% | 42.83% | 60.93% | 42.00% | 16.64% |
| Qwen3-1.7B-SFT (Amazon) | 26.11% | 45.77% | 32.94% | 53.67% | 0.00% | 10.54% |
| Qwen3-1.7B-SFT-Mixed (Amazon+UltraChat) | 45.53% | 55.17% | 43.09% | 55.41% | 14.94% | 7.39% |
| Qwen3-1.7B-RL (Amazon)  | 25.65% | 47.66% | 34.47% | 53.04% | 0.53% | 10.91% |
| Qwen3-1.7B-RL-Mixed (Amazon+UltraChat) | 46.08% | 55.86% | 42.75% | 56.59% | 12.36% | 15.90% |

**Note:** SFT/RL models show significant decrease in general capabilities due to domain specialization on Amazon recommendation data.

### Quick Start: Download Once, Evaluate Multiple Times (Recommended)

**Step 1: Install evaluation dependencies**
```bash
pip install -r requirements-eval.txt
```

**Step 2: Pre-download datasets (once)**
```bash
python download_eval_datasets.py \
  --tasks "mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval" \
  --cache_dir ~/.cache/huggingface/datasets \
  --verbose
```

This downloads all datasets to `~/.cache/huggingface/datasets/` and takes 5-30 minutes.

**Step 3: Run evaluation in OFFLINE MODE (no re-downloads)**
```bash
# Single GPU
SKIP_DOWNLOAD=1 bash scripts/eval_llm.sh Qwen/Qwen3-1.7B

# With higher batch size for more speed
SKIP_DOWNLOAD=1 BATCH_SIZE=32 bash scripts/eval_llm.sh Qwen/Qwen3-1.7B

# Evaluate checkpoint
SKIP_DOWNLOAD=1 bash scripts/eval_llm.sh output_dir/my_checkpoint/final_checkpoint
```

### Common Workflows

**Workflow 1: Single GPU (simplest)**
```bash
# Download once
python download_eval_datasets.py --cache_dir ~/.cache/huggingface/datasets

# Evaluate multiple times, offline mode
SKIP_DOWNLOAD=1 bash scripts/eval_llm.sh Qwen/Qwen3-1.7B
SKIP_DOWNLOAD=1 bash scripts/eval_llm.sh output_dir/my_checkpoint/final_checkpoint
```

**Workflow 2: Custom tasks and settings**
```bash
# Download only specific tasks
python download_eval_datasets.py \
  --tasks "mmlu,hellaswag" \
  --cache_dir ~/.cache/huggingface/datasets

# Evaluate with custom batch size
SKIP_DOWNLOAD=1 BATCH_SIZE=32 bash scripts/eval_llm.sh Qwen/Qwen3-1.7B "mmlu,hellaswag"

# Evaluate only subset (e.g., first 10% of examples)
SKIP_DOWNLOAD=1 bash scripts/eval_llm.sh Qwen/Qwen3-1.7B "mmlu,hellaswag" llm_eval 0.1
```

### Offline vs Online Mode

**Offline Mode (RECOMMENDED)**
- **Use pre-downloaded datasets**: No network access needed during evaluation
- **Avoids FUSE conflicts**: Multiple GPUs work independently on cached data
- **Fast**: Pure GPU computation, no I/O bottlenecks
- **Enabled by**: `SKIP_DOWNLOAD=1` environment variable

**Online Mode (NOT RECOMMENDED for production)**
- **Downloads on-the-fly**: Datasets downloaded during evaluation if not cached
- **FUSE conflicts**: Multiple concurrent downloads may fail
- **Slower**: Network I/O interferes with GPU computation
- **Only use if**: Single GPU and datasets not pre-downloaded

### Task Details

**Default Tasks Evaluated:**
| Task | Models | Notes |
|------|--------|-------|
| **MMLU** | 57 subtasks | Expands to individual categories (mmlu_anatomy, mmlu_abstract_algebra, etc.) |
| **HellaSwag** | 1 task | Common sense reasoning |
| **ARC-Challenge** | 1 task | Science QA |
| **Winogrande** | 1 task | Coreference resolution |
| **GSM8K** | 1 task | Math reasoning |
| **IFEval** | 1 task | Instruction following |

**Total: ~63 tasks evaluated** (1 + 57 MMLU subtasks + 5 other tasks)

All evaluations use **0-shot prompting** (no examples shown to model).

You can swap in other lm-eval tasks such as `bbh`, `truthfulqa`, `gpqa`, or `agieval` depending on coverage and budget.

### Troubleshooting

**Error: "Cache directory not found"**
```bash
# Solution: Pre-download first
python download_eval_datasets.py --cache_dir ~/.cache/huggingface/datasets
```

**Slow evaluation**
```bash
# Increase batch size (if GPU memory allows)
SKIP_DOWNLOAD=1 BATCH_SIZE=32 bash scripts/eval_llm.sh <model>
```

**Out of GPU memory**
```bash
# Reduce batch size
SKIP_DOWNLOAD=1 BATCH_SIZE=8 bash scripts/eval_llm.sh <model>
```

### Results Format

Evaluation results are saved to `llm_eval_results/` as JSON files:
```
llm_eval_results/
├── Qwen3-1.7B_20260106_120000.json
├── final_checkpoint_20260106_121500.json
└── ...
```

Extract results to markdown table:
```bash
python extract_llm_results.py
```

---

## 📦 Amazon Recommendation Evaluation Results

Evaluation results on the Amazon Industrial & Scientific dataset using parallel GPU evaluation (4 GPUs).

### Qwen3-1.7B Results

**Model:** Qwen3-1.7B (final_checkpoint)
**Dataset:** Industrial_and_Scientific_5_2016-10-2018-11
**Total Samples:** 4,533
**Number of Beams:** 50

|Model| Metric | @1 | @3 | @5 | @10 | @20 | @50 |
|-----|--------|-----|-----|-----|------|------|------|
|Qwen3-1.7B-SFT| **NDCG** | 6.13% | 7.84% | 8.47% | 9.37% | 10.05% | 10.81% |
|Qwen3-1.7B-SFT| **HR (Hit Rate)** | 6.13% | 9.07% | 10.61% | 13.39% | 16.06% | 19.85% |
|Qwen3-1.7B-SFT-Mixed| **NDCG** | 6.88% | 8.36% | 9.17% | 10.00% | 10.70% | 11.63% |
|Qwen3-1.7B-SFT-Mixed| **HR (Hit Rate)** | 6.88% | 9.42% | 11.41% | 13.99% | 16.77% | 21.40% |
|Qwen3-1.7B-RL| **NDCG** | 7.17% | 8.85% | 9.42% | 10.10% | 10.67% | 10.96% |
|Qwen3-1.7B-RL| **HR (Hit Rate)** | 7.17% | 10.15% | 11.52% | 13.61% | 15.84% | 17.32% |
|Qwen3-1.7B-RL-Mixed| **NDCG** | 7.52% | 8.89% | 9.23% | 9.61% | 9.88% | 10.11% |
|Qwen3-1.7B-RL-Mixed| **HR (Hit Rate)** | 7.52% | 9.79% | 10.61% | 11.76% | 12.84% | 13.96% |

**Key Observations:**
- **RL training improves early precision**: NDCG@1 increases from 6.13% to 7.17% (+17% relative improvement)
- **Better ranking quality**: RL achieves higher NDCG across all K values, indicating improved ranking of relevant items
- **Trade-off in recall**: HR@50 decreases slightly from 19.85% to 17.32%, suggesting RL optimizes for precision over coverage
- **Consistent performance**: Both models show performance improvements as K increases, validating the beam search quality

**Evaluation Command:**
```bash
bash scripts/evaluate.sh
```

The evaluation pipeline uses parallel GPU processing (GPUs 4-7) for 4x faster evaluation compared to single-GPU baseline.

---

## 🗞️ MIND Dataset (Optional)

The MIND (Microsoft News Dataset) is a large-scale news recommendation dataset for training and evaluating news recommendation models. It contains user click histories and news articles.

### Dataset Sizes

Microsoft provides two official versions:

| Version | Train Users | Dev Users | Test Users | Total Size | Use Case |
|---------|-------------|-----------|------------|------------|----------|
| **MINDsmall** | ~50K | ~7K | ~7K | ~30-50 MB | Development, quick experimentation |
| **MINDlarge** | ~1M | ~100K | ~100K | ~1-2 GB | Production training, **leaderboard submission** |

### Prepare MIND Dataset

**Important**: Microsoft has restricted public access to the original Azure blob storage URLs. You need to download the dataset manually first.

#### Step 1: Download from Kaggle or Official Source

**Option A - Kaggle (Recommended)**:
1. Visit the [MIND News Dataset on Kaggle](https://www.kaggle.com/datasets/arashnic/mind-news-dataset)
2. Download the ZIP files you need (MINDsmall_train.zip, MINDsmall_dev.zip, etc.)

**Option B - Official MIND Website**:
Visit [https://msnews.github.io/](https://msnews.github.io/) for the official dataset download links

#### Step 2: Extract Using Helper Script

Once you have the ZIP files downloaded locally, use the preparation script to extract them:

```bash
# For development with MINDsmall
python prepare_mind.py --root ../data/MIND --size small --splits train,dev --local ../downloaded/zips

# For leaderboard with MINDlarge
python prepare_mind.py --root ../data/MIND --size large --splits train,dev,test --local ../downloaded/zips
```

The script will automatically find and extract the ZIP files to the correct directory structure.

### Training on MIND

Train your model on the MIND dataset:
```bash
# Using MINDlarge for leaderboard submission
python sft_text.py \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --mind_behaviors_path data/MIND/train/behaviors.tsv \
  --mind_news_path data/MIND/train/news.tsv \
  --output_dir output_dir/mind_large_model \
  --num_train_epochs 3
```

### Evaluate on MIND

The `eval_mind.sh` script provides a convenient way to evaluate models on MIND with automatic data extraction:

```bash
# Quick test on dev split (100 impressions)
bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev 100

# Full dev evaluation
bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev

# With abstracts for better quality
USE_ABSTRACT=1 bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev

# Generate predictions for leaderboard submission (test split)
OUTPUT_FILE=predictions.txt bash scripts/eval_mind.sh Qwen/Qwen3-1.7B test

# Custom data root
MIND_ROOT=/path/to/data bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev
```

**Auto-extraction feature**: The script automatically extracts MIND data from ZIP files if `behaviors.tsv` and `news.tsv` are not found.

**Environment variables**:
- `MIND_ROOT`: Root directory containing MIND data (default: `../data/MIND`)
- `MIND_SIZE`: Dataset size - `small` or `large` (default: `small`)
- `MIND_ZIPS`: Directory containing MIND ZIP files (default: `~/wuc/downloaded/zips`)
- `USE_ABSTRACT`: Set to 1 to use abstracts (default: 0)
- `MAX_HISTORY`: Max history items to use (default: 50)
- `SKIP_EXTRACT`: Set to 1 to skip automatic extraction (default: 0)
- `OUTPUT_FILE`: Path to save predictions for MIND leaderboard submission (optional)

**Output**: Reports AUC, MRR, nDCG@5, nDCG@10 (same metrics used on the MIND leaderboard). Optionally generates prediction file with ranked news IDs for each impression.

#### Baseline Results

**Model**: Qwen3-1.7B (zero-shot, no fine-tuning)  
**Dataset**: MINDsmall dev split (73,152 impressions)  
**Command**: `GPU_ID=2 bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev`

| Metric | Score |
|--------|-------|
| AUC | 49.28% |
| MRR | 23.46% |
| nDCG@5 | 21.15% |
| nDCG@10 | 27.61% |

**Note**: These are baseline results without any fine-tuning on MIND data. For better performance, fine-tune the model on MIND training data using the training instructions above.

### Leaderboard Submission

To submit results to the **[official MIND leaderboard](https://msnews.github.io/)**:

1. **Train on MINDlarge train split** (see above)
2. **Evaluate on MINDlarge test split** to generate predictions
3. **Submit predictions** to the leaderboard portal at https://msnews.github.io/

**Important**: The test split labels are not public. You must submit your predictions to the leaderboard server for official scoring.

**Metrics evaluated**: AUC, MRR, nDCG@5, nDCG@10

---

## 🔥 VERL RL (Optional)

You can run MiniOneRec RL using the official VERL framework while keeping the original RL code intact. This repo adds a VERL-compatible data prep, reward functions, and a launcher.

### Install VERL
```bash
pip install -r requirements-verl.txt
```

### Prepare data (CSV → parquet)
```bash
python verl_data_prep.py \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir data/verl/Industrial_and_Scientific
```

### Run VERL GRPO
```bash
python rl_verl.py \
  --model_path output_dir/sft_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/final_checkpoint \
  --train_parquet data/verl/Industrial_and_Scientific/train.parquet \
  --eval_parquet data/verl/Industrial_and_Scientific/eval.parquet \
  --output_dir output_dir/verl_rl_Industrial_and_Scientific \
  --reward_type rule \
  --sid_info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

Reward options: `rule`, `ranking`, `ranking_only`, `semantic`, `sasrec`. For `semantic`, set `--ada_path`; for `sasrec`, set `--cf_path`. The reward functions are defined in `verl_reward.py` (ranking uses `extra_info.rank` when available).

---

## 🗂️ Repository Overview

| File / Directory          | Description                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `sft.sh`                  | Shell script to start the Supervised Fine-Tuning (SFT) stage                                           |
| `sft.py`                  | Python implementation of the SFT training loop                                                            |
| `sft_ds.sh`               | Shell script for memory-optimized SFT with DeepSpeed launcher (for 8B+ models)                           |
| `sft_ds.py`               | Python implementation of SFT with DeepSpeed, Flash Attention 2, and gradient checkpointing               |
| `ds_config_zero3.json`    | DeepSpeed ZeRO-3 configuration with CPU offloading                                                       |
| `hostfile.example`        | Example hostfile for multi-node DeepSpeed training                                                       |
| `rl.sh`                   | Shell script to start the Reinforcement Learning (RL) stage                             |
| `rl.py`                   | Python implementation of the RL training loop                                              |
| `minionerec_trainer.py`   | MiniOneRec trainer — GRPO-based trainer specialized for generative recommendation                              |
| `rl_verl.py`              | VERL-based GRPO launcher (optional)                                               |
| `minionerec_verl_trainer.py` | VERL trainer wrapper (optional)                                                |
| `verl_reward.py`          | Custom reward functions for VERL (rule/semantic/sasrec/ranking)                   |
| `verl_data_prep.py`       | CSV → parquet converter for VERL data prep                                        |
| `rl_verl.sh`              | Example VERL RL launch script                                                      |
| `requirements-verl.txt`   | VERL dependency (pip install from git)                                             |
| `configs/`                | YAML configuration files                                            |
| `evaluate.sh`     | One-click offline Top-K evaluation script                                                        |
| `evaluate.py`     | Evaluation utilities for computing HR@K and NDCG@K.                                                           |
| `LogitProcessor.py`                | Logit processor for constrained decoding (Python implementation)                                         |
| `data.py`                | Data pipeline for SFT and RL training                          |
| `convert_dataset.py`                | Converts an RQ-trained dataset to the SFT-then-RL format                                            |
| `data/amazon18_data_process.sh`                |    Shell script to filter and preprocess Amazon18 data into an RQ-ready format                                      |
| `data/amazon18_data_process.py`                |   Python implementation of the Amazon18 data preprocessing pipeline                                        |
| `data/amazon23_data_process.sh`                |    Shell script to filter and preprocess Amazon23 data into an RQ-ready format                                      |
| `data/amazon23_data_process.py`                |   Python implementation of the Amazon23 data preprocessing pipeline                                        |
| `rq/text2emb/amazon_text2emb.sh`                |   Shell script to generate item embeddings (title + description) via emb_model for the Amazon dataset                                   |
| `rq/text2emb/amazon_text2emb.py`                |   Python implementation of the above embedding generation                                         |
| `rq/generate_indices.py`                |   Generates the SID file after training an RQ-VAE model                                       |
| `rq/rqvae.sh`                |   Shell script to train RQ-VAE on Amazon item embeddings                        |
| `rq/rqvae.py`                |   Python implementation of RQ-VAE training                                            |
| `rq/rqkmeans_faiss.py`                |   Python implementation of RQ-Kmeans training based on faiss                                          |
| `rq/rqkmeans_constrained.py`                |   Python implementation of Constrained RQ-Kmeans                         |
| `rq/rqkmeans_constrained.sh`                |   Shell script to train constrained RQ-Kmeans constrained on Amazon item embeddings                        |
| `rq/rqkmeans_plus.py`                |   Python implementation of RQ-Kmeans+                        |
| `rq/rqkmeans_plus.sh`                |   Shell script to train RQ-Kmeans+ constrained on Amazon item embeddings                        |
| `rq/generate_indices_plus.py`                |   Generates the SID file after training an RQ-Kmeans+ model                                       |
| `rq/generate_indices_plus.sh`                |   Shell script to generate the SID file after training an RQ-Kmeans+ model                                       |
| `prepare_mind.py`        | Prepare (extract + organize) the MIND dataset from downloaded ZIPs                                       |
| `eval_mind.sh`           | MIND evaluation script with automatic data extraction from ZIPs                                           |
| `evaluate_mind.py`       | Core MIND evaluation implementation (AUC, MRR, nDCG@5, nDCG@10)                                           |
| `llm_eval.py`            | Single-GPU LLM capability evaluation (MMLU, HellaSwag, etc.)                                              |
| `llm_eval_parallel.py`   | Multi-GPU parallel LLM evaluation                                                                         |
| `eval_llm.sh`            | LLM evaluation script with decoupled dataset download                                                     |
| `download_eval_datasets.py` | Pre-download evaluation datasets for lm-eval-harness                                                  |
| `requirements.txt`        | List of Python dependencies                                                                                |
| `requirements-eval.txt`   | Optional dependencies for LLM capability evaluation                                                        |

---

## 🚀 Quickstart

Use the pre-trained Industrial/Office SIDs we provide for a quick start!
Reproduction can be achieved with just 4–8 A100/H100 GPUs.

### 1. Create an isolated Python environment

```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

For 8B+ models, also install Flash Attention 2:
```bash
pip install flash-attn --no-build-isolation
```

### 3. SFT

For models ≤ 4B parameters:
```bash
bash scripts/sft.sh
```

For models ≥ 8B parameters (memory-optimized with DeepSpeed):
```bash
bash scripts/sft_ds.sh
```

### 4. Recommendation-Oriented RL

```bash
bash scripts/rl.sh
```

### 5. Run the evaluation bash

```bash
bash scripts/evaluate.sh
```

---

## 🧩 Mixed SFT: Preventing Catastrophic Forgetting

When fine-tuning LLMs on recommendation-specific data, the model often suffers from **catastrophic forgetting** of general capabilities (MMLU, HellaSwag, GSM8K, etc.). To address this, use Mixed SFT to combine recommendation data with general instruction data.

### Problem

As shown in the evaluation results above, recommendation-only SFT can cause catastrophic forgetting:
- GSM8K drops to **0.00%** (from 42% baseline)
- MMLU drops to **26.11%** (from 55% baseline)

### Solution: Mixed SFT

The `sft_mixed.py` script combines:
1. **Recommendation data** (Amazon, SIDs, etc.) - for task-specific learning
2. **General instruction data** (UltraChat, Alpaca, etc.) - to maintain general capabilities

### Quick Start

**Step 1: Download General Data**
```bash
python download_ultrachat.py \
  --output data/general/ultrachat_200k.jsonl \
  --limit 50000
```

**Step 2: Run Mixed SFT**
```bash
bash scripts/sft_mixed.sh
```

By default, this uses:
- **70% recommendation data** (Amazon + SIDs + Fusion tasks)
- **30% general data** (UltraChat)

**Step 3: Evaluate Both Capabilities**

General capabilities:
```bash
bash scripts/eval_llm.sh output_dir/sft_mixed_*/final_checkpoint \
  mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval \
  llm_eval
```

Recommendation performance:
```bash
bash scripts/evaluate.sh output_dir/sft_mixed_*/final_checkpoint
```

### Configuration

**Data Mixing Ratio:**
```bash
# More general data (50/50 mix)
GENERAL_DATA_RATIO=0.5 bash scripts/sft_mixed.sh

# Less general data (10% general, 90% rec)
GENERAL_DATA_RATIO=0.1 bash scripts/sft_mixed.sh
```

**General Data Source:**
```bash
# UltraChat (default)
GENERAL_DATA_PATH=data/general/ultrachat_200k.jsonl bash scripts/sft_mixed.sh

# Your custom JSONL file
GENERAL_DATA_PATH=/path/to/custom_data.jsonl bash scripts/sft_mixed.sh
```

**General Data Amount:**
```bash
# Use 100K examples
GENERAL_DATA_SAMPLE=100000 bash scripts/sft_mixed.sh

# Use all available examples
GENERAL_DATA_SAMPLE=-1 bash scripts/sft_mixed.sh
```

### Expected Results

With mixed SFT, you should see:

✅ **Maintained general capabilities**
- GSM8K: >30% (instead of 0%)
- MMLU: >40% (instead of 26%)
- IFEval: >30% (instead of 10%)

✅ **Preserved recommendation performance**
- HR@10, NDCG@10 should be similar to recommendation-only SFT

### Comparison

| Approach | Rec Performance | General Performance | Training Time |
|----------|----------------|---------------------|---------------|
| Rec-only SFT (`sft.sh`) | ⭐⭐⭐⭐⭐ | ⭐ | Fast |
| Mixed SFT (`sft_mixed.sh`) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium (+30%) |
| Two-stage (General → Rec) | ⭐⭐⭐⭐ | ⭐⭐⭐ | Slow (2× training) |

**Recommendation**: Use mixed SFT for production models that need both capabilities.

---

## 📰 MIND Dataset (Optional)

The MIND dataset contains `news.tsv` and `behaviors.tsv`. This repo includes dataset helpers to build text-only SFT examples from MIND.

Classes:
- `MINDTextSFTDataset` (training)
- `EvalMINDTextDataset` (evaluation)

Basic usage example:
```python
from transformers import AutoTokenizer
from data import MINDTextSFTDataset

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
dataset = MINDTextSFTDataset(
    behaviors_path="path/to/behaviors.tsv",
    news_path="path/to/news.tsv",
    tokenizer=tokenizer,
    max_history=50,
    use_abstract=False,
)
```

### MIND leaderboard-style evaluation
This script scores each impression and reports AUC, MRR, nDCG@5, nDCG@10 (same metrics used on the MIND leaderboard).

```bash
bash evaluate_mind.sh /path/to/your_model /path/to/MIND dev
```

---

## 🔀 Mixed SFT (Amazon + MIND + General)

You can mix Amazon, MIND, and general instruction data in one SFT run using `sft_text_mixed.py`. Ratios are normalized automatically.

```bash
python sft_text_mixed.py \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --output_dir output_dir/sft_text_mixed \
  --amazon_train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --amazon_eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --amazon_category Industrial_and_Scientific \
  --amazon_item_meta_path data/Amazon/index/Industrial_and_Scientific.item.json \
  --mind_behaviors_path path/to/MIND/behaviors.tsv \
  --mind_news_path path/to/MIND/news.tsv \
  --general_jsonl data/general/ultrachat_200k.jsonl \
  --amazon_ratio 0.7 \
  --mind_ratio 0.2 \
  --general_ratio 0.1 \
  --eval_source amazon
```

---

## 🔧 Weights & Biases (wandb) Setup

You have **three options** to authenticate with wandb:

### Option 1: Environment Variable (Recommended for Automated Training)

Set the `WANDB_API_KEY` environment variable before running training:

```bash
# Get your API key from: https://wandb.ai/authorize
export WANDB_API_KEY=your_api_key_here

# Then run training
bash scripts/sft.sh
```

**Or add it to your shell profile** (`~/.bashrc` or `~/.zshrc`):
```bash
echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

**Or set it in your training script** (`sft.sh`):
```bash
export WANDB_API_KEY=your_api_key_here
export NCCL_IB_DISABLE=1
PROCESS_NUM=4
# ... rest of script
```

### Option 2: Manual Login (One-time Setup)

Run once to authenticate:
```bash
wandb login
```

This will prompt you to enter your API key. After this, wandb will remember your credentials in `~/.netrc` or `~/.config/wandb/settings`.

### Option 3: Config File

Create/edit `~/.netrc`:
```
machine api.wandb.ai
login user
password your_api_key_here
```

Or create `~/.config/wandb/settings`:
```ini
[default]
api_key = your_api_key_here
```

### Disable wandb (If Not Needed)

If you don't want to use wandb, you can disable it:

```bash
export WANDB_MODE=disabled
bash scripts/sft.sh
```

**Troubleshooting:**
- **"wandb: ERROR Not logged in"**: Set `WANDB_API_KEY` environment variable
- **"wandb: ERROR Network error"**: Use offline mode with `export WANDB_MODE=offline`
- **Want to run without wandb**: Disable it with `export WANDB_MODE=disabled`

---

## 📱 Feeds Recommendation Data Adaptation Guide

This guide explains how to adapt your Feeds recommendation data to work with MiniOneRec. MiniOneRec treats all items generically, so **feeds/articles/posts can be treated as "items"** just like Amazon products.

### Required Data Format

**1. CSV Training Files (train/valid/test)**

Required columns:
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
```

**Mapping for Feeds:**
- `user_id` → User ID (e.g., "user_123")
- `history_item_title` → List of feed titles the user interacted with
- `item_title` → Target feed title to predict
- `history_item_id` → List of feed IDs (numeric)
- `item_id` → Target feed ID (numeric)
- `history_item_sid` → List of feed SIDs
- `item_sid` → Target feed SID

**Example:**
```csv
user_id,history_item_title,item_title,history_item_id,item_id,history_item_sid,item_sid
user_123,"['AI Breakthrough News', 'Tech Industry Update']","New ML Research Published",[456,789],202,"['<a_15><b_23><c_41>', '<a_8><b_12><c_67>']","<a_20><b_5><c_33>"
```

**2. Item Metadata File (`feeds.item.json`)**

```json
{
  "0": {
    "title": "Breaking: New AI Model Released",
    "description": "A new breakthrough in AI technology...",
    "brand": "TechNews",
    "categories": "Technology, AI, Machine Learning"
  }
}
```

**3. SID Index File (`feeds.index.json`)**

```json
{
  "0": ["<a_15>", "<b_23>", "<c_41>"],
  "1": ["<a_8>", "<b_12>", "<c_67>"]
}
```

### Feeds → Amazon Format Mapping

| Amazon Concept | Feeds Equivalent |
|----------------|------------------|
| Product | Feed/Article/Post |
| Purchase/Review | Click/Read/Like/Share/View |
| Product Title | Feed Title/Headline |
| Product Description | Feed Content/Summary |
| Product Category | Feed Category/Topic |
| Brand | Publisher/Source |

### Adaptation Process

1. **Prepare Sequential Interaction Data** - chronologically ordered user interactions
2. **Generate SIDs for Feeds** - using RQ-VAE or RQ-Kmeans on feed text embeddings
3. **Create Item Metadata File** - title, description, categories, publisher
4. **Format CSV Files** - with all 7 required columns
5. **Organize Directory Structure** - train/valid/test splits

### Code Changes Needed

**Update `sft.sh`:**
```bash
for category in "Feeds"; do
    train_file=$(ls -f ./data/Feeds/train/${category}*.csv)
    eval_file=$(ls -f ./data/Feeds/valid/${category}*.csv)
    ...
    --category ${category} \
    --sid_index_path ./data/Feeds/index/Feeds.index.json \
    --item_meta_path ./data/Feeds/index/Feeds.item.json
done
```

**Update `sft.py` (line ~137):**
```python
category_dict = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Feeds": "news feeds and articles",  # Add this
    ...
}
```

For complete details on data format, SID generation, and validation, see the comments in the code and example data files.

---

## 📜 Full Pipeline Walk-through

### 0. Prerequisites
- GPUs: <e.g., 4–8 × A100/H100 80 GB or comparable>
- Python: 3.11

### 1. Environment Setup
- **1.1 Clone the repo**
```
git clone https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```
- **1.2 Create and activate a conda env**
```
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```
- **1.3 Install dependencies**
```
pip install -r requirements.txt
```

### 2. Data Preparation

- **2.1 Download the raw dataset (Optional)**  
  Get it from the official page:
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 
  [Amazon Reviews 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/), 
  [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
  Note: The Industrial and Office datasets are included in Amazon 2018; the Amazon 2014 and 2023 versions require slight modifications to our data/amazon18_data_process.py.
- **2.2 Filter and preprocess**
```
bash data/amazon18_data_process.sh \
     --dataset  your_dataset_type \ # e.g. Industrial
     --user_k 5 \
     --item_k 5 \
     --st_year 2017 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 11 \
     --output_path ./data/Amazon18
```
- **2.3 Encode item text to embeddings**
```
bash rq/amazon_text2emb.sh \
     --dataset your_dataset_type \ # e.g., Industrial 
     --root your_processed_dataset_path \
     --plm_name qwen \
     --plm_checkpoint your_emb_model_path
```

### 3. SID Construction

Choose either 3.1.1, 3.1.2, 3.1.3 or 3.1.4.

- **3.1.1 Train RQ-VAE on the embeddings**
```
bash rq/rqvae.sh \
      --data_path xxx/data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
```

- **3.1.2 Train RQ-Kmeans on the embeddings**

```
conda install faiss-gpu
python rqkmeans_faiss.py --dataset Industrial_and_Scientific # The RQ-Kmeans method based on semantic embeddings has a relatively high collision rate.
```

- **3.1.3 Train constrained RQ-Kmeans on the embeddings**
For conflicting items, we add an extra layer to perform deduplication; meanwhile, we use a balanced constraint to ensure that the SIDs are evenly distributed.
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
```

- **3.1.4 Train RQ-Kmeans+ on the embeddings**
```
pip install k_means_constrained
pip install polars
bash rqkmeans_constrained.sh
bash rqkmeans_plus.sh
```

- **3.2 Generate indices(only RQ-VAE & RQ-Kmeans+ needed)**
```
python rq/generate_indices.py
# or
bash rq/generate_indices_plus.sh
```

- **3.3 Convert dataset format**
```
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/ourput_dir \

```

### 4. SFT

#### 4.1 Standard SFT (for models ≤ 4B parameters)

```bash
bash scripts/sft.sh \
     --base_model your_model_path \
     --output_dir your_ourput_dir \
     --sid_index_path your_.index.json_path \
     --item_meta_path your_.item.json_path
```

#### 4.2 Memory-Optimized SFT with DeepSpeed (for models ≥ 8B parameters)

For large models that cause OOM errors, use the DeepSpeed-optimized training script with the DeepSpeed launcher.

**Single Node (8 GPUs):**
```bash
bash scripts/sft_ds.sh
```
The script will automatically create a default hostfile for single-node training.

**Multi-Node Training:**

1. Create a hostfile (e.g., `hostfile`) with your node configuration:
```bash
# For 2 nodes with 8 GPUs each:
node-0 slots=8
node-1 slots=8
```

2. Run the training script:
```bash
HOSTFILE=./hostfile bash scripts/sft_ds.sh
```

Or use IP addresses:
```bash
# hostfile content:
192.168.1.10 slots=8
192.168.1.11 slots=8
```

**Example hostfile configurations:**

See [hostfile.example](hostfile.example) for more examples.

**Requirements for Multi-Node:**
- Passwordless SSH set up between all nodes
- Same codebase and data accessible from all nodes
- Run from the master node (first node in hostfile)

**Alternative: Manual Multi-Node Training (2 nodes, 8 GPUs each):**

On Master Node (rank 0):
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

torchrun \
    --nproc_per_node 8 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    sft_ds.py \
    --base_model Qwen/Qwen3-8B \
    --batch_size 16 \
    --micro_batch_size 1 \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_11.csv \
    --output_dir output_dir/sft_Industrial_and_Scientific_qwen3-8b \
    --wandb_project MiniOneRec \
    --wandb_run_name sft_Industrial_and_Scientific_qwen3-8b \
    --category Industrial_and_Scientific \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --deepspeed_config ds_config_zero3.json
```

On Worker Node (rank 1):
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

torchrun \
    --nproc_per_node 8 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    sft_ds.py \
    [same arguments as master node...]
```

**Alternative: Using torchrun (not recommended, use DeepSpeed launcher instead)**

If you prefer torchrun over DeepSpeed launcher, you can manually run on each node:

On Master Node (rank 0):
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

torchrun \
    --nproc_per_node 8 \
    --nnodes 2 \
    --node_rank 0 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    sft_ds.py \
    [same training arguments as above...]
```

On Worker Nodes (rank 1, 2, ...):
```bash
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500

torchrun \
    --nproc_per_node 8 \
    --nnodes 2 \
    --node_rank 1 \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    sft_ds.py \
    [same training arguments as above...]
```

**Memory Optimization Features:**
- ✅ Flash Attention 2 for efficient attention computation
- ✅ Gradient checkpointing to reduce memory usage
- ✅ DeepSpeed ZeRO-3 with CPU offloading for optimizer and parameters
- ✅ Optimized batch sizes (`micro_batch_size=1` for 8B models)

**Installing Flash Attention 2:**
```bash
pip install flash-attn --no-build-isolation
```

**Troubleshooting OOM:**
If you still encounter OOM errors:
1. Reduce `batch_size` to 8
2. Reduce `cutoff_len` to 256 or 384
3. Enable NVMe offload in DeepSpeed config (for very large models)

### 5. Recommendation-Oriented RL
> (Optional) For production-scale datasets, considering the cost of reinforcement learning and diminishing marginal returns, you can perform the RL stage using only a relatively small subset on the order of tens of thousands of samples.
```
bash scripts/rl.sh \
     --model_path your_model_path \
     --output_dir output_dir \
```

### 6. Offline Evaluation

```
bash scripts/evaluate.sh \
     --exp_name your_model_path
```

### 7. Text-based SFT Evaluation with Similarity Matching

For text-based SFT models (trained with [sft_text.py](src/sft_text.py)), the evaluation now supports **similarity-based matching** instead of exact string matching. This is more robust for recommendation tasks where predicted item names may have minor variations from ground truth.

#### Evaluation Pipeline

The text-based evaluation consists of two stages:

1. **Generation** ([evaluate_text.py](src/evaluate_text.py)): Generate predictions using beam search
2. **Metrics Calculation** ([calc_text_similarity.py](src/calc_text_similarity.py)): Compute NDCG and HR with similarity matching

#### Quick Start

Run the complete evaluation pipeline:

```bash
bash scripts/evaluate_text.sh
```

By default, this uses **similarity-based matching** with threshold `0.85`.

#### Customization Options

**Use exact matching (original behavior):**
```bash
USE_SIMILARITY=false bash scripts/evaluate_text.sh
```

**Adjust similarity threshold (0.0 to 1.0):**
```bash
# More lenient matching (threshold = 0.75)
SIMILARITY_THRESHOLD=0.75 bash scripts/evaluate_text.sh

# Stricter matching (threshold = 0.90)
SIMILARITY_THRESHOLD=0.90 bash scripts/evaluate_text.sh
```

**Combine both options:**
```bash
USE_SIMILARITY=true SIMILARITY_THRESHOLD=0.80 bash scripts/evaluate_text.sh
```

#### Manual Evaluation (Advanced)

**Step 1: Generate predictions**
```bash
python src/evaluate_text.py \
    --base_model output_dir/sft_text_Industrial_and_Scientific/final_checkpoint \
    --category Industrial_and_Scientific \
    --test_data_path data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --item_meta_path data/Amazon/index/Industrial_and_Scientific.item.json \
    --result_json_data results_text/predictions.json \
    --batch_size 4 \
    --num_beams 20 \
    --max_new_tokens 256
```

**Step 2: Calculate metrics with similarity matching**
```bash
python src/calc_text_similarity.py \
    --path results_text/predictions.json \
    --item_path data/Amazon/info/Industrial_and_Scientific.txt \
    --similarity_threshold 0.85 \
    --use_similarity true
```

#### Understanding Similarity Matching

The similarity matching uses Python's `difflib.SequenceMatcher` to compute fuzzy string similarity:

- **Exact match first**: Always tries exact string match (fastest)
- **Fuzzy fallback**: If no exact match, checks similarity ratio
- **Threshold**: Items with similarity ≥ threshold are considered matches
- **Best rank**: If multiple fuzzy matches, uses the highest-ranked prediction

**Example:**
```
Ground truth: "Black Ballpoint Pen"
Predictions:
  1. "Blue Pen"               → similarity: 0.50 (no match)
  2. "Black Ballpoint Pens"   → similarity: 0.95 (MATCH! ✓)
  3. "Black Ballpoint Pen"    → exact match (MATCH! ✓)
```

With threshold `0.85`, prediction #2 would be considered a hit at rank 2.

#### Output Format

The evaluation prints:

1. **Match Statistics**:
   - Exact matches: Perfect string matches
   - Fuzzy matches: Similarity-based matches
   - No matches: Failed to find target in predictions

2. **Metrics Table**:
   - NDCG@K: Position-aware ranking quality
   - HR@K: Hit rate (binary hit/miss)
   - K values: 1, 3, 5, 10, 20, 50

**Example output:**
```
============================================================
Evaluation Mode: SIMILARITY-BASED
Similarity Threshold: 0.85
============================================================

Number of beams: 20
Valid top-k values: [1, 3, 5, 10, 20]

Match Statistics:
  Exact matches: 850/1000 (85.00%)
  Fuzzy matches: 120/1000 (12.00%)
  No matches: 30/1000 (3.00%)

Metrics:
NDCG:	['0.3245', '0.4512', '0.4876', '0.5234', '0.5456']
HR:	['0.3245', '0.6123', '0.7234', '0.8345', '0.9012']

============================================================
Metric     @1         @3         @5         @10        @20        @50
============================================================
NDCG       32.45%     45.12%     48.76%     52.34%     54.56%       N/A
HR         32.45%     61.23%     72.34%     83.45%     90.12%       N/A
============================================================
```

#### Comparison: Exact vs Similarity Matching

| Approach | Precision | Robustness | Use Case |
|----------|-----------|------------|----------|
| **Exact Matching** | Strictest | Low | When item names are standardized |
| **Similarity (0.90+)** | Very High | Medium | Minor variations (plurals, punctuation) |
| **Similarity (0.85)** | High | High | Recommended default |
| **Similarity (0.75-0.80)** | Medium | Very High | Noisy or abbreviated item names |

**Recommendation**: Start with `0.85` threshold. If you see many "No matches" but visually similar predictions, lower to `0.80`. If you need stricter evaluation, raise to `0.90` or use exact matching.

---

## 📝 Upcoming Features

We are actively extending MiniOneRec’s capabilities. The following enhancements are already on our roadmap:
* ⏱️ **More SID Construction Algorithms**: forthcoming support for R-VQ, RQ-Kmeans, RQ-OPQ, and RQ-VAE-v2 (PLUM).
* ⚙️ **MiniOneRec-Think**: a module that seamlessly integrates dialogue, reasoning, and personalized recommendation, providing an all-in-one solution for complex interactive scenarios.
* 🔍 **Broader Dataset Support**: additional popular public datasets, including Yelp, to further validate the generality of our algorithms.

---

## 🏫 Institutions  <!-- omit in toc -->

This project is developed by the following institutions:

- <img src="assets/lds.png" width="28px"> [LDS](https://data-science.ustc.edu.cn/_upload/tpl/15/04/5380/template5380/index.html)
- <img src="assets/alphalab.jpg" width="28px"> [AlphaLab](https://alphalab-ustc.github.io/index.html)
- <img src="assets/next.jpg" width="28px"> [NExT](https://www.nextcenter.org/)
 
---

## 🧩 Contributing

We welcome and appreciate all contributions! If you have ideas to improve MiniOneRec, please feel free to submit a pull request (PR).

---
## 🙏 Acknowledgements

This repository reuses or adapts portions of code from the following open-source projects. We gratefully acknowledge their authors and contributors:

- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)

---

## 🔖 Citation <!-- omit in toc -->

If you find our code/paper/model helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@misc{MiniOneRec,
      title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation}, 
      author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
      year={2025},
      eprint={2510.24431},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
}

@article{ReRe,
      title={Reinforced Preference Optimization for Recommendation}, 
      author={Junfei Tan and Yuxin Chen and An Zhang and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Xiang Wang},
      journal={arXiv preprint arXiv:2510.12211},
      year={2025},
}

@inproceedings{RecZero,
      title={Think before Recommendation: Autonomous Reasoning-enhanced Recommender}, 
      author={Xiaoyu Kong and Junguang Jiang and Bin Liu and Ziru Xu and Han Zhu and Jian Xu and Bo Zheng and Jiancan Wu and Xiang Wang},
      year={2025},
      booktitle={NeurIPS},
}

```

---

<div align="center">
We welcome contributions from the community! 🤝
</div>

#### Troubleshooting

**"No matches" percentage is high (>20%)**

Solution: Lower similarity threshold
```bash
SIMILARITY_THRESHOLD=0.80 bash scripts/evaluate_text.sh
```

**Metrics seem too optimistic**

Solution: Increase threshold or use exact matching
```bash
SIMILARITY_THRESHOLD=0.90 bash scripts/evaluate_text.sh
# or
USE_SIMILARITY=false bash scripts/evaluate_text.sh
```

**Need to compare both approaches**

Solution: Run both and compare results
```bash
# Similarity-based
bash scripts/evaluate_text.sh > results_similarity.txt

# Exact matching
USE_SIMILARITY=false bash scripts/evaluate_text.sh > results_exact.txt

# Compare
diff results_similarity.txt results_exact.txt
```

#### Advanced Usage

**Multiple Thresholds Comparison:**
```bash
for threshold in 0.75 0.80 0.85 0.90 0.95; do
    echo "=== Threshold: $threshold ==="
    SIMILARITY_THRESHOLD=$threshold bash scripts/evaluate_text.sh
done
```

**Batch Multiple Categories:**
```bash
for category in "Industrial_and_Scientific" "Office_Products" "Toys_and_Games"; do
    echo "Processing: $category"
    # Update category in scripts/evaluate_text.sh
    bash scripts/evaluate_text.sh
done
```

**Compare All Methods:**
```bash
bash scripts/compare_matching_methods.sh results_text/predictions.json data/Amazon/info/Industrial_and_Scientific.txt
```

#### Technical Details

**String Normalization:**
- Convert to lowercase
- Strip leading/trailing whitespace

**Similarity Calculation:**

Uses Python's `difflib.SequenceMatcher` with Ratcliff/Obershelp algorithm:
```python
from difflib import SequenceMatcher

def fuzzy_match(str1, str2, threshold=0.85):
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    ratio = SequenceMatcher(None, s1, s2).ratio()
    return ratio >= threshold
```

Formula: `ratio = 2 * M / T`
- `M` = number of matching characters
- `T` = total number of characters in both strings


---

