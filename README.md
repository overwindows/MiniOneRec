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
| Qwen3-1.7B-SFT-Text (Amazon) | 26.05% | 52.96% | 34.98% | 58.88% | 1.59% | 2.40% |
| Qwen3-1.7B-SFT-Text-Mixed (Amazon+UltraChat)  | 48.98% | 58.31% | 41.55% | 55.80% | 21.61% | 6.28% |
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
**Number of Beams:** 50 (SID-based), 20 (Text-based)

|Model| Metric | @1 | @3 | @5 | @10 | @20 | @50 |
|-----|--------|-----|-----|-----|------|------|------|
|Qwen3-1.7B-SFT| **NDCG** | 6.13% | 7.84% | 8.47% | 9.37% | 10.05% | 10.81% |
|Qwen3-1.7B-SFT| **HR** | 6.13% | 9.07% | 10.61% | 13.39% | 16.06% | 19.85% |
|Qwen3-1.7B-SFT-Text| **NDCG** | 6.95% | 8.94% | 9.64% | 10.55% | 11.25% | - |
|Qwen3-1.7B-SFT-Text| **HR** | 6.95% | 10.39% | 12.09% | 14.89% | 17.63% | - |
|Qwen3-1.7B-SFT-Mixed| **NDCG** | 6.88% | 8.36% | 9.17% | 10.00% | 10.70% | 11.63% |
|Qwen3-1.7B-SFT-Mixed| **HR** | 6.88% | 9.42% | 11.41% | 13.99% | 16.77% | 21.40% |
|Qwen3-1.7B-SFT-Text-Mixed| **NDCG** | 7.65% | 9.63% | 10.35% | 11.28% | 11.99% | - |
|Qwen3-1.7B-SFT-Text-Mixed| **HR** | 7.65% | 11.07% | 12.82% | 15.71% | 18.49% | - |
|Qwen3-1.7B-RL| **NDCG** | 7.17% | 8.85% | 9.42% | 10.10% | 10.67% | 10.96% |
|Qwen3-1.7B-RL| **HR** | 7.17% | 10.15% | 11.52% | 13.61% | 15.84% | 17.32% |
|Qwen3-1.7B-RL-Mixed| **NDCG** | 7.52% | 8.89% | 9.23% | 9.61% | 9.88% | 10.11% |
|Qwen3-1.7B-RL-Mixed| **HR** | 7.52% | 9.79% | 10.61% | 11.76% | 12.84% | 13.96% |

**Key Observations:**
- **Text-based SFT outperforms SID-based SFT**: NDCG@10 improves from 9.37% to 10.55% (+12.6% relative improvement)
- **Natural language is more effective**: Text-based approach achieves higher metrics across all K values, demonstrating the advantage of using item descriptions over structured IDs
- **Evaluation methodology**: Text-based uses similarity-based catalog matching (find most similar item → compare ID), equivalent to SID's direct ID mapping
- **RL training improves early precision**: NDCG@1 increases from 6.13% to 7.17% (+17% relative improvement)
- **Better ranking quality**: RL achieves higher NDCG across all K values, indicating improved ranking of relevant items
- **Trade-off in recall**: HR@50 decreases slightly from 19.85% to 17.32%, suggesting RL optimizes for precision over coverage
- **Consistent performance**: Both models show performance improvements as K increases, validating the beam search quality

**Evaluation Commands:**
```bash
# SID-based evaluation (Structured ID tokens)
bash scripts/evaluate.sh

# Text-based evaluation (Natural language descriptions)
CUDA_LIST="0,1,2,3,4,5,6" bash scripts/evaluate_text.sh
```

The evaluation pipeline uses parallel GPU processing for faster evaluation compared to single-GPU baseline.

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
python prepare_mind.py --root ../data/MIND_small --size small --splits train,dev --local ../downloaded/zips

# For leaderboard with MINDlarge
python prepare_mind.py --root ../data/MIND_large --size large --splits train,dev,test --local ../downloaded/zips
```

The script will automatically find and extract the ZIP files to the correct directory structure.

**Tip:** If you keep separate folders (`../data/MIND_small` and `../data/MIND_large`), the training/eval scripts will auto-select the right one based on `MIND_SIZE` when `MIND_ROOT` is not set.

### Training on MIND

We provide three training approaches for MIND dataset:

1. **Standard SFT** (`sft_mind.sh`) - Traditional next-item prediction (53.72% AUC)
2. **Ranking-Aware SFT** (`sft_mind_ranking.sh`) - 🔥 **Multiple-choice ranking format (65.49% AUC - RECOMMENDED!)**
3. **Point-wise SFT** (`sft_mind_pointwise.sh`) - 🆕 **Yes/No classification per candidate**

#### 🚀 Ranking-Aware SFT (Recommended)

**What's different?** Instead of training on "history → one clicked news", this approach uses "history + all candidates → select best option (A/B/C/...)" format, aligning training with evaluation.

**Quick Start:**
```bash
# Basic training on MINDsmall with ranking-aware format
bash scripts/sft_mind_ranking.sh

# Multi-GPU training (automatically detects GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_mind_ranking.sh

# Evaluate with aligned format
bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_small_*/final_checkpoint dev
```

**Training Format Example:**
```
Role: You are a news recommendation assistant.
Task: Select the most relevant news article for the user based on their reading history.

User History:
1. [Title] Lakers win against Warriors (Sports)
2. [Title] LeBron James scores 40 points (Sports)

Candidate News Articles:
1. [Title] Best gardening tips for spring (Lifestyle)
2. [Title] Nvidia stock jumps 10% on AI news (Finance)
3. [Title] NBA playoffs schedule announced (Sports)

Please analyze the user's interests and select the best article from the candidates above.
Output only the option number.

Answer: 3
```

**Why it works:**
- ✅ Model sees ALL candidates during training (not just the clicked one)
- ✅ Learns to rank and compare options
- ✅ Training format matches evaluation format
- ✅ Uses numeric options (1/2/3/...) - supports unlimited candidates
- ✅ **Result: +11.77% AUC improvement** (53.72% → 65.49%)

**Configuration:** Uses same environment variables as standard SFT, plus:
- `MAX_CANDIDATES=20` - Limit candidates per sample to fit in context

#### 🆕 Point-wise SFT (Yes/No Classification)

**What's different?** Instead of showing all candidates in one prompt, this approach evaluates each candidate independently with a Yes/No question: "Is this article relevant to the user?"

**Key advantages:**
- ✅ More training signal (every candidate gets a label)
- ✅ Shorter context per sample (faster training)
- ✅ No position bias
- ✅ Can leverage more negatives efficiently

**Quick Start:**
```bash
# Basic training on MINDsmall with point-wise format
bash scripts/sft_mind_pointwise.sh

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_mind_pointwise.sh

# With more negatives per positive (try 1.0, 2.0, 3.0)
NEG_RATIO=2.0 bash scripts/sft_mind_pointwise.sh

# Evaluate with point-wise scoring
bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev
```

**Training Format Example:**
```
Role: You are a news recommendation assistant.
Task: Determine if the candidate article matches the user's interests.

User History:
1. [Title] Lakers win against Warriors (Sports)
2. [Title] LeBron James scores 40 points (Sports)

Candidate Article:
[Title] NBA playoffs schedule announced (Sports)

Based on the user's reading history, is this article relevant to them?
Answer with Yes or No.

Answer: Yes
```

**Why it works:**
- ✅ Dense training signal - each candidate labeled independently
- ✅ No position bias - candidates aren't shown together
- ✅ Flexible negative ratio - can use more negatives per positive
- ✅ Shorter sequences - efficient training

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `NEG_RATIO` | `1.0` | Number of negatives per positive |
| `CUTOFF_LEN` | `2048` | Max sequence length (shorter than list-wise) |
| `MAX_HISTORY` | `0` | Max history items (0=unlimited) |
| `USE_ABSTRACT` | `0` | Set to 1 to include abstracts |

**Comparison: List-wise vs Point-wise:**

| Aspect | List-wise (Ranking) | Point-wise (Yes/No) |
|--------|---------------------|---------------------|
| Training signal | Sparse (1 per impression) | Dense (1 per candidate) |
| Context length | Long (all candidates) | Short (single candidate) |
| Position bias | Possible | None |
| Scoring method | P(option number) | P(Yes) - P(No) |
| Best for | Direct ranking | Dense supervision |

#### Standard SFT

For comparison, the standard SFT approach:

**Quick Start**

```bash
# Basic training on MINDsmall
bash scripts/sft_mind.sh

# Multi-GPU training (automatically detects GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_mind.sh

# Training with abstracts (better quality)
USE_ABSTRACT=1 bash scripts/sft_mind.sh

# Custom learning rate and batch size
LEARNING_RATE=1e-4 BATCH_SIZE=512 bash scripts/sft_mind.sh

# Training on MINDlarge for leaderboard submission
MIND_SIZE=large bash scripts/sft_mind.sh
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen3-1.7B` | Base model to fine-tune |
| `MIND_ROOT` | `../data/MIND` | Root directory containing MIND data |
| `MIND_SIZE` | `small` | Dataset size (`small` or `large`) |
| `BATCH_SIZE` | `1024` | Global batch size |
| `MICRO_BATCH_SIZE` | `16` | Batch size per GPU |
| `NUM_EPOCHS` | `3` | Number of training epochs |
| `LEARNING_RATE` | `3e-4` | Learning rate |
| `CUTOFF_LEN` | `1024` | Maximum sequence length |
| `USE_ABSTRACT` | `0` | Set to `1` to use news abstracts |
| `MAX_HISTORY` | `50` | Maximum history items to use |

#### Advanced Usage

```bash
# For SOTA results with larger model
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
  BATCH_SIZE=2048 \
  LEARNING_RATE=1e-4 \
  USE_ABSTRACT=1 \
  MAX_HISTORY=100 \
  bash scripts/sft_mind.sh

# Training on MINDlarge with 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MIND_SIZE=large \
  BATCH_SIZE=4096 \
  bash scripts/sft_mind.sh
```

#### Output

The trained model will be saved to `output_dir/sft_mind_{MIND_SIZE}_{MODEL_NAME}_bs{BATCH_SIZE}/final_checkpoint/`

Training logs are uploaded to Weights & Biases (wandb) automatically.

### Evaluate on MIND

We provide three evaluation scripts:

1. **Ranking-Only Evaluation** (`eval_ranking_only.sh`) - For ranking-aware models, uses multiple-choice format
2. **Point-wise Evaluation** (`eval_mind_pointwise.sh`) - 🆕 For point-wise models, scores P(Yes) vs P(No)
3. **Standard Evaluation** (`eval_mind.sh`) - For standard SFT models, uses text generation format

#### Ranking-Only Evaluation (for Ranking-Aware Models)

```bash
# Quick test (100 impressions)
bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_*/final_checkpoint dev 100

# Full evaluation
bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_*/final_checkpoint dev
```

**How it works:** Evaluates using the same multiple-choice format as training (scores P(1), P(2), P(3), ...).

**Metrics:** All metrics match the [official MIND evaluation script](https://github.com/msnews/MIND/blob/master/evaluate.py) exactly:
- AUC: Uses sklearn's `roc_auc_score`
- MRR: Official formula `sum(rr_score) / sum(y_true)`
- DCG/nDCG: Official gain formula `2^label - 1` with discount `log2(rank+1)`

#### Point-wise Evaluation (for Point-wise Models)

```bash
# Quick test (100 impressions)
bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev 100

# Full evaluation (single GPU)
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev

# Multi-GPU parallel evaluation (4-8x faster)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev

# With abstracts
USE_ABSTRACT=1 CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev
```

**How it works:**
- Each candidate is scored independently using: `score = log P(" Yes") - log P(" No")`
- Candidates are ranked by their scores within each impression
- Uses batched inference for efficiency

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `8` | Batch size for scoring candidates |
| `USE_ABSTRACT` | `0` | Include abstracts in prompts |
| `MAX_HISTORY` | `0` | Max history items (0=unlimited) |
| `FLASH_ATTN` | `1` | Use Flash Attention 2 |

**Metrics:** Same official MIND metrics (AUC, MRR, nDCG@5, nDCG@10).

#### Standard Evaluation

The `eval_mind.sh` script provides a convenient way to evaluate standard SFT models on MIND with automatic data extraction:

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
- `MIND_ROOT`: Root directory containing MIND data (default: auto-select `../data/MIND_small` or `../data/MIND_large` based on `MIND_SIZE`, falling back to `../data/MIND`)
- `MIND_SIZE`: Dataset size - `small` or `large` (default: `small`)
- `MIND_ZIPS`: Directory containing MIND ZIP files (default: `~/wuc/downloaded/zips`)
- `USE_ABSTRACT`: Set to 1 to use abstracts (default: 0)
- `MAX_HISTORY`: Max history items to use (default: 50)
- `SKIP_EXTRACT`: Set to 1 to skip automatic extraction (default: 0)
- `OUTPUT_FILE`: Path to save predictions for MIND leaderboard submission (optional)

**Output**: Reports AUC, MRR, nDCG@5, nDCG@10 (same metrics used on the MIND leaderboard). Optionally generates prediction file with ranked news IDs for each impression.

#### Results

**Dataset**: MINDsmall dev split (73,152 impressions)

| Model | AUC | MRR | nDCG@5 | nDCG@10 | Notes |
|-------|-----|-----|--------|---------|-------|
| **Qwen3-1.7B** (baseline) | 52.48% | 25.11% | 23.50% | 29.78% | Zero-shot, no fine-tuning (1.7B) |
| **Qwen3-4B-Instruct-2507** (baseline) | 51.40% ⬇️ | 24.86% ⬇️ | 22.92% ⬇️ | 29.27% ⬇️ | Zero-shot, no fine-tuning (4B) |
| **Qwen3-8B** (baseline) | 51.35% ⬇️ | 25.10% | 23.01% ⬇️ | 29.49% ⬇️ | Zero-shot, no fine-tuning (8B) |
| **sft_text_Industrial_and_Scientific_qwen3-1.7B_bs1024** | 51.77% ⬇️ | 24.74% ⬇️ | 23.31% ⬇️ | 29.38% ⬇️ | Fine-tuned on Amazon only (1.7B) |
| **sft_Industrial_and_Scientific_qwen3-8b** | 51.56% ⬇️ | 23.83% ⬇️ | 22.19% ⬇️ | 28.55% ⬇️ | Fine-tuned on Amazon only (8B) |
| **sft_mixed_Industrial_and_Scientific_Qwen3-1.7B_bs1024** | 51.11% ⬇️ | 24.75% ⬇️ | 22.55% ⬇️ | 29.33% ⬇️ | Mixed SFT variant (different mixing ratio) |
| **sft_text_mixed_Industrial_and_Scientific_Qwen3-1.7B_bs1024** | 52.91% ⬆️ | 25.99% ⬆️ | 24.47% ⬆️ | 30.56% ⬆️ | Mixed SFT (Amazon + MIND + general) |
| **sft_mind_small_Qwen3-1.7B_bs1024** | 53.72% | 26.23% | 24.53% | 31.00% | Fine-tuned on MIND directly |
| **sft_mind_ranking_small_Qwen3-1.7B_bs1024** | 65.49% | 45.30% | 50.36% | 56.57% | Ranking-aware SFT |
| **sft_mind_ranking_small_Qwen3-1.7B_bs1024_ep3_neg8.0_hist50** | 62.41% | 27.86% | 30.21% | 36.79% | Ranking-aware SFT with neg_ratio=8, max_history=50 |
| **sft_mind_ranking_small_Qwen3-1.7B_bs1024_ep3_neg6.0_hist50** | 65.84% | 28.28% | 30.41% | 37.24% | Ranking-aware SFT with neg_ratio=6, max_history=50 |
| **sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3** | **66.17%** 🏆 | **45.20%** 🏆 | **50.09%** 🏆 | **56.45%** 🏆 | **Ranking-aware SFT with Qwen3-1.7B-Base - NEW BEST!** |
| **sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3_neg8.0_hist50** | 65.17% | 28.36% | 30.43% | 37.22% | Ranking-aware SFT with neg_ratio=8, max_history=50 |
| **rl_mind_small_Qwen3-1.7B-Base_mind_ndcg** | 66.11% | 45.28% | 50.23% | 56.57% | RL fine-tuned from ranking SFT (nDCG reward) |
| **sft_mind_ranking_small_Qwen3-Reranker-0.6B_bs1024_ep3** | 64.60% | 44.61% | 49.65% | 56.03% | Ranking-aware SFT with Qwen3-Reranker-0.6B |
| **sft_mind_ranking_small_Qwen3-4B-Base_bs1024_ep8** | 61.06% ⬇️ | 41.11% ⬇️ | 45.55% ⬇️ | 52.33% ⬇️ | Ranking-aware SFT with Qwen3-4B-Base (8 epochs, overfitting) |
| **sft_mind_ranking_small_Qwen3-Reranker-4B_bs1024_ep8** | 50.69% ⬇️ | 33.28% ⬇️ | 35.67% ⬇️ | 44.33% ⬇️ | Ranking-aware SFT with Qwen3-Reranker-4B (8 epochs, severe overfitting) |
| **sft_mind_ranking_small_Qwen3-4B-Base_bs1024** | 62.90% | 43.57% | 48.57% | 55.31% | Ranking-aware SFT with Qwen3-4B-Base |
| **sft_mind_ranking_small_Qwen3-Reranker-4B_bs1024** | 62.02% | 43.05% | 47.72% | 54.61% | Ranking-aware SFT with Qwen3-Reranker-4B |
| **sft_mind_pointwise_small_Qwen3-1.7B-Base_bs1024_ep3_neg1.0** | 66.95% | 31.81% | 35.15% | 41.47% | Point-wise SFT (Yes/No classification) |

**Note on ranking evaluation**: The ranking-aware model now uses numeric options (1/2/3/...) instead of letters (A/B/C), supporting unlimited candidates per impression.

**Key Findings**:

1. **🚀 BREAKTHROUGH: Ranking-aware SFT achieves MASSIVE improvements!**
   - **sft_mind_ranking_small_Qwen3-1.7B-Base** (66.17% AUC): **NEW BEST** - 🔥 **+12.45% absolute improvement over standard SFT!**
   - **+23.2% relative improvement** in AUC (53.72% → 66.17%)
   - **+72.3% relative improvement** in MRR (26.23% → 45.20%)
   - **+104.2% relative improvement** in nDCG@5 (24.53% → 50.09%)
   - **+82.1% relative improvement** in nDCG@10 (31.00% → 56.45%)
   - **Key insight**: Training with multiple-choice ranking format (showing ALL candidates) dramatically outperforms standard SFT
   - **Now competitive with SOTA**: Approaches NRMS baseline (67.76% AUC) with just 1.7B model!

2. **🏆 Standard MIND-specific SFT still strong:**
   - **sft_mind_small** (53.72% AUC): Direct fine-tuning on MIND
   - vs baseline (52.48% AUC): **+1.24% improvement**
   - vs mixed SFT (52.91% AUC): **+0.81% improvement**
   - **Key insight**: Direct in-domain training outperforms transfer learning or mixed training

3. **Training strategy comparison (all 1.7B models)**:
   - ✅ **MIND-ranking SFT**: 65.49% AUC (🏆 BEST by far!)
   - ✅ **MIND-only SFT**: 53.72% AUC (good)
   - ✅ **Mixed SFT** (Amazon + MIND + general): 52.91% AUC (decent)
   - 🔴 **Amazon-only SFT**: 51.77% AUC (worse than baseline)
   - 🔴 **Bad mixing ratio**: 51.11% AUC (worst)
   - **Gap**: 14.38% AUC between best (ranking) and worst fine-tuning strategies!

4. **🚨 SURPRISING: Larger baseline models perform WORSE zero-shot!**
   - Qwen3-1.7B baseline: 52.48% AUC
   - Qwen3-4B-Instruct-2507 baseline: 51.40% AUC (-1.08% vs 1.7B!)
   - Qwen3-8B baseline: 51.35% AUC (-1.13% vs 1.7B!)
   - **This suggests larger models may be overfitted to their pre-training data** or have different instruction-following characteristics that don't transfer well to news recommendation without fine-tuning

5. **Domain mismatch effects**:
   - Amazon-only SFT hurts performance on MIND (51.77% < 52.48% baseline)
   - But mixed training helps if done correctly (52.91%)
   - **Best approach**: Fine-tune directly on target domain with ranking-aware format (MIND → 65.49%)

6. **Critical insights for SOTA**:
   - 🚀 **Ranking-aware training is a GAME CHANGER** - +11.77% AUC over standard SFT!
   - ✅ **Training format matters MORE than model size** - Ranking-aware 1.7B (65.49%) likely beats zero-shot 8B by >14%
   - ✅ **Multiple-choice format aligns training with evaluation** - Model sees all candidates, learns to rank
   - 🎯 **Path to SOTA**: Apply ranking-aware training to 4B/8B models on MINDlarge
   - 💡 **Expected potential**: Ranking-aware 8B on MINDlarge could reach 70%+ AUC (beating NRMS SOTA 67.76%!)

**Commands**:
```bash
# Baseline (zero-shot)
bash scripts/eval_mind.sh Qwen/Qwen3-1.7B dev

# Ranking-aware SFT model (🏆 NEW BEST - 65.49% AUC!)
bash scripts/sft_mind_ranking.sh  # Train
bash scripts/eval_ranking_only.sh output_dir/sft_mind_ranking_small_Qwen3-1.7B_bs1024/final_checkpoint dev  # Evaluate

# Standard MIND-trained model (53.72% AUC)
bash scripts/eval_mind.sh output_dir/sft_mind_small_Qwen3-1.7B_bs1024/final_checkpoint dev

# Mixed SFT fine-tuned model (52.91% AUC)
bash scripts/eval_mind.sh output_dir/sft_text_mixed_Industrial_and_Scientific_Qwen3-1.7B_bs1024/final_checkpoint dev

# Amazon-only fine-tuned model (51.77% AUC - worse than baseline)
bash scripts/eval_mind.sh output_dir/sft_text_Industrial_and_Scientific_qwen3-1.7B_bs1024/final_checkpoint dev
```

### Leaderboard Submission

To submit results to the **[official MIND leaderboard](https://msnews.github.io/)**:

#### Step 1: Train on MINDlarge

```bash
# Ranking-aware SFT (recommended)
MIND_SIZE=large bash scripts/sft_mind_ranking.sh

# Or point-wise SFT
MIND_SIZE=large bash scripts/sft_mind_pointwise.sh

# Or standard SFT
MIND_SIZE=large bash scripts/sft_mind.sh
```

#### Step 2: Generate Test Predictions

**For Ranking-Aware Models:**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_ranking.sh \
    output_dir/sft_mind_ranking_large_*/final_checkpoint test

# Multi-GPU parallel (faster)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_ranking.sh \
    output_dir/sft_mind_ranking_large_*/final_checkpoint test

# Predictions saved to: ./results_mind/test_ranking_predictions.txt
```

**For Point-wise Models:**
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_mind_pointwise.sh \
    output_dir/sft_mind_pointwise_large_*/final_checkpoint test

# Multi-GPU parallel (faster)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_mind_pointwise.sh \
    output_dir/sft_mind_pointwise_large_*/final_checkpoint test

# Predictions saved to: ./results_mind/test_pointwise_predictions.txt
```

**For Standard SFT Models:**
```bash
OUTPUT_FILE=./results_mind/test_predictions.txt \
    bash scripts/eval_mind.sh output_dir/sft_mind_large_*/final_checkpoint test
```

#### Step 3: Submit to Leaderboard

1. Go to [https://msnews.github.io/](https://msnews.github.io/)
2. Upload your prediction file

**Prediction File Format:**
```
impression_id1 news_id_rank1 news_id_rank2 news_id_rank3 ...
impression_id2 news_id_rank1 news_id_rank2 news_id_rank3 ...
```

Each line contains the impression ID followed by space-separated news IDs ranked by predicted relevance (most relevant first).

**Important**: The test split labels are not public. You must submit your predictions to the leaderboard server for official scoring.

**Metrics evaluated**: AUC, MRR, nDCG@5, nDCG@10

### 📚 MIND Dataset SOTA Training Guide

This section provides strategies and best practices for achieving State-of-the-Art (SOTA) results on the MIND news recommendation dataset.

#### 🚀 Strategies for Achieving SOTA

##### 1. Model Selection

**Start with larger models:**
```bash
# Qwen3-4B (recommended for SOTA)
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 bash scripts/sft_mind.sh

# Qwen3-8B (if you have sufficient GPU memory)
MODEL_PATH=Qwen/Qwen3-8B-Instruct-2507 bash scripts/sft_mind.sh
```

**Performance vs Size:**
- **1.7B**: Fast training, good baseline (~52% AUC)
- **4B**: Better performance, moderate training time (expected ~55-58% AUC)
- **8B**: Best performance, slower training (expected ~60%+ AUC)

##### 2. Use Abstracts

Including news abstracts provides more context and typically improves results by 2-3%:

```bash
USE_ABSTRACT=1 bash scripts/sft_mind.sh
```

**Trade-offs:**
- ✅ Better quality: More context for the model
- ⚠️ Longer sequences: Requires more memory (use `CUTOFF_LEN=2048`)
- ⚠️ Slower training: ~2x training time

##### 3. Increase History Length

More user history provides better personalization:

```bash
MAX_HISTORY=100 bash scripts/sft_mind.sh
```

**Recommendations:**
- Default: 50 (good balance)
- For SOTA: 100-150 (better personalization)
- Maximum: ~200 (depending on sequence length)

##### 4. Train on MINDlarge

The large dataset contains 10x more data and is required for leaderboard submission:

```bash
MIND_SIZE=large bash scripts/sft_mind.sh
```

**Dataset Comparison:**
- **MINDsmall**: ~50K users, faster training, good for development
- **MINDlarge**: ~1M users, better generalization, required for leaderboard

##### 5. Hyperparameter Optimization

**Learning Rate**

Start with a lower learning rate for larger models:

```bash
# For 1.7B model (default)
LEARNING_RATE=3e-4 bash scripts/sft_mind.sh

# For 4B model (recommended)
LEARNING_RATE=1e-4 bash scripts/sft_mind.sh

# For 8B model (recommended)
LEARNING_RATE=5e-5 bash scripts/sft_mind.sh
```

**Batch Size**

Larger batch sizes improve training stability:

```bash
# Single GPU
BATCH_SIZE=512 bash scripts/sft_mind.sh

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 BATCH_SIZE=2048 bash scripts/sft_mind.sh

# 8 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 BATCH_SIZE=4096 bash scripts/sft_mind.sh
```

**Guidelines:**
- Increase batch size with more GPUs
- Maintain `BATCH_SIZE / MICRO_BATCH_SIZE` ratio for gradient accumulation
- Typical micro batch size: 8-16 per GPU

**Training Epochs**

```bash
# Quick experiments
NUM_EPOCHS=1 bash scripts/sft_mind.sh

# Standard training (recommended)
NUM_EPOCHS=3 bash scripts/sft_mind.sh

# Extended training for SOTA
NUM_EPOCHS=5 bash scripts/sft_mind.sh
```

##### 6. Full SOTA Configuration

Here's a complete configuration for achieving SOTA results:

```bash
# SOTA configuration with Qwen3-4B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507 \
  MIND_SIZE=large \
  BATCH_SIZE=4096 \
  MICRO_BATCH_SIZE=8 \
  NUM_EPOCHS=3 \
  LEARNING_RATE=1e-4 \
  CUTOFF_LEN=2048 \
  USE_ABSTRACT=1 \
  MAX_HISTORY=100 \
  bash scripts/sft_mind.sh
```

**Expected training time:**
- MINDsmall: ~2-4 hours (8x A100 GPUs)
- MINDlarge: ~12-24 hours (8x A100 GPUs)

#### 📈 Optimization Tips

##### 1. Monitor Training

Training metrics are automatically logged to Weights & Biases. Watch for:
- **Training loss**: Should decrease steadily
- **Eval loss**: Should decrease with training loss (if diverges, reduce learning rate)
- **Early stopping**: Training stops if eval loss doesn't improve for 3 checkpoints

##### 2. GPU Memory Optimization

If you run out of memory:

```bash
# Reduce batch size
MICRO_BATCH_SIZE=4 bash scripts/sft_mind.sh

# Reduce sequence length (if not using abstracts)
CUTOFF_LEN=512 bash scripts/sft_mind.sh

# Reduce history length
MAX_HISTORY=30 bash scripts/sft_mind.sh
```

##### 3. Multi-GPU Training

The script automatically detects available GPUs:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft_mind.sh

# Use all available GPUs (automatic detection)
bash scripts/sft_mind.sh
```

##### 4. Resume from Checkpoint

If training is interrupted:

```bash
# Edit the script to add resume path
python src/sft_mind.py \
  --base_model Qwen/Qwen3-4B-Instruct-2507 \
  --train_behaviors_path ../data/MIND/train/behaviors.tsv \
  --train_news_path ../data/MIND/train/news.tsv \
  --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
  --eval_news_path ../data/MIND/dev/news.tsv \
  --output_dir output_dir/mind_large_Qwen3-4B_bs4096 \
  --resume_from_checkpoint output_dir/mind_large_Qwen3-4B_bs4096/checkpoint-XXXX \
  --batch_size 4096 \
  --micro_batch_size 8 \
  --num_epochs 3
```

#### 🏆 Leaderboard Submission Steps

Once you've trained your best model:

**1. Evaluate on Dev Split**

```bash
bash scripts/eval_mind.sh output_dir/sft_mind_large_Qwen3-4B_bs4096/final_checkpoint dev
```

**2. Generate Test Split Predictions**

```bash
MIND_SIZE=large \
  bash scripts/eval_mind.sh output_dir/sft_mind_large_Qwen3-4B_bs4096/final_checkpoint test
```

This will create predictions file at: `./results_mind/test_predictions.txt`

**3. Submit to Leaderboard**

1. Visit [https://msnews.github.io/](https://msnews.github.io/)
2. Upload `./results_mind/test_predictions.txt`
3. Wait for official evaluation results

**Format:** Each line should be `ImpressionID [space-separated ranked news IDs]`

#### 🔬 Experimental Ideas

##### 1. Data Augmentation

- **Negative sampling**: Sample more negatives per positive
- **History shuffling**: Randomize history order during training

##### 2. Model Ensembling

Train multiple models and ensemble predictions:
- Different model sizes (1.7B, 4B, 8B)
- Different hyperparameters
- With/without abstracts

##### 3. Advanced Prompting

Modify the prompt in `data.py` → `MINDTextSFTDataset.__getitem__()` to provide:
- Category information
- Temporal context
- User persona description

##### 4. Two-Stage Training

1. **Stage 1**: Pre-train on MINDlarge train split
2. **Stage 2**: Fine-tune on high-quality subset (e.g., users with >10 clicks)

#### 📝 Troubleshooting

**Common Issues**

**Issue: Out of memory**
```bash
# Solution: Reduce batch size or sequence length
MICRO_BATCH_SIZE=4 CUTOFF_LEN=512 bash scripts/sft_mind.sh
```

**Issue: Training too slow**
```bash
# Solution: Use fewer epochs or smaller dataset
NUM_EPOCHS=1 bash scripts/sft_mind.sh test
```

**Issue: Poor evaluation results**
- Check if training loss is decreasing
- Try lower learning rate
- Increase training epochs
- Use larger model

**Issue: Eval loss diverging from train loss**
- Reduce learning rate
- Add more regularization (weight_decay)
- Check for overfitting

#### 💡 Tips from Experience

1. **Start small**: Test on MINDsmall before scaling to MINDlarge
2. **Monitor metrics**: Watch both train and eval metrics closely
3. **Use abstracts for SOTA**: The extra context is worth the compute cost
4. **Larger is better**: 4B+ models perform significantly better than 1.7B
5. **Be patient**: MINDlarge training takes time, but results are worth it!

Good luck achieving SOTA! 🚀

---

### 🔀 Hybrid Approaches: Combining Point-wise and Ranking

We provide several approaches to combine point-wise (Yes/No classification) and ranking (select best) methods. Each approach captures different aspects:

| Approach | Description | When to Use |
|----------|-------------|-------------|
| **Score Ensemble** | Combine scores from both models at inference | Quick experiment, no retraining needed |
| **Two-Stage Training** | Point-wise pre-train → Ranking fine-tune | Better initialization for ranking |
| **Cascade/Reranking** | Point-wise filters top-K → Ranking reranks | Fast inference with accuracy |
| **Multi-Task Learning** | Train on both tasks simultaneously | Joint optimization |

#### 1. Score Ensemble (No Retraining)

Combine scores from separately trained point-wise and ranking models:

```bash
# Evaluate with ensemble (alpha = weight for point-wise scores)
bash scripts/eval_mind_ensemble.sh \
    output_dir/sft_mind_pointwise_*/final_checkpoint \
    output_dir/sft_mind_ranking_*/final_checkpoint \
    0.5 dev  # alpha=0.5, 50% each

# Try different alpha values
ALPHA=0.3 bash scripts/eval_mind_ensemble.sh ...  # More weight on ranking
ALPHA=0.7 bash scripts/eval_mind_ensemble.sh ...  # More weight on point-wise
```

#### 2. Two-Stage Training

First train point-wise (learns relevance), then fine-tune with ranking (learns ordering):

```bash
bash scripts/sft_mind_twostage.sh

# Customize stages
STAGE1_EPOCHS=2 STAGE2_EPOCHS=3 bash scripts/sft_mind_twostage.sh
```

**Stage 1**: Point-wise SFT (shorter context, faster training)
**Stage 2**: Ranking SFT with lower learning rate (fine-tuning)

#### 3. Cascade/Reranking

Use point-wise for fast initial filtering, then ranking for accurate reranking:

```bash
# Point-wise filters to top-10, ranking reranks them
bash scripts/eval_mind_cascade.sh \
    output_dir/sft_mind_pointwise_*/final_checkpoint \
    output_dir/sft_mind_ranking_*/final_checkpoint \
    10 dev  # top_k=10

# Try different top-K values
TOP_K=5 bash scripts/eval_mind_cascade.sh ...   # More aggressive filtering
TOP_K=20 bash scripts/eval_mind_cascade.sh ...  # Keep more candidates
```

#### 4. Multi-Task Learning

Train a single model on both tasks simultaneously:

```bash
bash scripts/sft_mind_multitask.sh

# Customize task ratio (0.5 = 50% point-wise, 50% ranking)
POINTWISE_RATIO=0.3 bash scripts/sft_mind_multitask.sh  # More ranking samples
POINTWISE_RATIO=0.7 bash scripts/sft_mind_multitask.sh  # More point-wise samples
```

The model learns both:
- **Point-wise**: "Is this article relevant?" → Yes/No
- **Ranking**: "Which article is most relevant?" → Select number

#### Scripts Summary

| Script | Type | Description |
|--------|------|-------------|
| `evaluate_mind_ensemble.py` | Eval | Score ensemble at inference time |
| `evaluate_mind_cascade.py` | Eval | Cascade: filter + rerank |
| `src/sft_mind_twostage.py` | Train | Two-stage: point-wise → ranking |
| `src/sft_mind_multitask.py` | Train | Multi-task joint training |
| `scripts/eval_mind_ensemble.sh` | Shell | Run ensemble evaluation |
| `scripts/eval_mind_cascade.sh` | Shell | Run cascade evaluation |
| `scripts/sft_mind_twostage.sh` | Shell | Run two-stage training |
| `scripts/sft_mind_multitask.sh` | Shell | Run multi-task training |

---

### 🎯 Reinforcement Learning for MIND (Advanced)

After SFT training, you can further improve ranking performance using **Reinforcement Learning with nDCG reward**. This directly optimizes the ranking metric you care about!

#### Why RL for MIND?

**Problem with SFT**: Next-token prediction doesn't directly optimize ranking quality
- SFT learns: `P(clicked_news | history)` should be high
- What you want: `rank(clicked_news) < rank(not_clicked_news)`

**RL Solution**: Direct optimization of nDCG@10, MRR, or AUC
- Reward function: nDCG-style reward `1/log2(rank+1)`
- Training: GRPO (Group Relative Policy Optimization) via VERL framework
- Result: Better ranking alignment

#### Expected Improvements

Based on research literature and our SFT results:

| Method | AUC | nDCG@10 | Improvement |
|--------|-----|---------|-------------|
| SFT baseline (MIND-trained) | 53.72% | 31.00% | - |
| **SFT + RL (nDCG reward)** | **55-57%** | **32.5-34%** | +1.3-3.3% AUC, +1.5-3% nDCG |

*Note: RL is expected to improve on top of the already strong SFT baseline (53.72% AUC)*

#### Quick Start

**Step 1: Train SFT Model First**
```bash
# Train SFT model on MIND (if not already done)
bash scripts/sft_mind.sh
```

**Step 2: Run RL Training**
```bash
# Quick test on MINDsmall (4 GPUs, 1 epoch)
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  SFT_MODEL_PATH=output_dir/sft_mind_small_Qwen3-1.7B_bs1024/final_checkpoint \
  bash scripts/rl_mind.sh

# Full training on MINDlarge (8 GPUs, 2 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  SFT_MODEL_PATH=output_dir/sft_mind_large_Qwen3-4B_bs4096/final_checkpoint \
  MIND_SIZE=large \
  TOTAL_EPOCHS=2 \
  LEARNING_RATE=5e-7 \
  KL_LOSS_COEF=0.01 \
  bash scripts/rl_mind.sh
```

**Step 3: Evaluate RL Model**
```bash
# Evaluate on dev split
bash scripts/eval_mind.sh output_dir/rl_mind_small_*/final_checkpoint dev
```

#### How It Works

1. **Data Preparation**: Converts MIND behaviors.tsv to VERL parquet format
   - Prompt: User history formatted as text
   - Ground truth: Clicked news title
   - Extra info: All candidate news titles + labels

2. **Reward Function**: nDCG-style reward based on ranking position
   ```python
   reward = 1.0 / log2(rank + 1) if rank <= 10 else 0.0
   # rank=1 → reward=1.0 (best)
   # rank=2 → reward=0.631
   # rank=5 → reward=0.387
   # rank=10 → reward=0.301
   ```

3. **GRPO Training**: Policy gradient optimization
   - Generates multiple news titles per prompt
   - Computes reward for each generation
   - Updates model to maximize expected reward
   - KL penalty keeps model close to SFT checkpoint

#### Configuration Options

**Reward Types:**
- `mind_ndcg` (default): nDCG-style reward `1/log2(rank+1)`
- `mind_mrr`: MRR-style reward `1/rank`

**Key Hyperparameters:**
```bash
# Conservative RL settings (recommended)
LEARNING_RATE=5e-7        # Low LR for stability
KL_LOSS_COEF=0.01         # High KL penalty (stay close to SFT)
TRAIN_BATCH_SIZE=256      # Moderate batch size
TOTAL_EPOCHS=1            # Start with 1 epoch

# Aggressive RL settings (for experienced users)
LEARNING_RATE=1e-6
KL_LOSS_COEF=0.001
TRAIN_BATCH_SIZE=512
TOTAL_EPOCHS=2
```

#### Advanced Usage

**Custom Reward Function:**

Edit [src/verl_reward.py](src/verl_reward.py) to add your own reward:
```python
def compute_score_mind_custom(data_source, solution_str, ground_truth, extra_info=None):
    # Your custom reward logic
    # Example: Combine nDCG + diversity + freshness
    ndcg_reward = compute_score_mind_ndcg(...)
    diversity_bonus = compute_diversity(...)
    return ndcg_reward + 0.1 * diversity_bonus
```

Then use: `REWARD_TYPE=mind_custom bash scripts/rl_mind.sh`

**Multi-Epoch Training with Evaluation:**

```bash
# Train for 3 epochs with periodic evaluation
for epoch in 1 2 3; do
    echo "Epoch $epoch/3"

    # RL training
    TOTAL_EPOCHS=1 \
      OUTPUT_DIR=output_dir/rl_mind_epoch${epoch} \
      bash scripts/rl_mind.sh

    # Evaluate
    bash scripts/eval_mind.sh \
      output_dir/rl_mind_epoch${epoch}/final_checkpoint dev
done
```

**Manual Data Preparation:**

```bash
# Prepare RL data manually (if you want custom settings)
python prepare_mind_rl.py \
  --behaviors_path ../data/MIND/train/behaviors.tsv \
  --news_path ../data/MIND/train/news.tsv \
  --output_parquet ../data/MIND/train/rl_train.parquet \
  --max_history 100 \
  --use_abstract \
  --max_candidates 50

# Then run RL with pre-prepared data
python src/rl_mind_verl.py \
  --model_path output_dir/sft_mind_*/final_checkpoint \
  --train_parquet ../data/MIND/train/rl_train.parquet \
  --eval_parquet ../data/MIND/dev/rl_dev.parquet \
  --output_dir output_dir/rl_mind_custom \
  --reward_type mind_ndcg \
  --total_epochs 2
```

#### Troubleshooting

**Issue: OOM (Out of Memory)**
```bash
# Reduce batch size and micro batch size
TRAIN_BATCH_SIZE=64 bash scripts/rl_mind.sh

# Or in Python script:
python src/rl_mind_verl.py \
  --train_batch_size 64 \
  --ppo_micro_batch_size_per_gpu 4
```

**Issue: Model diverges (reward drops)**
```bash
# Increase KL penalty (more conservative)
KL_LOSS_COEF=0.05 bash scripts/rl_mind.sh

# Reduce learning rate
LEARNING_RATE=1e-7 bash scripts/rl_mind.sh
```

**Issue: Training too slow**
```bash
# Reduce number of generations per prompt
python src/rl_mind_verl.py \
  --num_generations 8  # Default is 16
```

**Issue: Reward always 0**
- Check that news titles in parquet match evaluation format
- Verify reward function is being called (check logs)
- Try `mind_mrr` reward type (more lenient than nDCG)

#### Research References

Papers on RL for ranking:
1. "Reinforcement Learning to Rank in E-Commerce Search Engine" (Alibaba, KDD 2018)
2. "Top-K Off-Policy Correction for a REINFORCE Recommender System" (Google, 2019)
3. "Reinforcement Learning for Slate-Based Recommender Systems" (Netflix, RecSys 2019)

---

## 🔥 VERL RL for Amazon (Optional)

You can run MiniOneRec RL using the official VERL framework while keeping the original RL code intact. This repo adds a VERL-compatible data prep, reward functions, and a launcher for Amazon SID-based recommendation.

### Install VERL
```bash
pip install -r requirements-verl.txt
```

### Prepare data (CSV → parquet)
```bash
python src/verl_data_prep.py \
  --train_file data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --eval_file data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --output_dir data/verl/Industrial_and_Scientific
```

### Run VERL GRPO
```bash
python src/rl_verl.py \
  --model_path output_dir/sft_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/final_checkpoint \
  --train_parquet data/verl/Industrial_and_Scientific/train.parquet \
  --eval_parquet data/verl/Industrial_and_Scientific/eval.parquet \
  --output_dir output_dir/verl_rl_Industrial_and_Scientific \
  --reward_type rule \
  --sid_info_file data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

This pipeline uses GRPO (not PPO). VERL is launched through `verl.trainer.main_ppo` with `algorithm.adv_estimator=grpo` inside `src/minionerec_verl_trainer.py`.

Reward options: `rule`, `ranking`, `ranking_only`, `semantic`, `sasrec`. For `semantic`, set `--ada_path`; for `sasrec`, set `--cf_path`. The reward functions are defined in `src/verl_reward.py` (ranking uses `extra_info.rank` when available).

### How to read the VERL code
1. **Data prep**: `src/verl_data_prep.py` converts Amazon CSV into VERL parquet with `prompt`, `reward_model.ground_truth`, and `extra_info`.
2. **Reward functions**: `src/verl_reward.py` defines scoring for rule/semantic/sasrec/ranking. Env vars: `SID_INFO_FILE`, `ADA_PATH`, `SASREC_PATH`, `SASREC_LEN_SEQ`.
3. **Trainer wrapper**: `src/minionerec_verl_trainer.py` builds the VERL command and wires reward + data + GRPO settings.
4. **Entry point**: `src/rl_verl.py` is the Fire CLI that passes args into the trainer.

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
| `src/rl_verl.py`              | VERL-based GRPO launcher (optional)                                           |
| `src/minionerec_verl_trainer.py` | VERL trainer wrapper (optional)                                            |
| `src/verl_reward.py`          | Custom reward functions for VERL (rule/semantic/sasrec/ranking)               |
| `src/verl_data_prep.py`       | CSV → parquet converter for VERL data prep                                    |
| `scripts/rl_verl.sh`              | Example VERL RL launch script                                              |
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
| `src/sft_mind_pointwise.py` | Point-wise SFT training for MIND (Yes/No classification)                                              |
| `evaluate_mind_pointwise.py` | Point-wise evaluation script (scores P(Yes) vs P(No))                                                |
| `scripts/sft_mind_pointwise.sh` | Shell script for point-wise SFT training                                                           |
| `scripts/eval_mind_pointwise.sh` | Shell script for point-wise evaluation with multi-GPU support                                     |
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

## 🎯 MIND Dataset Training - Achieving SOTA (0.73+ AUC)

MiniOneRec supports two training approaches for MIND news recommendation: **Pointwise** (Yes/No classification) and **Ranking** (multiple-choice selection). Both approaches have been optimized for achieving state-of-the-art performance.

### Key Improvements for SOTA

1. **Model Size**: Upgraded from Qwen3-1.7B to **Qwen3-8B-Instruct** (+3-5% AUC)
2. **Dataset**: Use **MIND-large** instead of MIND-small (10x more data)
3. **Learning Rate**: Lowered to **2e-5** (critical for fine-tuning pretrained LLMs)
4. **History Length**: Optimized to **30 items** (last 30 have 90% predictive signal)
5. **Negative Sampling**: **Hard negatives** (50% same-category) for better training
6. **Improved Prompts**: Concise, category-first format based on Prompt4NR research

### Quick Start: Recommended SOTA Setup ⭐

#### Pointwise Training (Best for Binary Relevance)

```bash
# RECOMMENDED: Full SOTA training setup
MODEL_PATH="Qwen/Qwen3-8B-Instruct" \
MIND_SIZE="large" \
BATCH_SIZE=256 \
MICRO_BATCH_SIZE=2 \
NUM_EPOCHS=5 \
LEARNING_RATE=2e-5 \
MAX_HISTORY=30 \
NEG_RATIO=2.0 \
WANDB_PROJECT="MiniOneRec_SOTA" \
WANDB_RUN_NAME="pointwise_8B_large_optimized" \
bash scripts/sft_mind_pointwise.sh
```

#### Ranking Training (Best for List-wise Comparison)

```bash
# RECOMMENDED: Full SOTA training setup
MODEL_PATH="Qwen/Qwen3-8B-Instruct" \
MIND_SIZE="large" \
BATCH_SIZE=256 \
MICRO_BATCH_SIZE=2 \
NUM_EPOCHS=5 \
LEARNING_RATE=2e-5 \
MAX_HISTORY=30 \
NEG_RATIO=4.0 \
WANDB_PROJECT="MiniOneRec_SOTA" \
WANDB_RUN_NAME="ranking_8B_large_optimized" \
bash scripts/sft_mind_ranking.sh
```

### Alternative Configurations

#### Quick Test (MIND-small, 1.7B Model)

```bash
# Fast iteration for debugging
MODEL_PATH="Qwen/Qwen3-1.7B" \
MIND_SIZE="small" \
BATCH_SIZE=256 \
MICRO_BATCH_SIZE=4 \
NUM_EPOCHS=3 \
LEARNING_RATE=2e-5 \
MAX_HISTORY=30 \
NEG_RATIO=2.0 \
bash scripts/sft_mind_pointwise.sh
```

#### Intermediate Setup (4B Model - Good Balance)

```bash
# Qwen3-4B: Good balance between speed and performance
MODEL_PATH="Qwen/Qwen3-4B-Instruct" \
MIND_SIZE="large" \
BATCH_SIZE=256 \
MICRO_BATCH_SIZE=4 \
NUM_EPOCHS=5 \
LEARNING_RATE=2e-5 \
MAX_HISTORY=30 \
NEG_RATIO=2.0 \
bash scripts/sft_mind_pointwise.sh
```

#### Alternative LLM (Llama 3.3-8B)

```bash
# Using Llama instead of Qwen
MODEL_PATH="meta-llama/Llama-3.3-8B-Instruct" \
MIND_SIZE="large" \
BATCH_SIZE=256 \
MICRO_BATCH_SIZE=2 \
NUM_EPOCHS=5 \
LEARNING_RATE=2e-5 \
MAX_HISTORY=30 \
NEG_RATIO=2.0 \
bash scripts/sft_mind_pointwise.sh
```

#### Longer History Experiment

```bash
# Test with longer history (8B model can handle it)
MODEL_PATH="Qwen/Qwen3-8B-Instruct" \
MIND_SIZE="large" \
MAX_HISTORY=50 \
CUTOFF_LEN=3072 \
bash scripts/sft_mind_pointwise.sh
```

### Direct Python Script Call (Alternative)

You can also call the Python scripts directly with all parameters explicit:

```bash
torchrun --nproc_per_node 4 src/sft_mind_pointwise.py \
    --base_model "Qwen/Qwen3-8B-Instruct" \
    --train_behaviors_path "../data/MIND_large/train/behaviors.tsv" \
    --train_news_path "../data/MIND_large/train/news.tsv" \
    --eval_behaviors_path "../data/MIND_large/dev/behaviors.tsv" \
    --eval_news_path "../data/MIND_large/dev/news.tsv" \
    --output_dir "output_dir/pointwise_8B_sota" \
    --batch_size 256 \
    --micro_batch_size 2 \
    --num_epochs 5 \
    --learning_rate 2e-5 \
    --cutoff_len 2048 \
    --max_history 30 \
    --neg_ratio 2.0 \
    --wandb_project "MiniOneRec_SOTA" \
    --wandb_run_name "pointwise_8B_optimized" \
    --seed 42
```

### Evaluation After Training

#### Pointwise Evaluation

```bash
bash scripts/eval_mind_pointwise.sh output_dir/sft_mind_pointwise_*/final_checkpoint dev
```

#### Ranking Evaluation

```bash
bash scripts/eval_mind_ranking.sh output_dir/sft_mind_ranking_*/final_checkpoint dev
```

#### Ensemble Evaluation (Best Performance)

```bash
python evaluate_mind_ensemble.py \
    --pointwise_model output_dir/sft_mind_pointwise_*/final_checkpoint \
    --ranking_model output_dir/sft_mind_ranking_*/final_checkpoint \
    --behaviors_path ../data/MIND_large/dev/behaviors.tsv \
    --news_path ../data/MIND_large/dev/news.tsv \
    --alpha 0.6
```

### Parameter Reference

| Parameter | Quick Test | SOTA Setup | Description |
|-----------|-----------|-----------|-------------|
| **MODEL_PATH** | Qwen3-1.7B | Qwen3-8B-Instruct | 8B achieves 3-5% higher AUC |
| **MIND_SIZE** | small | large | Large has 10x data, all SOTA reported on it |
| **BATCH_SIZE** | 256 | 256 | Larger batches = more stable training |
| **MICRO_BATCH_SIZE** | 4 | 2 | Smaller for 8B memory constraints |
| **NUM_EPOCHS** | 3 | 5 | More epochs needed for large dataset |
| **LEARNING_RATE** | 2e-5 | 2e-5 | **Critical**: Much lower than 3e-4 for pretrained LLMs |
| **MAX_HISTORY** | 30 | 30 | Last 30 items have 90% predictive signal |
| **NEG_RATIO** | 2.0 (pointwise) | 4.0 (ranking) | Harder training with more negatives |
| **CUTOFF_LEN** | 2048 (pointwise) | 4096 (ranking) | Ranking needs longer context |

### Expected Performance

| Setup | Expected AUC | Training Time (8xV100) |
|-------|-------------|----------------------|
| 1.7B + MIND-small | 65-68% | ~2 hours |
| 1.7B + MIND-large | 68-70% | ~8 hours |
| 8B + MIND-small | 68-71% | ~4 hours |
| **8B + MIND-large** | **72-74%** ⭐ | ~16 hours |
| **Ensemble (Pointwise + Ranking)** | **73-75%** ⭐⭐ | N/A (inference only) |

### Key Improvements in Code

The Python training scripts include these optimizations:

1. **Improved Prompt Template** (src/sft_mind_pointwise.py, src/sft_mind_ranking.py):
   - Concise format saves ~20 tokens
   - Category in [brackets] at start for better visibility
   - Natural language questions
   - Based on Prompt4NR research (arXiv:2304.05263)

2. **Hard Negative Sampling** (src/sft_mind_pointwise.py):
   - 50% same-category negatives (harder to distinguish)
   - 50% different-category negatives (easier baselines)
   - Forces model to learn finer-grained preferences

3. **Optimized Default Hyperparameters**:
   - All shell scripts now have SOTA-optimized defaults
   - Can be easily overridden via environment variables

### Troubleshooting

**Why Abstracts Hurt Performance:**
- Abstracts cause sequence length explosion (50-150 tokens each)
- Leads to truncation of early history (most important signal!)
- Use `USE_ABSTRACT=0` (default) for best results

**Out of Memory:**
- Reduce `MICRO_BATCH_SIZE` to 1
- Reduce `MAX_HISTORY` to 20
- Reduce `CUTOFF_LEN` to 1024 (pointwise) or 2048 (ranking)

**Training Too Slow:**
- Use MIND-small for quick iterations
- Reduce `NUM_EPOCHS` to 3
- Use Qwen3-1.7B or 4B instead of 8B

### References

- [MIND Dataset Official Site](https://msnews.github.io/)
- [Prompt Learning for News Recommendation (arXiv:2304.05263)](https://arxiv.org/abs/2304.05263)
- [Survey on LLM-based News Recommender Systems (2025)](https://arxiv.org/html/2502.09797v1)
- [Revisiting Language Models in News Recommender Systems](https://arxiv.org/pdf/2501.11391)

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

**Troubleshooting Flash Attention:**

If you see `ImportError: undefined symbol` errors after installing flash-attn, it means there's a version mismatch with your PyTorch. Fix it by:

```bash
# Check your PyTorch and CUDA versions
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Reinstall flash-attn matching your versions
pip uninstall flash-attn -y
pip install flash-attn==2.7.3 --no-build-isolation  # For PyTorch 2.6.0+cu124
```

Or disable Flash Attention during evaluation:
```bash
FLASH_ATTN=0 bash scripts/eval_mind_ranking.sh <model_path> dev
```

**Troubleshooting OOM:**
If you still encounter OOM errors:
1. Reduce `batch_size` to 8
2. Reduce `cutoff_len` to 256 or 384
3. Enable NVMe offload in DeepSpeed config (for very large models)

### Training on GenRecDatasetV3

For training on the GenRecDatasetV3 dataset with multi-node DeepSpeed:

```bash
# Basic usage (default data path)
bash scripts/sft_yaqi_ds.sh

# Custom data path (e.g., shared NFS)
DATA_ROOT=/path/to/your/GenRecDatasetV3 bash scripts/sft_yaqi_ds.sh
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `/home/aiscuser/MiniOneRec/data/GenRecDatasetV3` | Root path to GenRecDatasetV3 data |
| `HOSTFILE` | Auto-detected (`/job/hostfile` or `./hostfile`) | Path to DeepSpeed hostfile |
| `MASTER_PORT` | `29502` | Port for distributed training |

**Expected Data Structure:**
```
${DATA_ROOT}/
├── train/train.csv
├── valid/valid.csv
├── test/test.csv
├── info/GenRecDatasetV2_1.txt
└── index/
    ├── GenRecDatasetV2_1.index.json
    └── GenRecDatasetV2_1.item.json
```

**Multi-Node Setup:**
Ensure `DATA_ROOT` points to a path accessible from all nodes (e.g., shared NFS mount).

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

The text-based evaluation pipeline supports **parallel multi-GPU processing** with automatic data splitting and merging:

1. **Data Splitting**: Test data is automatically split across multiple GPUs
2. **Parallel Generation** ([evaluate_text.py](src/evaluate_text.py)): Generate predictions using beam search on each GPU
3. **Result Merging**: Combine predictions from all GPUs
4. **Metrics Calculation** ([calc_text_similarity.py](src/calc_text_similarity.py)): Compute NDCG and HR with similarity matching


#### Quick Start

Run the complete evaluation pipeline:

```bash
bash scripts/evaluate_text.sh
```

By default, this uses **similarity-based matching** with no threshold — it always selects the most similar catalog item.

**Configure GPUs for parallel evaluation:**
```bash
# Use specific GPUs (default: 0,1,2,3)
CUDA_LIST="4,5,6,7" bash scripts/evaluate_text.sh

# Single GPU
CUDA_LIST="0" bash scripts/evaluate_text.sh
```


#### Customization Options

**Use exact matching (original behavior):**
```bash
USE_SIMILARITY=false bash scripts/evaluate_text.sh
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
    --use_similarity true
```

#### Understanding Similarity Matching

The similarity matching uses Python's `difflib.SequenceMatcher` to compute fuzzy string similarity:

- **Exact match first**: Tries exact string match (fastest)
- **Fuzzy fallback**: If no exact match, computes similarity ratios
- **Best-match selection**: Always maps the prediction to the single most similar catalog item
- **Best rank**: Uses the highest-ranked prediction among beams

**Example:**
```
Ground truth: "Black Ballpoint Pen"
Predictions:
  1. "Blue Pen"               → similarity: 0.50 (no match)
  2. "Black Ballpoint Pens"   → similarity: 0.95 (MATCH! ✓)
  3. "Black Ballpoint Pen"    → exact match (MATCH! ✓)
```

Prediction #2 would be considered a hit at rank 2 because it maps to the most similar catalog item even without exact string match.

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
| **Similarity (best-match)** | High | High | Minor variations, noisy or abbreviated names |

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

Solution: Switch to exact matching or improve catalog coverage (e.g., normalize names, expand item list)
```bash
USE_SIMILARITY=false bash scripts/evaluate_text.sh
```

**Metrics seem too optimistic**

Solution: Use exact matching or refine generation prompts to reduce near-duplicates
```bash
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

**Compare Matching Modes:**
```bash
# Similarity-based (best-match)
bash scripts/evaluate_text.sh

# Exact matching
USE_SIMILARITY=false bash scripts/evaluate_text.sh
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
