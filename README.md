# DOCA Pointwise CTR Prediction — Experiment Log

> Task: Feed recommendation click prediction (pointwise Yes/No)
> Created: 2026-04-23
> Branch: `zxy_dev_doca`

---

## 0. Quickstart

### Environment Setup

AML image: `torch251-cuda124-deepspeed-flashattn-training:5`

```bash
git clone https://github.com/overwindows/MiniOneRec
cd MiniOneRec
git checkout zxy_dev_doca
bash scripts/setup_multi_node.sh
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate MiniOneRec
```

### Download Data

Download from [cosmos09 MSN.DnI](https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/doca/data/260423/doca/) and put files into `data/doca/`:
- `train.jsonl`
- `dev.jsonl`

### Train

```bash
NUM_EPOCHS=1 bash scripts/sft_doca_pointwise_ds.sh
```

### Eval

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval_doca_pointwise.sh Qwen/Qwen3-1.7B --all
```

---

## 1. Data

### Source

- **Table**: `mai_ws_discover.analytics.ods_doca_feed_grounded_v7_partitioned`
- **Platform**: Databricks (Azure), host `adb-3355567219430035.15.azuredatabricks.net`
- **Date range**: 20260407 — 20260420 (14 days)
- **Split**: Train = first 12 days (0407–0418), Dev = last 2 days (0419–0420)
- **Prep script**: `python src/prepare_doca.py --output_dir data/doca --train_days 12 --dev_days 2`

### Statistics

| Split | Feeds | Candidates | Clicks | CTR |
|-------|------:|----------:|-------:|----:|
| Train | 114,156 | 446,827 | 30,878 | 6.91% |
| Dev   |  18,622 |  68,499 |  5,174 | 7.55% |

- Dev feeds with at least 1 click: 3,301 (17.7%)
- Only **impressed** candidates included (sectionIndex != None), filtering out ~67% non-shown cards
- Average ~3.9 candidates per feed (train), ~3.7 per feed (dev)

### Training Samples (after neg sampling with neg_ratio=2.0)

| Split | Total Samples | Positive | Negative | Actual Ratio |
|-------|-------------:|----------:|----------:|-------------:|
| Train | 170,625 | 30,878 | 139,747 | 1:4.53 |
| Val   |   5,000 |    935 |   4,065 | 1:4.35 |

Note: `neg_ratio=2.0` caps per-feed negative sampling at 2× positives, but many feeds' impressed non-click candidates are already below this cap, so all are retained. Actual ratio ends up ~4.5.

### Data Fields (per feed JSONL row)

| Field | Description |
|-------|-------------|
| `feed_id`, `user_id`, `ref_ts`, `bizdate` | Feed metadata |
| `candidates` | Array of {itemid, title, summary, is_clicked} — only impressed cards |
| `interests` | User interests with name, strength, domain, sources, intent, classification, status, keywords, rationale |
| `negative_interests` | Disliked topics with name, keywords, sources, rationale |
| `conversation` | Recent chat history with text, is_inline_curation flag |
| `shown_10d` | Recently shown articles {title, event_time} |

### Prompt Design

Each candidate is scored independently. The prompt includes:

1. **System prompt**: Role description + 7 ranking rules (from liquid ranking template) + output format
2. **User prompt**: interests → negative_interests → conversation (with [CURATED] tag) → shown_10d → candidate title/summary → "Will this user click on this article? Answer:"

See `data.py` (`DOCAPointwiseSFTDataset`) for training prompt, `src/evaluate_doca_pointwise.py` for local eval prompt, `src/evaluate_doca_openai.py` for API eval prompt. All three share identical prompt body; only output format differs.

---

## 2. How to Run

### Data Preparation

```bash
# Requires Databricks access (AAD auth)
python src/prepare_doca.py --output_dir data/doca --train_days 12 --dev_days 2
```

### Training (SFT with DeepSpeed)

```bash
# Default: Qwen3-1.7B, 8 GPU, bs256, micro_bs2, lr 2e-5, 3 epochs, neg_ratio 2.0
bash scripts/sft_doca_pointwise_ds.sh

# Override epochs
NUM_EPOCHS=1 bash scripts/sft_doca_pointwise_ds.sh

# Override model
MODEL_PATH=Qwen/Qwen3-0.6B NUM_EPOCHS=1 bash scripts/sft_doca_pointwise_ds.sh
```

Key training hyperparameters (env var overrides):

| Param | Default | Description |
|-------|---------|-------------|
| `MODEL_PATH` | Qwen/Qwen3-1.7B | Base model |
| `NUM_EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 256 | Global batch size |
| `MICRO_BATCH_SIZE` | 2 | Per-GPU micro batch |
| `LEARNING_RATE` | 2e-5 | Learning rate |
| `CUTOFF_LEN` | 4096 | Max sequence length |
| `NEG_RATIO` | 2.0 | Negative samples per positive per feed |
| `DS_CONFIG` | ds_configs/ds_config_zero2.json | DeepSpeed config |

Training target: model outputs " Yes" or " No" token; loss is cross-entropy on that token only.

### Evaluation — Local Model

```bash
# Multi-GPU eval (splits dev set across GPUs, then merges)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_doca_pointwise.sh <checkpoint_path> --all

# Single-GPU
python src/evaluate_doca_pointwise.py \
    --model_path <checkpoint_path> \
    --eval_jsonl data/doca/dev.jsonl \
    --max_feeds 100
```

Scoring: `log P(Yes) - log P(No)` from model logits (continuous, unbounded).

### Evaluation — OpenAI API Baseline

```bash
AZURE_OPENAI_API_KEY=<key> python src/evaluate_doca_openai.py \
    --eval_jsonl data/doca/dev.jsonl \
    --max_feeds 100 --max_workers 8
```

Scoring: Model outputs "Yes 85" or "No 20"; parsed to [-1, 1] confidence score.
Note: GPT-5.1 is a reasoning model — does not support `logprobs` or `temperature` params.

### Metrics

All metrics are computed **per feed** (only feeds with ≥1 click), then averaged:
- **AUC**: Area under ROC curve (positive vs negative candidate pairs)
- **MRR**: Mean Reciprocal Rank (rank of first clicked candidate)
- **nDCG@5, nDCG@10**: Normalized Discounted Cumulative Gain

---

## 3. Experiments

### Exp 0: GPT-5.1 Zero-Shot Baseline

| Config | Value |
|--------|-------|
| Model | GPT-5.1 (gpt-5.1-2025-11-13) via Azure OpenAI |
| Endpoint | msncompanioneu2.cognitiveservices.azure.com |
| Prompt | Same as SFT training (+ confidence score output format) |
| Feeds evaluated | 100 (dev set, with clicks) |
| API calls | 624 |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|---------|
| 0.5854 | 0.6041 | 0.6519 | 0.6959 |

### Exp 0.5: Qwen3-1.7B Zero-Shot Baseline

| Config | Value |
|--------|-------|
| Model | Qwen/Qwen3-1.7B (pretrained, no SFT) |
| Feeds evaluated | 3,125 (dev set, with clicks) |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|---------|
| 0.5242 | 0.5798 | 0.6039 | 0.6684 |

### Exp 1: SFT Qwen3-1.7B, 1 Epoch

| Config | Value |
|--------|-------|
| Model | Qwen/Qwen3-1.7B |
| Epochs | 1 |
| Batch size | 256 |
| Micro batch | 2 |
| Learning rate | 2e-5 |
| Neg ratio | 2.0 |
| DeepSpeed | ZeRO-2 |
| GPUs | 8x A100 |
| Cutoff length | 4096 |
| Feeds evaluated | 3,125 |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|--------|
| 0.5909 | 0.6311 | 0.6609 | 0.7090 |

**Takeaway**: 1 epoch SFT on 1.7B already outperforms GPT-5.1 zero-shot on all metrics (AUC +0.55pp, MRR +2.7pp, nDCG@10 +1.3pp).

### Exp 2: SFT Qwen3-4B, 3 Epochs

| Config | Value |
|--------|-------|
| Model | Qwen/Qwen3-4B |
| Epochs | 3 |
| Batch size | 256 |
| Micro batch | 1 |
| Learning rate | 1e-5 |
| Neg ratio | 2.0 |
| DeepSpeed | ZeRO-2 |
| GPUs | 8x A100 |
| Cutoff length | 4096 |
| Feeds evaluated | 3,125 |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|---------|
| 0.5747 | 0.6249 | 0.6513 | 0.7027 |

**Takeaway**: 4B with lr=1e-5 and 3 epochs still underperforms 1.7B with lr=2e-5 and 1 epoch on all metrics (AUC -1.6pp, nDCG@10 -0.6pp). More epochs didn't help — lr=1e-5 may be too low for this task.

### Results Summary

| Experiment | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-----------|-------|-----|-----|--------|---------|
| Exp 0 | GPT-5.1 zero-shot | 0.5854 | 0.6041 | 0.6519 | 0.6959 |
| Exp 0.5 | Qwen3-1.7B zero-shot | 0.5242 | 0.5798 | 0.6039 | 0.6684 |
| Exp 1 | Qwen3-1.7B SFT 1ep | **0.5909** | **0.6311** | **0.6609** | **0.7090** |
| Exp 2 | Qwen3-4B SFT 3ep | 0.5747 | 0.6249 | 0.6513 | 0.7027 |

---

## 4. Notes

- Loss 0.693 = random (50%), loss 0.5 ≈ 61% accuracy, loss 0.3 ≈ 74%
- `neg_ratio=2.0` means ~33% Yes, ~67% No in training data
- Impression filter is critical: 67% of raw candidates have sectionIndex=None (never shown to user), must be excluded
- GPT-5.1 reasoning model does not support `logprobs`/`temperature`; use text confidence scoring instead
