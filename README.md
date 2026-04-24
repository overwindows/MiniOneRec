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

Download from [cosmos09 MSN.DnI](https://www.cosmos09.osdinfra.net/cosmos/MSN.DnI/shares/users/zxy/doca/data/260423/doca/) and put files into `data/doca_v8/`:
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

- **Table**: `mai_ws_discover.analytics.ods_doca_feed_grounded_v8_partitioned`
- **Platform**: Databricks (Azure), host `adb-3355567219430035.15.azuredatabricks.net`
- **Date range**: 20260330 — 20260420 (22 days)
- **Split**: Train = first 20 days (0330–0418), Dev = last 2 days (0419–0420)
- **Prep script**: `python src/prepare_doca.py --output_dir data/doca_v8 --train_days 20 --dev_days 2`

### Statistics

| Split | Feeds | Candidates | Clicks | CTR |
|-------|------:|----------:|-------:|----:|
| Train | 182,948 | 715,070 | 50,275 | 7.03% |
| Dev   |  18,622 |  68,499 |  5,174 | 7.55% |

- Only **impressed** candidates included (sectionIndex != None), filtering out non-shown cards
- Average ~3.9 candidates per feed (train), ~3.7 per feed (dev)

### Data Fields (per feed JSONL row)

| Field | Description |
|-------|-------------|
| `feed_id`, `user_id`, `ref_ts`, `bizdate` | Feed metadata |
| `candidates` | Array of {itemid, title, summary, is_clicked} — only impressed cards |
| `interests` | User interests with name, strength, domain, sources, intent, classification, status, keywords, rationale |
| `negative_interests` | Disliked topics with name, keywords, sources, rationale |
| `user_flight_ids` | User flight IDs (for debugging / segmentation) |
| `interactions` | User interactions: {clicks, thumbsUp, thumbsDown} extracted from interactions_90d |
| `conversation` | Recent chat history grouped by conversation_id, with [user]/[assistant] roles |
| `shown_10d` | Recently shown articles {title, event_time} |

### Prompt Design

Each candidate is scored independently. The prompt includes:

1. **System prompt**: Role description + 9 ranking signals (interest match, recency, quality, click-history relevance, interaction affinity, negative-interest match, novelty, short-term relevance, click likelihood) + output format
2. **User prompt**: 6 sections — (1) User interests, (2) Dislikes, (3) Recent conversations (grouped by conversation_id with [user]/[assistant] roles, [CURATED] tags), (4) User interactions (thumbs-up/thumbs-down/clicks from interactions_90d), (5) Recently shown articles, (6) Candidate article → "Will this user click on this article? Answer:"

See `data.py` (`DOCAPointwiseSFTDataset`) for training prompt, `src/evaluate_doca_pointwise.py` for local eval prompt, `src/evaluate_doca_openai.py` for API eval prompt. All three share identical prompt body; only output format differs.

---

## 2. How to Run

### Data Preparation

```bash
# Requires Databricks access (AAD auth)
python src/prepare_doca.py --output_dir data/doca_v8 --train_days 20 --dev_days 2
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
| `NUM_EPOCHS` | 1 | Training epochs |
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
    --eval_jsonl data/doca_v8/dev.jsonl \
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

## 3. v7 Data Results

> Data: `ods_doca_feed_grounded_v7_partitioned`, 14 days (20260407–20260420), Train 12d / Dev 2d
> Train: 114,156 feeds, 446,827 candidates, 30,878 clicks (CTR 6.91%)
> Dev: 18,622 feeds, 68,499 candidates, 5,174 clicks (CTR 7.55%)
> Prompt: 7 ranking rules, flat conversation history

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

### Exp 2.5: SFT Qwen3-4B, 3 Epochs (checkpoint-1024)

| Config | Value |
|--------|-------|
| Model | Qwen/Qwen3-4B |
| Epochs | 3 (eval at checkpoint-1024) |
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
|-----|-----|--------|----------|
| 0.5928 | 0.6365 | 0.6641 | 0.7116 |

**Takeaway**: 4B checkpoint-1024 (early in training) outperforms both 1.7B SFT 1ep and GPT-5.1 zero-shot on all metrics. Later checkpoints (3 full epochs) degrade — overfitting confirmed. Early stopping or 1 epoch is preferred.

### Results Summary (v7)

| Experiment | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-----------|-------|-----|-----|--------|----------|
| Exp 0 | GPT-5.1 zero-shot | 0.5854 | 0.6041 | 0.6519 | 0.6959 |
| Exp 0.5 | Qwen3-1.7B zero-shot | 0.5242 | 0.5798 | 0.6039 | 0.6684 |
| Exp 1 | Qwen3-1.7B SFT 1ep | 0.5909 | 0.6311 | 0.6609 | 0.7090 |
| Exp 2 | Qwen3-4B SFT 3ep | 0.5747 | 0.6249 | 0.6513 | 0.7027 |
| Exp 2.5 | Qwen3-4B SFT ckpt-1024 | **0.5928** | **0.6365** | **0.6641** | **0.7116** |

---

## 4. v8 Data Results

> Data: `ods_doca_feed_grounded_v8_partitioned`, 22 days (20260330–20260420), Train 20d / Dev 2d
> Train: 182,948 feeds, 715,070 candidates, 50,275 clicks (CTR 7.03%)
> Dev: 18,622 feeds, 68,499 candidates, 5,174 clicks (CTR 7.55%)
> Prompt: 9 ranking signals, grouped conversations, interactions (thumbs-up/down/clicks)

### Exp 3: GPT-5.1 Zero-Shot (v8 prompt)

| Config | Value |
|--------|-------|
| Model | GPT-5.1 (gpt-5.1-2025-11-13) via Azure OpenAI |
| Endpoint | msncompanioneu2.cognitiveservices.azure.com |
| Prompt | v8: 9 signals, grouped conversations, interactions |
| Feeds evaluated | 100 (dev set, with clicks) |
| API calls | 585 |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|----------|
| 0.5681 | 0.6257 | 0.6435 | 0.7028 |

**Takeaway**: v8 prompt with 9 signals, interactions, and grouped conversations. MRR +2.2pp and nDCG@10 +0.7pp vs v7 Exp 0 (0.6041→0.6257, 0.6959→0.7028), but AUC -1.7pp (0.5854→0.5681). The richer context helps ranking quality (MRR/nDCG) more than discrimination (AUC).

### Exp 3.5: Qwen3-1.7B Zero-Shot (v8 prompt)

| Config | Value |
|--------|-------|
| Model | Qwen/Qwen3-1.7B (pretrained, no SFT) |
| Prompt | v8: 9 signals, grouped conversations, interactions |
| Feeds evaluated | 3,125 (dev set, with clicks) |

**Results:**

| AUC | MRR | nDCG@5 | nDCG@10 |
|-----|-----|--------|----------|
| 0.5226 | 0.5830 | 0.6082 | 0.6694 |

**Takeaway**: v8 prompt on Qwen3-1.7B zero-shot. Slightly worse than v7 Exp 0.5 (AUC 0.5242→0.5226, nDCG@10 0.6684→0.6694). The richer prompt doesn't help a small pretrained model without SFT.

### Results Summary (v8)

| Experiment | Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-----------|-------|-----|-----|--------|----------|
| Exp 3 | GPT-5.1 zero-shot (v8) | 0.5681 | 0.6257 | 0.6435 | 0.7028 |
| Exp 3.5 | Qwen3-1.7B zero-shot (v8) | 0.5226 | 0.5830 | 0.6082 | 0.6694 |

---

## 5. Notes

- Loss 0.693 = random (50%), loss 0.5 ≈ 61% accuracy, loss 0.3 ≈ 74%
- `neg_ratio=2.0` means ~33% Yes, ~67% No in training data
- Impression filter is critical: 67% of raw candidates have sectionIndex=None (never shown to user), must be excluded
- GPT-5.1 reasoning model does not support `logprobs`/`temperature`; use text confidence scoring instead
