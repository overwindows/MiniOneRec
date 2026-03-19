# MIND Azure ML Pipelines

Azure ML pipelines for training and evaluating MIND models on A100 GPU nodes.

## 文件结构

```
pipeline/
├── run_pipeline.py                              # 训练pipeline提交脚本
├── run_eval_pipeline.py                         # 评估pipeline提交脚本
└── components/
    ├── mind_train/
    │   ├── component.yaml                       # 训练组件定义
    │   ├── config_mind_train.json               # 训练命令配置
    │   └── pipeline_executor.py                 # Pipeline执行器
    └── mind_eval/
        ├── component.yaml                       # 评估组件定义
        ├── config_mind_eval.json                # 评估命令配置
        └── pipeline_executor.py                 # Pipeline执行器
```

## 训练 Pipeline

### 训练流程

1. `cd /home/aiscuser`
2. `git clone https://github.com/overwindows/MiniOneRec.git`
3. `git checkout dev`
4. `bash scripts/setup_multi_node.sh`（配置多节点环境）
5. `bash scripts/sft_mind_pointwise_ds.sh`（启动 DeepSpeed SFT 训练）

### 用法

#### 使用默认超参提交

```bash
python run_pipeline.py
```

#### 自定义超参

```bash
python run_pipeline.py \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --batch-size 256 \
  --micro-batch-size 4 \
  --num-epochs 3 \
  --neg-ratio 2.0 \
  --max-history 30 \
  --use-chat-template 1
```

#### 调试模式（出错后容器保持运行）

```bash
python run_pipeline.py --debug
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment-name` | `mind_sft_training` | AML experiment 名称 |
| `--model-path` | `Qwen/Qwen3-1.7B` | 预训练模型路径或 HuggingFace 模型 ID |
| `--data-root` | `shares/users/wuc/data/MIND_small` | 数据集根目录（相对于 datastore 挂载路径） |
| `--batch-size` | `256` | 全局批次大小 |
| `--micro-batch-size` | `4` | 每个 GPU 的 micro batch 大小 |
| `--num-epochs` | `3` | 训练轮数 |
| `--neg-ratio` | `2.0` | 负样本比例 |
| `--max-history` | `30` | 最大历史记录长度 |
| `--use-chat-template` | `1` | 是否使用 chat template（`1`=是，`0`=否） |
| `--datastore` | `adls_msn_dni_09_rankfun` | Datastore 名称 |
| `--debug` | `false` | 调试模式：出错继续执行 + 训练结束后 sleep infinity |

---

## 评估 Pipeline

### 评估流程

1. `cd /home/aiscuser`
2. `git clone https://github.com/overwindows/MiniOneRec.git`
3. `git checkout dev`
4. `bash scripts/setup_multi_node.sh`（配置多节点环境）
5. 根据 `eval_type` 选择:
   - `bash scripts/eval_mind_pointwise.sh`（Point-wise 评估）
   - `bash scripts/eval_mind_ranking.sh`（Ranking 评估）

### 用法

#### Point-wise 评估（默认）

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep3_neg2.0_hist30/final_checkpoint
```

#### Ranking 评估

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/checkpoint_name/final_checkpoint \
  --eval-type ranking
```

#### Test 集评估

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/checkpoint_name/final_checkpoint \
  --split test
```

#### 完整参数示例

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/checkpoint_name/final_checkpoint \
  --data-root shares/users/wuc/data/MIND_large \
  --eval-type pointwise \
  --split dev \
  --use-chat-template 1 \
  --max-history 30 \
  --batch-size 8 \
  --num-gpus 8
```

#### 调试模式

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/checkpoint_name/final_checkpoint \
  --debug
```

### 评估参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment-name` | `mind_evaluation` | AML experiment 名称 |
| `--model-path` | **必需** | 模型checkpoint路径（相对于 datastore 挂载路径） |
| `--data-root` | `shares/users/wuc/data/MIND_small` | 数据集根目录（相对于 datastore 挂载路径） |
| `--eval-type` | `pointwise` | 评估类型（`pointwise` 或 `ranking`） |
| `--split` | `dev` | 评估数据集（`dev` 或 `test`） |
| `--use-chat-template` | `1` | 是否使用 chat template（`1`=是，`0`=否） |
| `--max-history` | `30` | 最大历史记录长度 |
| `--batch-size` | `8` | 评估 batch size |
| `--num-gpus` | `8` | 使用的GPU数量（用于并行评估） |
| `--datastore` | `adls_msn_dni_09_rankfun` | Datastore 名称 |
| `--debug` | `false` | 调试模式：出错继续执行 + 评估结束后 sleep infinity |

---

## Windows PowerShell Setup

The pipeline scripts are pure Python (no bash required) and run directly in PowerShell.

### 1. Create and activate environment

**Option A: conda (recommended)**
```powershell
conda create -n MiniOneRec-pipeline python=3.11 -y
conda activate MiniOneRec-pipeline
```

To deactivate when done:
```powershell
conda deactivate
```

**Option B: venv**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

To deactivate when done:
```powershell
deactivate
```

> If you see a script execution error, run this once to allow local scripts:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Install dependencies

```powershell
pip install -r pipeline/requirements.txt
```

### 3. Authenticate with Azure

Install the Azure CLI if you haven't already:
```powershell
winget install Microsoft.AzureCLI
```

Then log in:
```powershell
az login
```

`DefaultAzureCredential` in the scripts will automatically pick up your `az login` session.

### 4. Run pipeline scripts

```powershell
# Submit training pipeline (default params)
python pipeline/run_pipeline.py

# Submit with custom params (use backtick ` for line continuation in PowerShell)
python pipeline/run_pipeline.py `
  --model-path Qwen/Qwen3-1.7B `
  --batch-size 256 `
  --num-epochs 3

# Submit eval pipeline
python pipeline/run_eval_pipeline.py `
  --model-path shares/users/wuc/models/my_checkpoint/final_checkpoint

# Debug mode
python pipeline/run_pipeline.py --debug
```

---

## 环境配置

- **AML Environment**: `azureml:torch251-cuda124-deepspeed-flashattn-training:5`
- **Instance Type**: `Singularity.ND96amrs_A100_v4`（8x A100 80GB）
- **Virtual Cluster**: `recall`
- **Datastore**: `adls_msn_dni_09_rankfun` (RW_MOUNT 模式)

## 注意事项

1. **模型路径**: 评估时 `--model-path` 必须是相对于 datastore 挂载点的路径
2. **数据路径**: 数据集路径从 datastore 自动挂载，使用 RW_MOUNT 模式
3. **并行评估**: 默认使用 8 个 GPU 并行评估以加速
4. **调试模式**: 开启后，即使命令失败也会继续执行，最后保持容器运行方便调试

## 示例工作流

### 1. 训练模型

```bash
python run_pipeline.py \
  --model-path Qwen/Qwen3-1.7B \
  --data-root shares/users/wuc/data/MIND_small \
  --batch-size 256 \
  --num-epochs 3 \
  --experiment-name my_training_exp
```

### 2. 评估模型

训练完成后，使用保存的 checkpoint 进行评估：

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep3_neg2.0_hist30/final_checkpoint \
  --eval-type pointwise \
  --split dev \
  --experiment-name my_eval_exp
```

### 3. 测试集评估

```bash
python run_eval_pipeline.py \
  --model-path shares/users/wuc/models/best_checkpoint/final_checkpoint \
  --split test \
  --experiment-name final_test_eval
```
