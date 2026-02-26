# Train MiniOneRec Pipeline

Azure ML pipeline，在 A100 GPU 节点上对 MiniOneRec 模型进行 SFT pointwise 训练。

## 文件结构

```
Train_Minionerec_Pipeline/
├── run_pipeline.py                                  # 提交脚本
└── components/
    └── minionerec_train/
        ├── component.yaml                           # AML 组件定义
        └── run_minionerec_train.sh                  # 训练入口脚本
```

## 训练流程

1. `cd /home/aiscuser`
2. `git clone https://github.com/overwindows/MiniOneRec.git`
3. `git checkout dev`
4. `bash scripts/setup_multi_node.sh`（配置多节点环境）
5. `bash scripts/sft_mind_pointwise_ds.sh`（启动 DeepSpeed SFT 训练）

数据目录从 `adls_msn_dni_09_rankfun` datastore 以 RW_MOUNT 模式挂载。

## 用法

### 使用默认超参提交

```bash
python run_pipeline.py
```

### 自定义超参

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

### 调试模式（出错后容器保持运行）

```bash
python run_pipeline.py --debug
```

### 指定 experiment 名称

```bash
python run_pipeline.py --experiment-name my_minionerec_exp
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--experiment-name` | `minionerec_sft_training` | AML experiment 名称 |
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

## 环境

- **AML Environment**: `azureml:torch251-cuda124-deepspeed-flashattn-training:5`
- **Instance Type**: `Singularity.ND96amrs_A100_v4`（8x A100 80GB）
- **Virtual Cluster**: `ranking`
