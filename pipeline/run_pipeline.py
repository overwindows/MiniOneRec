"""
MIND SFT Training Pipeline - Azure ML SDK v2
在 A100 GPU 节点上克隆 MiniOneRec 仓库并执行 MIND SFT pointwise 训练

Usage:
    python run_pipeline.py
    python run_pipeline.py --model-path Qwen/Qwen3-1.7B --num-epochs 5
    python run_pipeline.py --batch-size 512 --micro-batch-size 8 --neg-ratio 3.0
    python run_pipeline.py --debug
"""
import argparse
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, load_component
from azure.ai.ml.entities import JobResourceConfiguration
from azure.ai.ml.constants import InputOutputModes

# ============================================================================
# 1) Connect to Workspace
# ============================================================================
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
    resource_group_name="AMLStudio",
    workspace_name="NewsFeedL2_AML",
)

# ============================================================================
# 2) Virtual Cluster Full ARM ID
# ============================================================================
# ranking
# VC_ARM_ID = (
#     "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
#     "/resourceGroups/rg-cs-ranking-ml-singularity"
#     "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
# )

# recall
VC_ARM_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-recall-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/recall"
)

# ============================================================================
# 3) Resource Configuration - 8-GPU ND96amrs_A100_v4
# ============================================================================
res_cfg = JobResourceConfiguration(
    instance_count=1,
    instance_type="Singularity.ND96amrs_A100_v4",
    properties={
        "singularity": {
            "slaTier": "Premium",
            "priority": "High",
            "enableAzmlInt": False,
        }
    },
)

# ============================================================================
# 4) Load Component from YAML
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
mind_train_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_train" / "component.yaml")
)

# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND SFT Pointwise Training Pipeline",
)
def mind_train_pipeline(
    msndni_input: Input,
    model_path: str = "Qwen/Qwen3-1.7B",
    data_root: str = "shares/users/wuc/data/MIND_small",
    output_root: str = "shares/users/wuc/output_dir",
    batch_size: int = 256,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    neg_ratio: float = 2.0,
    max_history: int = 30,
    use_chat_template: int = 1,
    debug_mode: str = "false",
    run_eval: int = 1,
    eval_split: str = "dev",
):
    """MIND SFT Training Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        model_path: 预训练模型路径或 HuggingFace 模型 ID (default: Qwen/Qwen3-1.7B)
        data_root: 数据集根目录（相对于挂载路径）(default: shares/users/wuc/data/MIND_small)
        output_root: 输出目录根路径（相对于挂载路径）(default: shares/users/wuc/output_dir)
        batch_size: 全局批次大小 (default: 256)
        micro_batch_size: 每个 GPU 的 micro batch 大小 (default: 4)
        num_epochs: 训练轮数 (default: 3)
        neg_ratio: 负样本比例 (default: 2.0)
        max_history: 最大历史记录长度 (default: 30)
        use_chat_template: 是否使用 chat template (1=是, 0=否) (default: 1)
        debug_mode: 调试模式 (true/false) - 出错继续 + sleep infinity (default: false)
        run_eval: 训练后是否运行评估 (1=是, 0=否) (default: 1)
        eval_split: 评估数据集 (dev/test) (default: dev)
    """

    train_node = mind_train_component(
        msndni=msndni_input,
        model_path=model_path,
        data_root=data_root,
        output_root=output_root,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        neg_ratio=neg_ratio,
        max_history=max_history,
        use_chat_template=use_chat_template,
        debug_mode=debug_mode,
        run_eval=run_eval,
        eval_split=eval_split,
    )

    # Bind Virtual Cluster and resource configuration
    train_node.compute = VC_ARM_ID
    train_node.resources = res_cfg


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND SFT Training Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="mind_sft_training",
                        help="AML experiment 名称 (default: mind_sft_training)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称，支持空格和特殊字符 (default: auto-generated)")
    parser.add_argument("--model-path", default="Qwen/Qwen3-1.7B",
                        help="预训练模型路径或 HuggingFace 模型 ID (default: Qwen/Qwen3-1.7B)")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_small",
                        help="数据集根目录，相对于 datastore 挂载路径 (default: shares/users/wuc/data/MIND_small)")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir",
                        help="输出目录根路径，相对于 datastore 挂载路径 (default: shares/users/wuc/output_dir)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="全局批次大小 (default: 256)")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="每个 GPU 的 micro batch 大小 (default: 4)")
    parser.add_argument("--num-epochs", type=int, default=3,
                        help="训练轮数 (default: 3)")
    parser.add_argument("--neg-ratio", type=float, default=2.0,
                        help="负样本比例 (default: 2.0)")
    parser.add_argument("--max-history", type=int, default=30,
                        help="最大历史记录长度 (default: 30)")
    parser.add_argument("--use-chat-template", type=int, default=1, choices=[0, 1],
                        help="是否使用 chat template, 1=是 0=否 (default: 1)")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun",
                        help="Datastore 名称 (default: adls_msn_dni_09_rankfun)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：出错继续执行 + 最后 sleep infinity 保持容器运行")
    parser.add_argument("--run-eval", type=int, default=1, choices=[0, 1],
                        help="训练后是否运行评估, 1=是 0=否 (default: 1)")
    parser.add_argument("--eval-split", default="dev", choices=["dev", "test"],
                        help="评估数据集 (default: dev)")
    args = parser.parse_args()

    debug_mode = "true" if args.debug else "false"

    # Create pipeline instance
    job = mind_train_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        model_path=args.model_path,
        data_root=args.data_root,
        output_root=args.output_root,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_epochs=args.num_epochs,
        neg_ratio=args.neg_ratio,
        max_history=args.max_history,
        use_chat_template=args.use_chat_template,
        debug_mode=debug_mode,
        run_eval=args.run_eval,
        eval_split=args.eval_split,
    )

    # Pipeline-level settings
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    # Submit job
    created = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print("Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:            {created.name}")
    print(f"Studio URL:        {created.studio_url}")
    print(f"Experiment:        {args.experiment_name}")
    print(f"Display Name:      {args.display_name or created.display_name}")
    print(f"Model Path:        {args.model_path}")
    print(f"Data Root:         {args.data_root}")
    print(f"Output Root:       {args.output_root}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Micro Batch Size:  {args.micro_batch_size}")
    print(f"Num Epochs:        {args.num_epochs}")
    print(f"Neg Ratio:         {args.neg_ratio}")
    print(f"Max History:       {args.max_history}")
    print(f"Use Chat Template: {args.use_chat_template}")
    print(f"Run Eval:          {args.run_eval}")
    print(f"Eval Split:        {args.eval_split}")
    print(f"Debug Mode:        {debug_mode}")
