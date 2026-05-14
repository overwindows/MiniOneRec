"""
MIND Model Evaluation Pipeline - Azure ML SDK v2
在 A100 GPU 节点上克隆 MiniOneRec 仓库并执行 MIND 模型评估

Usage:
    python run_eval_pipeline.py --model-path shares/users/wuc/models/checkpoint
    python run_eval_pipeline.py --model-path shares/users/wuc/models/checkpoint --eval-type ranking
    python run_eval_pipeline.py --model-path shares/users/wuc/models/checkpoint --split test
    python run_eval_pipeline.py --debug
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
mind_eval_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_eval" / "component.yaml")
)

# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND Model Evaluation Pipeline",
)
def mind_eval_pipeline(
    msndni_input: Input,
    model_path: str,
    data_root: str = "shares/users/wuc/data/MIND_small",
    eval_type: str = "pointwise",
    split: str = "dev",
    use_chat_template: int = 1,
    use_abstract: int = 0,
    max_history: int = 30,
    batch_size: int = 8,
    num_gpus: int = 8,
    cf_alpha: float = 0.0,
    debug_mode: str = "false",
):
    """MIND Model Evaluation Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        model_path: 模型checkpoint路径（相对于挂载路径）
        data_root: 数据集根目录（相对于挂载路径）(default: shares/users/wuc/data/MIND_small)
        eval_type: 评估类型 (pointwise/ranking) (default: pointwise)
        split: 评估数据集 (dev/test) (default: dev)
        use_chat_template: 是否使用 chat template (1=是, 0=否) (default: 1)
        use_abstract: 是否使用新闻摘要 (1=是, 0=否) (default: 0)
        max_history: 最大历史记录长度 (default: 30)
        batch_size: 评估batch size (default: 8)
        num_gpus: 使用的GPU数量 (default: 8)
        debug_mode: 调试模式 (true/false) - 出错继续 + sleep infinity (default: false)
    """

    eval_node = mind_eval_component(
        msndni=msndni_input,
        model_path=model_path,
        data_root=data_root,
        eval_type=eval_type,
        split=split,
        use_chat_template=use_chat_template,
        use_abstract=use_abstract,
        max_history=max_history,
        batch_size=batch_size,
        num_gpus=num_gpus,
        cf_alpha=cf_alpha,
        debug_mode=debug_mode,
    )

    # Bind Virtual Cluster and resource configuration
    eval_node.compute = VC_ARM_ID
    eval_node.resources = res_cfg


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND Model Evaluation Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="mind_evaluation",
                        help="AML experiment 名称 (default: mind_evaluation)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称，支持空格和特殊字符 (default: auto-generated)")
    parser.add_argument("--model-path", required=True,
                        help="模型checkpoint路径（相对于 datastore 挂载路径）")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_small",
                        help="数据集根目录，相对于 datastore 挂载路径 (default: shares/users/wuc/data/MIND_small)")
    parser.add_argument("--eval-type", default="pointwise", choices=["pointwise", "ranking"],
                        help="评估类型 (default: pointwise)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                        help="评估数据集 (default: dev)")
    parser.add_argument("--use-chat-template", type=int, default=1, choices=[0, 1],
                        help="是否使用 chat template, 1=是 0=否 (default: 1)")
    parser.add_argument("--use-abstract", type=int, default=0, choices=[0, 1],
                        help="是否使用新闻摘要, 1=是 0=否 (default: 0)")
    parser.add_argument("--max-history", type=int, default=30,
                        help="最大历史记录长度 (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="评估batch size (default: 8)")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="使用的GPU数量 (default: 8)")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun",
                        help="Datastore 名称 (default: adls_msn_dni_09_rankfun)")
    parser.add_argument("--cf-alpha", type=float, default=0.0,
                        help="CF blend weight: 0=no CF; >0 runs CF scoring first then blends LLM+CF (default: 0)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：出错继续执行 + 最后 sleep infinity 保持容器运行")
    args = parser.parse_args()

    debug_mode = "true" if args.debug else "false"

    # Create pipeline instance
    job = mind_eval_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        model_path=args.model_path,
        data_root=args.data_root,
        eval_type=args.eval_type,
        split=args.split,
        use_chat_template=args.use_chat_template,
        use_abstract=args.use_abstract,
        max_history=args.max_history,
        batch_size=args.batch_size,
        num_gpus=args.num_gpus,
        cf_alpha=args.cf_alpha,
        debug_mode=debug_mode,
    )

    # Pipeline-level settings
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    # Submit job
    created = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print("Evaluation Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:            {created.name}")
    print(f"Studio URL:        {created.studio_url}")
    print(f"Experiment:        {args.experiment_name}")
    print(f"Display Name:      {args.display_name or created.display_name}")
    print(f"Model Path:        {args.model_path}")
    print(f"Data Root:         {args.data_root}")
    print(f"Eval Type:         {args.eval_type}")
    print(f"Split:             {args.split}")
    print(f"Use Chat Template: {args.use_chat_template}")
    print(f"Use Abstract:      {args.use_abstract}")
    print(f"Max History:       {args.max_history}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Num GPUs:          {args.num_gpus}")
    print(f"Debug Mode:        {debug_mode}")
