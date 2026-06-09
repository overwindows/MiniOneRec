"""
MIND CoT RL Training Pipeline - Azure ML SDK v2
在 A100 GPU 节点上执行 MIND Chain-of-Thought RL 训练

Usage:
    python run_cot_rl_pipeline.py --sft-model shares/users/wuc/models/sft_checkpoint
    python run_cot_rl_pipeline.py --sft-model Qwen/Qwen3-1.7B --cot-style category
    python run_cot_rl_pipeline.py --reward-type mind_cot_ndcg --debug
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
VC_ARM_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

# ============================================================================
# 2b) Managed Identity (required by Singularity policy for datastore mounting)
# ============================================================================
UAI_RESOURCE_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

# ============================================================================
# 3) Resource Configuration - 8-GPU ND96amrs_A100_v4
# ============================================================================
res_cfg = JobResourceConfiguration(
    instance_count=1,
    instance_type="Singularity.ND96amrs_A100_v4",
    properties={
        "singularity": {
            "slaTier": "Standard",
            "priority": "High",
            "enableAzmlInt": False,
        }
    },
)

# ============================================================================
# 4) Load Component from YAML
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
mind_cot_rl_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_cot_rl" / "component.yaml")
)

# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND Chain-of-Thought RL Training Pipeline",
)
def mind_cot_rl_pipeline(
    msndni_input: Input,
    sft_model: str,
    data_root: str = "shares/users/wuc/data/MIND_small",
    output_root: str = "shares/users/wuc/output_dir",
    cot_style: str = "category",
    reward_type: str = "mind_cot_binary",
    max_response_length: int = 256,
    max_prompt_length: int = 4096,
    train_batch_size: int = 32,
    learning_rate: str = "1e-7",
    kl_loss_coef: float = 0.5,
    total_epochs: int = 1,
    num_generations: int = 4,
    max_history: int = 30,
    max_candidates: int = 30,
    debug_mode: str = "false",
    run_eval: int = 1,
    eval_split: str = "dev",
):
    """MIND CoT RL Training Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        sft_model: SFT checkpoint 路径（相对于挂载路径或 HuggingFace 模型 ID）
        data_root: 数据集根目录（相对于挂载路径）
        output_root: 输出目录根路径（相对于挂载路径）
        cot_style: CoT 风格 (standard/category/detailed)
        reward_type: 奖励类型 (mind_cot_binary/mind_cot_ndcg/mind_cot_auc/mind_cot_margin)
        max_response_length: 最大响应长度（CoT 需要更长）
        max_prompt_length: 最大提示长度
        train_batch_size: 训练批次大小
        learning_rate: 学习率
        kl_loss_coef: KL 损失系数
        total_epochs: 训练轮数
        num_generations: 每个样本的生成数量
        max_history: 最大历史记录长度
        max_candidates: 最大候选新闻数量
        debug_mode: 调试模式 (true/false)
        run_eval: 训练后是否运行评估 (1=是, 0=否)
        eval_split: 评估数据集 (dev/test)
    """

    train_node = mind_cot_rl_component(
        msndni=msndni_input,
        sft_model=sft_model,
        data_root=data_root,
        output_root=output_root,
        cot_style=cot_style,
        reward_type=reward_type,
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        kl_loss_coef=kl_loss_coef,
        total_epochs=total_epochs,
        num_generations=num_generations,
        max_history=max_history,
        max_candidates=max_candidates,
        debug_mode=debug_mode,
        run_eval=run_eval,
        eval_split=eval_split,
    )

    # Bind Virtual Cluster and resource configuration
    train_node.compute = VC_ARM_ID
    train_node.resources = res_cfg

    # Required by new Singularity policy: UAI enables the node to authenticate
    # against the datastore for RW_MOUNT access
    train_node.environment_variables = {
        "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
    }


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND CoT RL Training Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="mind_cot_rl_training",
                        help="AML experiment 名称 (default: mind_cot_rl_training)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称，支持空格和特殊字符 (default: auto-generated)")
    parser.add_argument("--sft-model", required=True,
                        help="SFT checkpoint 路径（相对于 datastore 挂载路径或 HuggingFace 模型 ID）")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_small",
                        help="数据集根目录，相对于 datastore 挂载路径 (default: shares/users/wuc/data/MIND_small)")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir",
                        help="输出目录根路径，相对于 datastore 挂载路径 (default: shares/users/wuc/output_dir)")
    parser.add_argument("--cot-style", default="category", choices=["standard", "category", "detailed"],
                        help="CoT 风格 (default: category)")
    parser.add_argument("--reward-type", default="mind_cot_binary",
                        choices=["mind_cot_binary", "mind_cot_ndcg", "mind_cot_auc", "mind_cot_margin"],
                        help="奖励类型 (default: mind_cot_binary)")
    parser.add_argument("--max-response-length", type=int, default=256,
                        help="最大响应长度 (default: 256)")
    parser.add_argument("--max-prompt-length", type=int, default=4096,
                        help="最大提示长度 (default: 4096)")
    parser.add_argument("--train-batch-size", type=int, default=32,
                        help="训练批次大小 (default: 32)")
    parser.add_argument("--learning-rate", default="1e-7",
                        help="学习率 (default: 1e-7)")
    parser.add_argument("--kl-loss-coef", type=float, default=0.5,
                        help="KL 损失系数 (default: 0.5)")
    parser.add_argument("--total-epochs", type=int, default=1,
                        help="训练轮数 (default: 1)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="每个样本的生成数量 (default: 4)")
    parser.add_argument("--max-history", type=int, default=30,
                        help="最大历史记录长度 (default: 30)")
    parser.add_argument("--max-candidates", type=int, default=30,
                        help="最大候选新闻数量 (default: 30)")
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
    job = mind_cot_rl_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        sft_model=args.sft_model,
        data_root=args.data_root,
        output_root=args.output_root,
        cot_style=args.cot_style,
        reward_type=args.reward_type,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        kl_loss_coef=args.kl_loss_coef,
        total_epochs=args.total_epochs,
        num_generations=args.num_generations,
        max_history=args.max_history,
        max_candidates=args.max_candidates,
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
    print(f"Run ID:              {created.name}")
    print(f"Studio URL:          {created.studio_url}")
    print(f"Experiment:          {args.experiment_name}")
    print(f"Display Name:        {args.display_name or created.display_name}")
    print(f"SFT Model:           {args.sft_model}")
    print(f"Data Root:           {args.data_root}")
    print(f"Output Root:         {args.output_root}")
    print(f"CoT Style:           {args.cot_style}")
    print(f"Reward Type:         {args.reward_type}")
    print(f"Max Response Length: {args.max_response_length}")
    print(f"Train Batch Size:    {args.train_batch_size}")
    print(f"Learning Rate:       {args.learning_rate}")
    print(f"KL Loss Coef:        {args.kl_loss_coef}")
    print(f"Total Epochs:        {args.total_epochs}")
    print(f"Run Eval:            {args.run_eval}")
    print(f"Eval Split:          {args.eval_split}")
    print(f"Debug Mode:          {debug_mode}")
