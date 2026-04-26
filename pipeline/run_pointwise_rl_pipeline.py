"""
MIND Pointwise RL Training Pipeline - Azure ML SDK v2
在 A100 GPU 节点上对 Pointwise SFT 模型执行 Yes/No RL 后训练

Usage:
    python run_pointwise_rl_pipeline.py --sft-model shares/users/wuc/output_dir/sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat/final_checkpoint
    python run_pointwise_rl_pipeline.py --sft-model ... --reward-type pointwise_asymmetric --use-abstract 1
    python run_pointwise_rl_pipeline.py --sft-model ... --debug
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
mind_pointwise_rl_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_pointwise_rl" / "component.yaml")
)

# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND Pointwise RL Training Pipeline",
)
def mind_pointwise_rl_pipeline(
    msndni_input: Input,
    sft_model: str,
    data_root: str = "shares/users/wuc/data/MIND_large",
    output_root: str = "shares/users/wuc/output_dir",
    reward_type: str = "pointwise_asymmetric",
    max_response_length: int = 8,
    max_prompt_length: int = 4096,
    train_batch_size: int = 128,
    learning_rate: str = "1e-7",
    kl_loss_coef: float = 0.1,
    total_epochs: int = 1,
    num_generations: int = 8,
    max_history: int = 30,
    neg_ratio: float = 2.0,
    use_abstract: int = 1,
    use_chat_template: int = 1,
    regenerate_data: int = 0,
    debug_mode: str = "false",
    run_eval: int = 1,
    eval_split: str = "dev",
):
    """MIND Pointwise RL Training Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        sft_model: SFT checkpoint 路径（相对于 datastore 挂载路径）
        data_root: 数据集根目录（相对于挂载路径）
        output_root: 输出目录根路径（相对于挂载路径）
        reward_type: 奖励类型 (pointwise_asymmetric/pointwise_binary/pointwise_weighted/pointwise_margin)
        max_response_length: 最大响应长度，Yes/No 设 8 即可
        max_prompt_length: 最大提示长度
        train_batch_size: 训练批次大小
        learning_rate: 学习率
        kl_loss_coef: KL 损失系数
        total_epochs: 训练轮数
        num_generations: 每个样本的 GRPO 生成数量
        max_history: 最大历史记录长度（必须与 SFT 训练一致）
        neg_ratio: 负样本比例（必须与 SFT 训练一致）
        use_abstract: 是否使用新闻摘要（必须与 SFT 训练一致）
        use_chat_template: 是否使用 chat template（必须与 SFT 训练一致）
        regenerate_data: 是否强制重新生成 parquet 数据
        debug_mode: 调试模式 (true/false)
        run_eval: 训练后是否运行评估
        eval_split: 评估数据集 (dev/test)
    """
    train_node = mind_pointwise_rl_component(
        msndni=msndni_input,
        sft_model=sft_model,
        data_root=data_root,
        output_root=output_root,
        reward_type=reward_type,
        max_response_length=max_response_length,
        max_prompt_length=max_prompt_length,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        kl_loss_coef=kl_loss_coef,
        total_epochs=total_epochs,
        num_generations=num_generations,
        max_history=max_history,
        neg_ratio=neg_ratio,
        use_abstract=use_abstract,
        use_chat_template=use_chat_template,
        regenerate_data=regenerate_data,
        debug_mode=debug_mode,
        run_eval=run_eval,
        eval_split=eval_split,
    )

    train_node.compute = VC_ARM_ID
    train_node.resources = res_cfg


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND Pointwise RL Training Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="mind_pointwise_rl",
                        help="AML experiment 名称 (default: mind_pointwise_rl)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称 (default: auto-generated)")
    parser.add_argument("--sft-model", required=True,
                        help="SFT checkpoint 路径（相对于 datastore 挂载路径）")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_large",
                        help="数据集根目录 (default: shares/users/wuc/data/MIND_large)")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir",
                        help="输出目录根路径 (default: shares/users/wuc/output_dir)")
    parser.add_argument("--reward-type", default="pointwise_asymmetric",
                        choices=["pointwise_asymmetric", "pointwise_binary",
                                 "pointwise_weighted", "pointwise_margin"],
                        help="奖励类型 (default: pointwise_asymmetric)")
    parser.add_argument("--max-response-length", type=int, default=8,
                        help="最大响应长度 (default: 8)")
    parser.add_argument("--max-prompt-length", type=int, default=4096,
                        help="最大提示长度 (default: 4096)")
    parser.add_argument("--train-batch-size", type=int, default=128,
                        help="训练批次大小 (default: 128)")
    parser.add_argument("--learning-rate", default="1e-7",
                        help="学习率 (default: 1e-7)")
    parser.add_argument("--kl-loss-coef", type=float, default=0.1,
                        help="KL 损失系数 (default: 0.1)")
    parser.add_argument("--total-epochs", type=int, default=1,
                        help="训练轮数 (default: 1)")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="每个样本的 GRPO 生成数量 (default: 8)")
    parser.add_argument("--max-history", type=int, default=30,
                        help="最大历史记录长度 (default: 30)")
    parser.add_argument("--neg-ratio", type=float, default=2.0,
                        help="负样本比例 (default: 2.0)")
    parser.add_argument("--use-abstract", type=int, default=1, choices=[0, 1],
                        help="是否使用新闻摘要 (default: 1)")
    parser.add_argument("--use-chat-template", type=int, default=1, choices=[0, 1],
                        help="是否使用 chat template (default: 1)")
    parser.add_argument("--regenerate-data", type=int, default=0, choices=[0, 1],
                        help="是否强制重新生成 parquet 数据 (default: 0)")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun",
                        help="Datastore 名称 (default: adls_msn_dni_09_rankfun)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：出错继续执行 + sleep infinity 保持容器运行")
    parser.add_argument("--run-eval", type=int, default=1, choices=[0, 1],
                        help="训练后是否运行评估 (default: 1)")
    parser.add_argument("--eval-split", default="dev", choices=["dev", "test"],
                        help="评估数据集 (default: dev)")
    args = parser.parse_args()

    debug_mode = "true" if args.debug else "false"

    job = mind_pointwise_rl_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        sft_model=args.sft_model,
        data_root=args.data_root,
        output_root=args.output_root,
        reward_type=args.reward_type,
        max_response_length=args.max_response_length,
        max_prompt_length=args.max_prompt_length,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        kl_loss_coef=args.kl_loss_coef,
        total_epochs=args.total_epochs,
        num_generations=args.num_generations,
        max_history=args.max_history,
        neg_ratio=args.neg_ratio,
        use_abstract=args.use_abstract,
        use_chat_template=args.use_chat_template,
        regenerate_data=args.regenerate_data,
        debug_mode=debug_mode,
        run_eval=args.run_eval,
        eval_split=args.eval_split,
    )

    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print("Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:              {created.name}")
    print(f"Studio URL:          {created.studio_url}")
    print(f"Experiment:          {args.experiment_name}")
    print(f"Display Name:        {args.display_name or created.display_name}")
    print(f"SFT Model:           {args.sft_model}")
    print(f"Reward Type:         {args.reward_type}")
    print(f"Use Abstract:        {args.use_abstract}")
    print(f"Use Chat Template:   {args.use_chat_template}")
    print(f"Neg Ratio:           {args.neg_ratio}")
    print(f"Learning Rate:       {args.learning_rate}")
    print(f"KL Loss Coef:        {args.kl_loss_coef}")
    print(f"Train Batch Size:    {args.train_batch_size}")
    print(f"Num Generations:     {args.num_generations}")
    print(f"Total Epochs:        {args.total_epochs}")
    print(f"Run Eval:            {args.run_eval}")
    print(f"Eval Split:          {args.eval_split}")
    print(f"Debug Mode:          {debug_mode}")
