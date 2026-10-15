"""
MIND Dual-Encoder Train + Eval — Azure ML SDK v2 submission.

Launches the from-scratch NRMS-style contrastive dual-encoder on the AML
A100 node: clones MiniOneRec dev, trains on MIND, evaluates on dev, and emits
a per-impression normalized score file that folds into ensemble_from_scores.py
alongside the LLM pointwise members.

Usage:
    python run_dual_train_pipeline.py --data-root shares/users/wuc/data/MIND_small --num-epochs 8
    python run_dual_train_pipeline.py --mind-size large --data-root shares/users/wuc/data/MIND_large --batch-size 512 --num-epochs 10 --d-model 128
    python run_dual_train_pipeline.py --debug
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
VC_ARM_ID_MAP = {
    "ranking": (
        "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
        "/resourceGroups/rg-cs-ranking-ml-singularity"
        "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
    ),
    "recall": (
        "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
        "/resourceGroups/rg-cs-recall-ml-singularity"
        "/providers/Microsoft.MachineLearningServices/virtualClusters/recall"
    ),
}
VC_ARM_ID = VC_ARM_ID_MAP["ranking"]

# ============================================================================
# 2b) Managed Identity (required by Singularity policy for datastore mounting)
# ============================================================================
UAI_RESOURCE_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

# ============================================================================
# 3) Resource Configuration — 8-GPU ND96amrs_A100_v4
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
dual_train_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_dual_train" / "component.yaml")
)


# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND Contrastive Dual-Encoder Train + Eval",
)
def dual_train_pipeline(
    msndni_input: Input,
    data_root: str = "shares/users/wuc/data/MIND_small",
    output_root: str = "shares/users/wuc/output_dir",
    mind_size: str = "small",
    d_model: int = 128,
    num_hist: int = 20,
    max_news_len: int = 64,
    batch_size: int = 256,
    num_epochs: int = 8,
    max_history: int = 50,
    lr: float = 0.001,
    use_abstract: int = 0,
    per_pos_neg: int = 2,
    history_as_neg: int = 0,
    infonce_weight: float = 1.0,
    bce_weight: float = 1.0,
    temperature: float = 0.07,
):
    """MIND Dual-Encoder Train + Eval Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        data_root: 数据集根目录（相对于挂载路径）
        output_root: 输出目录根路径（相对于挂载路径）
        mind_size: MIND 数据集大小 (small/large)
        d_model: 嵌入/隐藏维度
        num_hist: 用户历史新闻数量
        max_news_len: 新闻 token 最大长度
        batch_size: 训练批次大小
        num_epochs: 训练轮数
        max_history: 最大历史记录长度
        lr: 学习率
        use_abstract: 是否使用新闻摘要 (1=是, 0=否)
        per_pos_neg: 每个正样本的负样本采样数
        history_as_neg: 是否用历史新闻做难负样本 (1=是, 0=否)
        infonce_weight: InfoNCE loss 权重
        bce_weight: BCE loss 权重
        temperature: InfoNCE 温度
    """
    node = dual_train_component(
        msndni=msndni_input,
        data_root=data_root,
        output_root=output_root,
        mind_size=mind_size,
        d_model=d_model,
        num_hist=num_hist,
        max_news_len=max_news_len,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_history=max_history,
        lr=lr,
        use_abstract=use_abstract,
        per_pos_neg=per_pos_neg,
        history_as_neg=history_as_neg,
        infonce_weight=infonce_weight,
        bce_weight=bce_weight,
        temperature=temperature,
    )
    node.compute = VC_ARM_ID
    node.resources = res_cfg
    node.environment_variables = {
        "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
    }
    node.settings.force_rerun = True


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND Dual-Encoder Train + Eval 提交脚本")
    parser.add_argument("--experiment-name", default="mind_dual_train",
                        help="AML experiment 名称 (default: mind_dual_train)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称 (default: auto-generated)")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_small",
                        help="数据集根目录，相对于 datastore 挂载路径 (default: MIND_small)")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir",
                        help="输出目录根路径，相对于 datastore 挂载路径 (default: output_dir)")
    parser.add_argument("--mind-size", default="small", choices=["small", "large"],
                        help="MIND 数据集大小 (default: small)")
    parser.add_argument("--d-model", type=int, default=128, help="嵌入维度 (default: 128)")
    parser.add_argument("--num-hist", type=int, default=20, help="历史新闻数量 (default: 20)")
    parser.add_argument("--max-news-len", type=int, default=64, help="新闻最大 token 长度 (default: 64)")
    parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小 (default: 256)")
    parser.add_argument("--num-epochs", type=int, default=8, help="训练轮数 (default: 8)")
    parser.add_argument("--max-history", type=int, default=50, help="最大历史记录长度 (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率 (default: 1e-3)")
    parser.add_argument("--use-abstract", type=int, default=0, choices=[0, 1],
                        help="是否使用新闻摘要 (default: 0)")
    parser.add_argument("--per-pos-neg", type=int, default=2, help="每正样本负样本数 (default: 2)")
    parser.add_argument("--history-as-neg", type=int, default=0, choices=[0, 1],
                        help="是否用历史新闻做难负样本 (default: 0)")
    parser.add_argument("--infonce-weight", type=float, default=1.0, help="InfoNCE 权重 (default: 1.0)")
    parser.add_argument("--bce-weight", type=float, default=1.0, help="BCE 权重 (default: 1.0)")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE 温度 (default: 0.07)")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun",
                        help="Datastore 名称 (default: adls_msn_dni_09_rankfun)")
    parser.add_argument("--vm-cluster", default="ranking", choices=["ranking", "recall"],
                        help="Virtual cluster (default: ranking)")
    args = parser.parse_args()

    VC_ARM_ID = VC_ARM_ID_MAP[args.vm_cluster]

    job = dual_train_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        data_root=args.data_root,
        output_root=args.output_root,
        mind_size=args.mind_size,
        d_model=args.d_model,
        num_hist=args.num_hist,
        max_news_len=args.max_news_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_history=args.max_history,
        lr=args.lr,
        use_abstract=args.use_abstract,
        per_pos_neg=args.per_pos_neg,
        history_as_neg=args.history_as_neg,
        infonce_weight=args.infonce_weight,
        bce_weight=args.bce_weight,
        temperature=args.temperature,
    )

    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print("Dual-Encoder Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:            {created.name}")
    print(f"Studio URL:        {created.studio_url}")
    print(f"Experiment:        {args.experiment_name}")
    print(f"Display Name:      {args.display_name or created.display_name}")
    print(f"Virtual Cluster:   {args.vm_cluster}")
    print(f"Mind Size:         {args.mind_size}")
    print(f"Data Root:         {args.data_root}")
    print(f"Output Root:       {args.output_root}")
    print(f"d_model:           {args.d_model}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Num Epochs:        {args.num_epochs}")
    print(f"Max History:       {args.max_history}")
    print(f"Use Abstract:      {args.use_abstract}")
