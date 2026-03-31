"""
MIND Ensemble Evaluation Pipeline - Azure ML SDK v2
Runs multi-wave pointwise ensemble evaluation on MIND dataset.

Usage:
    python run_ensemble_eval_pipeline.py --wave1-models "path/to/model1,path/to/model2"
    python run_ensemble_eval_pipeline.py --wave1-models "m1,m2" --wave2-models "m3,m4"
    python run_ensemble_eval_pipeline.py --wave1-models "m1,m2" --split test --run-convert 1
    python run_ensemble_eval_pipeline.py --wave1-models "m1" --debug
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
ensemble_eval_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_ensemble_eval" / "component.yaml")
)

# ============================================================================
# 5) Build Pipeline
# ============================================================================
@dsl.pipeline(
    description="MIND Pointwise Ensemble Evaluation Pipeline",
)
def mind_ensemble_eval_pipeline(
    msndni_input: Input,
    wave1_models: str,
    wave2_models: str = "",
    data_root: str = "shares/users/wuc/data/MIND_large",
    output_root: str = "shares/users/wuc/output_dir",
    output_subdir: str = "ensemble_results",
    split: str = "dev",
    mind_size: str = "large",
    max_history: int = 30,
    batch_size: int = 8,
    run_convert: int = 0,
    debug_mode: str = "false",
):
    """MIND Pointwise Ensemble Evaluation Pipeline

    Args:
        msndni_input: Datastore 挂载路径
        wave1_models: Wave 1 模型路径，逗号分隔，相对于挂载路径
        wave2_models: Wave 2 模型路径，逗号分隔（可选，留空跳过 Wave 2）
        data_root: MIND 数据集根目录（相对于挂载路径）
        output_root: 输出目录根路径（相对于挂载路径）
        output_subdir: 结果子目录名称
        split: 评估集 (dev/test)
        mind_size: MIND 数据集大小 (small/large)
        max_history: 最大历史记录长度
        batch_size: 每个 GPU 的批次大小
        run_convert: 是否转换为 Leaderboard 格式 (1=是, 0=否)
        debug_mode: 调试模式 (true/false)
    """
    eval_node = ensemble_eval_component(
        msndni=msndni_input,
        wave1_models=wave1_models,
        wave2_models=wave2_models,
        data_root=data_root,
        output_root=output_root,
        output_subdir=output_subdir,
        split=split,
        mind_size=mind_size,
        max_history=max_history,
        batch_size=batch_size,
        run_convert=run_convert,
        debug_mode=debug_mode,
    )

    eval_node.compute = VC_ARM_ID
    eval_node.resources = res_cfg


# ============================================================================
# 6) Parse Args and Submit
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIND Ensemble Eval Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="mind_ensemble_eval",
                        help="AML experiment 名称 (default: mind_ensemble_eval)")
    parser.add_argument("--display-name", default=None,
                        help="Job 显示名称 (default: auto-generated)")
    parser.add_argument("--wave1-models", required=True,
                        help="Wave 1 模型路径，逗号分隔，相对于 datastore 挂载路径")
    parser.add_argument("--wave2-models", default="",
                        help="Wave 2 模型路径，逗号分隔（可选，留空跳过）")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_large",
                        help="数据集根目录，相对于 datastore 挂载路径 (default: shares/users/wuc/data/MIND_large)")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir",
                        help="输出目录根路径，相对于 datastore 挂载路径 (default: shares/users/wuc/output_dir)")
    parser.add_argument("--output-subdir", default="ensemble_results",
                        help="结果子目录名称 (default: ensemble_results)")
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                        help="评估集 (default: dev)")
    parser.add_argument("--mind-size", default="large", choices=["small", "large"],
                        help="MIND 数据集大小 (default: large)")
    parser.add_argument("--max-history", type=int, default=30,
                        help="最大历史记录长度 (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="每个 GPU 的批次大小 (default: 8)")
    parser.add_argument("--run-convert", type=int, default=0, choices=[0, 1],
                        help="是否转换为 Leaderboard 格式, 1=是 0=否 (default: 0)")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun",
                        help="Datastore 名称 (default: adls_msn_dni_09_rankfun)")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：出错继续执行 + 最后 sleep infinity 保持容器运行")
    args = parser.parse_args()

    debug_mode = "true" if args.debug else "false"

    # Create pipeline instance
    job = mind_ensemble_eval_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        wave1_models=args.wave1_models,
        wave2_models=args.wave2_models,
        data_root=args.data_root,
        output_root=args.output_root,
        output_subdir=args.output_subdir,
        split=args.split,
        mind_size=args.mind_size,
        max_history=args.max_history,
        batch_size=args.batch_size,
        run_convert=args.run_convert,
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
    print("Ensemble Eval Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:            {created.name}")
    print(f"Studio URL:        {created.studio_url}")
    print(f"Experiment:        {args.experiment_name}")
    print(f"Display Name:      {args.display_name or created.display_name}")
    print(f"Wave 1 Models:     {args.wave1_models}")
    print(f"Wave 2 Models:     {args.wave2_models or '(none)'}")
    print(f"Data Root:         {args.data_root}")
    print(f"Output Root:       {args.output_root}")
    print(f"Output Subdir:     {args.output_subdir}")
    print(f"Split:             {args.split}")
    print(f"MIND Size:         {args.mind_size}")
    print(f"Max History:       {args.max_history}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Run Convert:       {args.run_convert}")
    print(f"Debug Mode:        {debug_mode}")
