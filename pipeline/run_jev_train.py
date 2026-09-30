"""
JEV-4B SFT Training Pipeline (with within-model distillation) - Azure ML SDK v2

Submits Campaign-1 of the JEV training campaign: fine-tune APUS-OpenJev-v1-4B
on MIND noul pointwise data with a within-model KL distillation term.

Usage:
    python run_jev_train.py
    python run_jev_train.py --kl-beta 0.1 --num-epochs 3
    python run_jev_train.py --micro-batch-size 2 --neg-ratio 1.0
"""
import argparse
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, load_component
from azure.ai.ml.entities import JobResourceConfiguration
from azure.ai.ml.constants import InputOutputModes

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
    resource_group_name="AMLStudio",
    workspace_name="NewsFeedL2_AML",
)

VC_ARM_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-ranking-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/ranking"
)

UAI_RESOURCE_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
)

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

SCRIPT_DIR = Path(__file__).parent
jev_train_component = load_component(
    source=str(SCRIPT_DIR / "components" / "jev_train" / "component.yaml")
)


@dsl.pipeline(description="JEV-4B SFT Training Pipeline (within-model distillation)")
def jev_train_pipeline(
    msndni_input: Input,
    model_path: str = "shares/users/wuc/models/APUS-OpenJev-v1-4B",
    data_root: str = "shares/users/wuc/data/MIND_large",
    output_root: str = "shares/users/wuc/output_dir",
    batch_size: int = 128,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    max_history: int = 30,
    cutoff_len: int = 4096,
    use_chat_template: int = 0,
    use_abstract: int = 1,
    kl_beta: str = "0.1",
    enable_kl: str = "1",
):
    train_node = jev_train_component(
        msndni=msndni_input,
        model_path=model_path,
        data_root=data_root,
        output_root=output_root,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        max_history=max_history,
        cutoff_len=cutoff_len,
        use_chat_template=use_chat_template,
        use_abstract=use_abstract,
        kl_beta=kl_beta,
        enable_kl=enable_kl,
    )
    train_node.compute = VC_ARM_ID
    train_node.resources = res_cfg
    train_node.environment_variables = {
        "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEV-4B SFT Training Pipeline 提交脚本")
    parser.add_argument("--experiment-name", default="jev_sft_training")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--model-path", default="shares/users/wuc/models/APUS-OpenJev-v1-4B")
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_large")
    parser.add_argument("--output-root", default="shares/users/wuc/output_dir")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--hard-neg-ratio", type=float, default=0.5)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--use-chat-template", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use-abstract", type=int, default=1, choices=[0, 1])
    parser.add_argument("--kl-beta", default="0.1")
    parser.add_argument("--enable-kl", default="1", choices=["0", "1"])
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun")
    args = parser.parse_args()

    job = jev_train_pipeline(
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
        hard_neg_ratio=args.hard_neg_ratio,
        max_history=args.max_history,
        cutoff_len=args.cutoff_len,
        use_chat_template=args.use_chat_template,
        use_abstract=args.use_abstract,
        kl_beta=args.kl_beta,
        enable_kl=args.enable_kl,
    )

    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print("JEV-4B SFT Pipeline submitted successfully!")
    print("=" * 60)
    print(f"Run ID:            {created.name}")
    print(f"Studio URL:        {created.studio_url}")
    print(f"Experiment:        {args.experiment_name}")
    print(f"Display Name:      {args.display_name or created.display_name}")
    print(f"Model Path:        {args.model_path}")
    print(f"Data Root:         {args.data_root}")
    print(f"KL Beta:           {args.kl_beta}")
    print(f"Enable KL:         {args.enable_kl}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Micro Batch Size:  {args.micro_batch_size}")
    print(f"Num Epochs:        {args.num_epochs}")
    print(f"Neg Ratio:         {args.neg_ratio}")
    print(f"Max History:       {args.max_history}")
    print(f"Use Abstract:      {args.use_abstract}")
