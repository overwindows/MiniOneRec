"""Submit the OpenJev MIND evaluation on the GPU (A100) cluster.

Usage:
    python run_jev_eval.py --model-path shares/users/wuc/models/APUS-OpenJev-v1-4B
    python run_jev_eval.py --model-path .../APUS-OpenJev-v1-4B --split dev --max-impressions 200
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

# recall virtual cluster - A100 GPU
VC_ARM_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/rg-cs-recall-ml-singularity"
    "/providers/Microsoft.MachineLearningServices/virtualClusters/recall"
)

# Single A100 (1 GPU is enough for inference; keeps job small + avoid hogging 8x)
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

SCRIPT_DIR = Path(__file__).parent
jev_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_jev_eval" / "component.yaml")
)


@dsl.pipeline(description="MIND OpenJev Decision-Model Evaluation")
def mind_jev_pipeline(
    msndni_input: Input,
    model_path: str,
    data_root: str = "shares/users/wuc/data/MIND_small",
    split: str = "dev",
    use_abstract: int = 0,
    max_history: int = 30,
    max_impressions: int = 0,
    debug_mode: str = "false",
):
    node = jev_component(
        msndni=msndni_input,
        model_path=model_path,
        data_root=data_root,
        split=split,
        use_abstract=use_abstract,
        max_history=max_history,
        max_impressions=max_impressions,
        debug_mode=debug_mode,
    )
    node.compute = VC_ARM_ID
    node.resources = res_cfg
    node.settings.force_rerun = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mind_jev_evaluation")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="shares/users/wuc/data/MIND_small")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--use-abstract", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-impressions", type=int, default=0)
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    job = mind_jev_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        model_path=args.model_path,
        data_root=args.data_root,
        split=args.split,
        use_abstract=args.use_abstract,
        max_history=args.max_history,
        max_impressions=args.max_impressions,
        debug_mode="true" if args.debug else "false",
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)
    print("=" * 60)
    print("OpenJev Evaluation submitted")
    print("=" * 60)
    print(f"Run ID:     {created.name}")
    print(f"Studio URL: {created.studio_url}")
    print(f"Model:      {args.model_path}")
    print(f"Split:      {args.split}")
    print(f"MaxImp:     {args.max_impressions}")
