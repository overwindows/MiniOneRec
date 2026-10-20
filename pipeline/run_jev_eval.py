"""Submit the OpenJev MIND evaluation on the GPU (A100) cluster.

Usage:
    python run_jev_eval.py --model-path shares/users/wuc/models/APUS-OpenJev-v1-4B
    python run_jev_eval.py --model-path .../APUS-OpenJev-v1-4B --split dev --max-impressions 200
    python run_jev_eval.py --model-path .../final_checkpoint --virtual-cluster ranking --instance-type Singularity.ND12am_A100_v4 --sla-tier Standard --max-impressions 5000
"""
import argparse
from pathlib import Path
# Use the repo's reliable `cmd /c az` credential (DefaultAzureCredential's bare-`az`
# subprocess spawn is flaky on this Windows box — see azure_auth.py / run_jev_train.py).
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from azure_auth import get_credential
from jev_source import prepare_source_component
from azure.ai.ml import MLClient, Input, dsl, load_component
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration
from azure.ai.ml.constants import InputOutputModes

ml_client = MLClient(
    get_credential(),
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

def make_res_cfg(sla_tier="Premium", instance_type="Singularity.ND96amrs_A100_v4"):
    # instance_count counts nodes; ND96 reserves eight GPUs even though scoring uses GPU 0.
    return JobResourceConfiguration(
        instance_count=1,
        instance_type=instance_type,
        properties={
            "singularity": {
                "slaTier": sla_tier,
                "priority": "High",
                "enableAzmlInt": False,
            }
        },
    )

# User-assigned managed identity required by Singularity policy to authenticate
# against the datastore for RW_MOUNT access (same UAI the training pipeline uses).
UAI_RESOURCE_ID = (
    "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
    "/resourceGroups/AMLStudio"
    "/providers/Microsoft.ManagedIdentity/userAssignedIdentities/rankfun_aml"
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
    prompt_contract: str = "auto",
    source_revision: str = "origin/dev",
    max_prompt_tokens: int = 0,
    output_root: str = "",
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
        prompt_contract=prompt_contract,
        source_revision=source_revision,
        max_prompt_tokens=max_prompt_tokens,
        output_root=output_root,
    )
    node.compute = VC_ARM_ID
    node.resources = make_res_cfg("Premium")
    # Singularity policy: UAI lets the node authenticate to the RW_MOUNT datastore
    node.environment_variables = {
        "_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID,
    }


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
    parser.add_argument("--prompt-contract", default="auto", choices=["auto", "legacy", "native_click", "both"])
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--bundle-source", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=0,
                        help="Fail instead of abstaining if any prompt exceeds this shared budget")
    parser.add_argument("--output-root", default="",
                        help="Separate results directory relative to the datastore mount")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun")
    parser.add_argument("--sla-tier", default="Premium", choices=["Basic", "Standard", "Premium"])
    parser.add_argument("--virtual-cluster", default="recall", choices=["recall", "ranking"])
    parser.add_argument("--instance-type", default="Singularity.ND96amrs_A100_v4",
                        help="ND96 reserves 8 GPUs; use Singularity.ND12am_A100_v4 for a single A100.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_prompt_tokens < 0 or args.max_prompt_tokens > 8192:
        parser.error("--max-prompt-tokens must be 0..8192")
    try:
        args.source_revision, jev_component, source_upload = prepare_source_component(
            SCRIPT_DIR / "components" / "mind_jev_eval" / "component.yaml",
            args.source_revision, args.bundle_source,
        )
    except ValueError as error:
        parser.error(str(error))

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
        prompt_contract=args.prompt_contract,
        source_revision=args.source_revision,
        max_prompt_tokens=args.max_prompt_tokens,
        output_root=args.output_root,
    )
    compute = (
        "/subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895"
        f"/resourceGroups/rg-cs-{args.virtual_cluster}-ml-singularity"
        f"/providers/Microsoft.MachineLearningServices/virtualClusters/{args.virtual_cluster}"
    )
    for n in job.jobs.values():
        if hasattr(n, "resources") and n.name == "node":
            n.compute = compute
            n.resources = make_res_cfg(args.sla_tier, args.instance_type)
            if args.timeout_seconds is not None:
                n.limits = CommandJobLimits(timeout=args.timeout_seconds)
    job.settings.default_compute = compute
    job.settings.force_rerun = True
    job.experiment_name = args.experiment_name
    job.tags = {"source_revision": args.source_revision, "jev_prompt": args.prompt_contract}
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)
    if source_upload is not None:
        source_upload.cleanup()
    print("=" * 60)
    print("OpenJev Evaluation submitted")
    print("=" * 60)
    print(f"Run ID:     {created.name}")
    print(f"Studio URL: {created.studio_url}")
    print(f"Model:      {args.model_path}")
    print(f"Split:      {args.split}")
    print(f"MaxImp:     {args.max_impressions}")
    print(f"Compute:    {args.virtual_cluster} / {args.instance_type} / {args.sla_tier}")
