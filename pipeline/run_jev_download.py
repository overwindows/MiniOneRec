"""Submit the OpenJev model pre-staging job on CPU compute (no A100 hours).

Usage:
    python run_jev_download.py --repo-id apus-ailab/APUS-OpenJev-v1-4B \
        --dest-path shares/users/wuc/models/APUS-OpenJev-v1-4B
"""
import argparse
from pathlib import Path
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, dsl, load_component
from azure.ai.ml.constants import InputOutputModes

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
    resource_group_name="AMLStudio",
    workspace_name="NewsFeedL2_AML",
)

# CPU compute - L2-8CR-64GB (E8a, 64GB RAM) - no GPU needed for download
CPU_COMPUTE = "L2-8CR-64GB"

SCRIPT_DIR = Path(__file__).parent
download_component = load_component(
    source=str(SCRIPT_DIR / "components" / "mind_jev_download" / "component.yaml")
)


@dsl.pipeline(description="MIND OpenJev Download (CPU)")
def jev_download_pipeline(
    msndni_input: Input,
    repo_id: str,
    dest_path: str,
    debug_mode: str = "false",
):
    dl_node = download_component(
        msndni=msndni_input,
        repo_id=repo_id,
        dest_path=dest_path,
        debug_mode=debug_mode,
    )
    dl_node.compute = CPU_COMPUTE
    dl_node.settings.force_rerun = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="mind_jev_download")
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--repo-id", default="apus-ailab/APUS-OpenJev-v1-4B")
    parser.add_argument("--dest-path", default="shares/users/wuc/models/APUS-OpenJev-v1-4B")
    parser.add_argument("--datastore", default="adls_msn_dni_09_rankfun")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    job = jev_download_pipeline(
        msndni_input=Input(
            type="uri_folder",
            path=f"azureml://datastores/{args.datastore}/paths/",
            mode=InputOutputModes.RW_MOUNT,
        ),
        repo_id=args.repo_id,
        dest_path=args.dest_path,
        debug_mode="true" if args.debug else "false",
    )
    job.settings.default_compute = CPU_COMPUTE
    job.experiment_name = args.experiment_name
    if args.display_name:
        job.display_name = args.display_name

    created = ml_client.jobs.create_or_update(job)
    print("=" * 60)
    print("OpenJev Download submitted (CPU compute)")
    print("=" * 60)
    print(f"Run ID:     {created.name}")
    print(f"Studio URL: {created.studio_url}")
    print(f"Repo:       {args.repo_id}")
    print(f"Dest:       {args.dest_path}")
