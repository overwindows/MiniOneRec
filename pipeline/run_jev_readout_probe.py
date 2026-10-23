"""Submit one bounded one-A100 frozen-backbone recovery probe."""

import argparse
from pathlib import Path

from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration

from jev_source import prepare_source_component
from run_jev_eval import ml_client, VC_ARM_ID, UAI_RESOURCE_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-source", action="store_true")
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    component_path = Path(__file__).parent / "components" / "mind_jev_eval" / "readout_probe.yaml"
    revision, component, upload = prepare_source_component(
        component_path, args.source_revision, args.bundle_source,
    )

    @dsl.pipeline(description="Preserve frozen JEV; select residual readout on train-only holdout")
    def pipeline():
        node = component(
            msndni=Input(
                type="uri_folder",
                path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                mode=InputOutputModes.RW_MOUNT,
            ),
            model_path="shares/users/wuc/models/APUS-OpenJev-v1-4B",
            data_root="shares/users/wuc/data/MIND_small",
            output_root=args.output_root,
            source_revision=revision,
        )
        node.compute = VC_ARM_ID
        node.resources = JobResourceConfiguration(
            instance_count=1, instance_type="Singularity.ND12am_A100_v4",
            properties={"singularity": {
                "slaTier": "Standard", "priority": "High", "enableAzmlInt": False,
            }},
        )
        node.limits = CommandJobLimits(timeout=21600)
        node.environment_variables = {"_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID}

    job = pipeline()
    job.settings.default_compute = VC_ARM_ID
    job.settings.force_rerun = True
    job.experiment_name = "jev_readout_recovery"
    job.display_name = "JEV frozen-backbone residual readout pilot"
    job.tags = {
        "source_revision": revision, "jev_experiment": "frozen_residual_readout",
        "budget": "one_A100_six_hours", "dev_selection": "none",
    }
    try:
        created = ml_client.jobs.create_or_update(job)
    finally:
        if upload is not None:
            upload.cleanup()
    print(f"Run ID: {created.name}", flush=True)
    print(f"Source revision: {revision}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print("Budget: one Standard ranking A100, six-hour worker timeout", flush=True)


if __name__ == "__main__":
    main()
