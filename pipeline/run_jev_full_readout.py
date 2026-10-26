"""Submit one bounded stage of the uncapped official-training readout fit."""

import argparse
from pathlib import Path

from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration

from jev_source import prepare_source_component
from run_jev_train import ml_client, VC_ARM_ID, UAI_RESOURCE_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=range(4))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bundle-source", action="store_true")
    args = parser.parse_args()
    if not VC_ARM_ID.lower().endswith("/virtualclusters/ranking"):
        raise RuntimeError("Full-training readout requires the Ranking virtual cluster")
    revision, component, upload = prepare_source_component(
        Path(__file__).parent / "components" / "mind_jev_eval" / "readout_full_train.yaml",
        args.source_revision, args.bundle_source,
    )

    @dsl.pipeline(description="Use every official training impression;no train-holdout selection")
    def pipeline():
        node = component(
            msndni=Input(
                type="uri_folder", path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                mode=InputOutputModes.RW_MOUNT,
            ),
            model_path="shares/users/wuc/models/APUS-OpenJev-v1-4B",
            data_root="shares/users/wuc/data/MIND_small", output_root=args.output_root,
            source_revision=revision, full_training_stage=args.stage,
            readout_cache_root="shares/users/wuc/output_dir/jev_readout_recovery_20261024/double_train_01",
            readout_holdout_cache="shares/users/wuc/output_dir/jev_readout_recovery_20261023/run_02",
        )
        node.compute = VC_ARM_ID
        node.resources = JobResourceConfiguration(
            instance_count=1, instance_type="Singularity.ND12am_A100_v4",
            properties={"singularity": {"slaTier": "Standard", "priority": "High", "enableAzmlInt": False}},
        )
        node.limits = CommandJobLimits(timeout=21600)
        node.environment_variables = {"_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID}

    job = pipeline()
    job.settings.default_compute = VC_ARM_ID
    job.settings.force_rerun = True
    job.experiment_name = "jev_full_official_training_readout"
    job.display_name = f"JEV full-train frozen-feature stage{args.stage},65536 new-row cap"
    job.tags = {
        "source_revision": revision, "jev_experiment": "full_training_readout",
        "full_training_stage": str(args.stage), "training_scope": "all156965_impressions",
        "training_rows": "469264", "train_holdout_impressions": "0",
        "budget": "one_A100_six_hours_per_stage", "dev_selection": "none",
    }
    try:
        created = ml_client.jobs.create_or_update(job)
    finally:
        if upload is not None:
            upload.cleanup()
    print(f"Run ID: {created.name}", flush=True)
    print(f"Source revision: {revision}", flush=True)
    print(f"Output root: {args.output_root}", flush=True)
    print(f"Full-training stage: {args.stage}", flush=True)


if __name__ == "__main__":
    main()
