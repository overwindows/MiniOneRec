"""Submit one explicitly registered uniform-feature stage, never an automatic retry."""

import argparse
import hashlib
from pathlib import Path
import re

from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration

from jev_source import prepare_source_component
from run_jev_train import ml_client, VC_ARM_ID, UAI_RESOURCE_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(5), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--bundle-source", action="store_true")
    parser.add_argument("--parity-cache", type=Path, required=True)
    parser.add_argument("--parity-sha256", required=True)
    parser.add_argument("--evaluation-gate", type=Path)
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.parity_sha256):
        parser.error("Parity manifest requires a SHA256")
    if args.stage == 4:
        import json
        gate = json.loads(args.evaluation_gate.read_bytes()) if args.evaluation_gate else {}
        if (gate.get("stage") != "full_training_fitted" or not gate.get("opens_dev_gate")
                or gate.get("training_rows") != 472626
                or gate.get("training_impressions") != 156965
                or gate.get("feature_source_revision") != args.source_revision
                or gate.get("final_refit_holdout_impressions") != 0):
            parser.error("Dev extraction requires the frozen full-data model and open registered gate")
        for field, name in (("head_sha256", "selected-full.head.npz"),
                            ("population_sha256", "population.npz")):
            path = args.evaluation_gate.parent / name
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != gate.get(field):
                parser.error("Frozen full-data evaluation artifact checksum differs: " + name)
    if not VC_ARM_ID.lower().endswith("/virtualclusters/ranking"):
        raise ValueError("Uniform features require the Ranking virtual cluster")
    revision, component, upload = prepare_source_component(
        Path(__file__).parent / "components" / "mind_jev_eval" / "uniform_features.yaml",
        args.source_revision, args.bundle_source,
    )

    @dsl.pipeline(description="Corrected uniform sampling; no backbone fitting or dev tuning")
    def pipeline():
        node = component(
            msndni=Input(type="uri_folder", path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                         mode=InputOutputModes.RW_MOUNT),
            model_path="shares/users/wuc/models/APUS-OpenJev-v1-4B",
            data_root="shares/users/wuc/data/MIND_small", output_root=args.output_root,
            source_revision=revision, full_training_stage=args.stage,
            parity_cache=Input(type="uri_folder", path=str(args.parity_cache.resolve()), mode="download"),
            parity_sha256=args.parity_sha256,
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
    job.experiment_name = "jev_uniform_frozen_features"
    job.display_name = f"JEV uniform frozen features stage{args.stage};one Standard A100"
    job.tags = {
        "source_revision": revision, "jev_experiment": "uniform_frozen_features",
        "feature_stage": str(args.stage), "budget": "one_A100_six_hours_no_retry",
        "parity_sha256": args.parity_sha256, "training_rows": "472626",
    }
    if args.stage == 4:
        job.tags["evaluation_gate_sha256"] = hashlib.sha256(args.evaluation_gate.read_bytes()).hexdigest()
        job.tags["selected_head_sha256"] = gate["head_sha256"]
    try:
        created = ml_client.jobs.create_or_update(job)
    finally:
        if upload is not None:
            upload.cleanup()
    print(f"Run ID: {created.name}", flush=True)
    print(f"Source revision: {revision}", flush=True)


if __name__ == "__main__":
    main()
