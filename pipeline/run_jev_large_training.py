"""Submit the distinct single-GPU, training-only full-impression semantic experiment."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration

from jev_source import prepare_source_component
from run_jev_train import ml_client, VC_ARM_ID, UAI_RESOURCE_ID


CONTRACT_MANIFEST = "4daa3377b4601aca4ae5b72e45ed2931becb8dd648dd23998f17fb3e1342a5f4"
CONTRACT_PROOF = "de60c130dcd7ea853ce8d2b7fbe81b02b83d6b604fbbba99b96bbfa1ec841b25"
RUN_NAME = "jev_large_semantic_20261029_01"
RECOVERY_RUN_NAME = "jev_large_semantic_20261029_03"
RECONSTRUCTED_RUN_NAME = "jev_large_semantic_20261029_04"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--parity-cache", type=Path, required=True)
    parser.add_argument("--parity-sha256", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bundle-source", action="store_true")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--runtime-recovery", action="store_true")
    runtime.add_argument("--reconstructed-runtime", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{64}", args.parity_sha256):
        parser.error("The original exact parity archive requires its registered SHA256")
    for path, expected in ((args.contract / "manifest.json", CONTRACT_MANIFEST),
                           (args.contract / "verification.json", CONTRACT_PROOF),
                           (args.parity_cache / "manifest.json", args.parity_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error("Verified contract or original parity receipt changed")
    proof = json.loads((args.contract / "verification.json").read_bytes())
    if proof["impressions_replayed"] != 2232748 or proof["candidate_exposures_replayed"] != 83507374:
        parser.error("Every large-training impression and candidate must have been independently replayed")
    if not VC_ARM_ID.lower().endswith("/virtualclusters/ranking"):
        raise ValueError("Only the registered Ranking virtual cluster is allowed")
    revision, component, upload = prepare_source_component(
        Path(__file__).parent / "components" / "mind_jev_large_training" / "component.yaml",
        args.source_revision, args.bundle_source,
    )
    worker_seconds = 14400 if args.reconstructed_runtime else (18000 if args.runtime_recovery else 21600)
    @dsl.pipeline(description="JEV-LARGE-SEMANTIC-1; full training impressions; no dev/test contents")
    def pipeline():
        node = component(
            msndni=Input(type="uri_folder", path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                         mode=InputOutputModes.RW_MOUNT),
            contract=Input(type="uri_folder", path=str(args.contract.resolve()), mode="download"),
            parity_cache=Input(type="uri_folder", path=str(args.parity_cache.resolve()), mode="download"),
            source_revision=revision, output_root=args.output_root, parity_sha256=args.parity_sha256,
            max_seconds=worker_seconds,
        )
        node.compute = VC_ARM_ID
        node.resources = JobResourceConfiguration(
            instance_count=1, instance_type="Singularity.ND12am_A100_v4",
            properties={"singularity": {"slaTier": "Standard", "priority": "High", "enableAzmlInt": False}},
        )
        node.limits = CommandJobLimits(timeout=worker_seconds)
        node.environment_variables = {"_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID}

    job = pipeline()
    job.name = (RECONSTRUCTED_RUN_NAME if args.reconstructed_runtime
                else (RECOVERY_RUN_NAME if args.runtime_recovery else RUN_NAME))
    job.settings.default_compute = VC_ARM_ID
    job.settings.force_rerun = True
    job.experiment_name = "jev_large_semantic_training"
    job.display_name = f"JEV full-impression semantic training;one Standard A100;{worker_seconds}s"
    job.tags = {
        "source_revision": revision, "jev_experiment": "large_semantic_training",
        "budget": ("reconstructed_legacy_runtime_one_A100_14400s" if args.reconstructed_runtime
                   else ("qualified_recovery_one_A100_18000s" if args.runtime_recovery
                         else "one_A100_six_hours_no_retry")),
        "contract_manifest_sha256": CONTRACT_MANIFEST,
        "contract_proof_sha256": CONTRACT_PROOF, "parity_sha256": args.parity_sha256,
        "training_impressions": "2232748", "training_exposures": "83507374",
    }
    try:
        created = ml_client.jobs.create_or_update(job)
    finally:
        if upload is not None:
            upload.cleanup()
    print(f"Run ID: {created.name}", flush=True)
    print(f"Source revision: {revision}", flush=True)


if __name__ == "__main__":
    main()
