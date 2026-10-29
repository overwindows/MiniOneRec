"""Submit one separately bounded, frozen-head complete official MIND-large dev evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from azure.ai.ml import Input, dsl
from azure.ai.ml.constants import InputOutputModes
from azure.ai.ml.entities import CommandJobLimits, JobResourceConfiguration

from jev_source import prepare_source_component
from run_jev_large_training import CONTRACT_MANIFEST, CONTRACT_PROOF
from run_jev_train import ml_client, VC_ARM_ID, UAI_RESOURCE_ID


RUN_NAME = "jev_large_frozen_dev_20261029_01"
RESULT_SHA = "5c5169a84b3a092762694a32b12b6c927dd99780d660a4db3cde99b0240d2432"
OUTPUT = "shares/users/wuc/output_dir/jev_readout_recovery_20261028/large_frozen_dev_01"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "parity-cache", "training-proof"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("parity-sha256", "training-proof-sha256", "source-revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bundle-source", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch("[0-9a-f]{64}", value)
           for value in (args.parity_sha256, args.training_proof_sha256)):
        parser.error("Immutable parity and independent training proof hashes are required")
    for path, expected in ((args.contract / "manifest.json", CONTRACT_MANIFEST),
                           (args.contract / "verification.json", CONTRACT_PROOF),
                           (args.parity_cache / "manifest.json", args.parity_sha256),
                           (args.training_proof / "verification.json", args.training_proof_sha256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error("A frozen source contract, parity archive or independent training proof changed")
    proof = json.loads((args.training_proof / "verification.json").read_bytes())
    if (proof["result_sha256"] != RESULT_SHA or not proof["opens_dev_gate"]
            or not proof["original16_bit_exact_parity_independently_verified"]
            or not proof["full_training_epoch_source_and_coverage_verified"]
            or proof["selected_recipe"] != "candidate_attention"
            or proof["official_dev_allocations"] != 0
            or proof["parent_status"] != "Completed" or proof["worker_status"] != "Completed"):
        parser.error("Completed independently verified all-training fit is mandatory")
    if not VC_ARM_ID.lower().endswith("/virtualclusters/ranking"):
        raise ValueError("Only the registered Ranking virtual cluster is allowed")
    revision, component, upload = prepare_source_component(
        Path(__file__).parent / "components" / "mind_jev_large_training" / "evaluation.yaml",
        args.source_revision, args.bundle_source,
    )

    @dsl.pipeline(description="Complete official MIND-large dev; frozen final all-training head; no fitting")
    def pipeline():
        node = component(
            msndni=Input(type="uri_folder", path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                         mode=InputOutputModes.RW_MOUNT),
            contract=Input(type="uri_folder", path=str(args.contract.resolve()), mode="download"),
            parity_cache=Input(type="uri_folder", path=str(args.parity_cache.resolve()), mode="download"),
            training_proof=Input(type="uri_folder", path=str(args.training_proof.resolve()), mode="download"),
            training_proof_sha256=args.training_proof_sha256, training_result_sha256=RESULT_SHA,
            source_revision=revision, output_root=OUTPUT, parity_sha256=args.parity_sha256,
            max_seconds=21600,
        )
        node.compute = VC_ARM_ID
        node.resources = JobResourceConfiguration(
            instance_count=1, instance_type="Singularity.ND12am_A100_v4",
            properties={"singularity": {"slaTier": "Standard", "priority": "High", "enableAzmlInt": False}},
        )
        node.limits = CommandJobLimits(timeout=21600)
        node.environment_variables = {"_AZUREML_SINGULARITY_JOB_UAI": UAI_RESOURCE_ID}

    job = pipeline()
    job.name = RUN_NAME
    job.settings.default_compute = VC_ARM_ID
    job.settings.force_rerun = True
    job.experiment_name = "jev_large_frozen_official_dev"
    job.display_name = "Complete MIND-large dev;frozen all-training head;one Standard A100/21600s"
    job.tags = {
        "source_revision": revision, "jev_experiment": "large_frozen_official_dev",
        "budget": "one_A100_six_hours_no_retry;separate_evaluation_not_training_retry",
        "training_result_sha256": RESULT_SHA, "training_proof_sha256": args.training_proof_sha256,
        "parity_sha256": args.parity_sha256, "model_fits": "0", "head_or_recipe_retuning": "false",
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
