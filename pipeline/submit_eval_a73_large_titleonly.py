"""Submit dev eval for the completed A7.3 MIND_large TITLE-ONLY (abs0) checkpoint.

Training job `patient_brick_b829btzn4v` (A7.3L-pw0.5-abs0) COMPLETED with
final_checkpoint on the datastore. This submits a standalone pointwise dev eval
for the same config the model was trained with (title-only, chat template), so
we can compute dev AUC and fold it into the L13+L17 score ensemble.

Usage:
    python submit_eval_a73_large_titleonly.py
"""
import sys
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from run_eval_pipeline import mind_eval_pipeline, VC_ARM_ID

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

DATA = "shares/users/wuc/data/MIND_large"
CKPT = ("shares/users/wuc/output_dir/"
        "sft_mind_pointwise_pairwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_pw0.5_chat/"
        "final_checkpoint")
LABEL = "A7.3L pw0.5 title-only (abs0)"


def main():
    job = mind_eval_pipeline(
        msndni_input=Input(type="uri_folder",
                           path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                           mode=InputOutputModes.RW_MOUNT),
        model_path=CKPT,
        data_root=DATA,
        eval_type="pointwise", split="dev",
        use_chat_template=1, use_abstract=0, use_subcategory=0,
        max_history=30,
        batch_size=8, num_gpus=8,
        cf_alpha=0.0, max_impressions=0,
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = "a73_pairwise_large_eval"
    job.display_name = f"eval {LABEL}"
    created = ML.jobs.create_or_update(job)
    print(f"eval submitted {LABEL:<28} {created.name}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
