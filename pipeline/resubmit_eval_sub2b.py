"""Resubmit failed sub-2B eval-only recovery jobs.

Fixes applied since the original submission (dev branch, forces fresh clone):
  - eval_mind_pointwise.sh CF guard was a string compare: "0.0" != "0" -> True,
    so cf_alpha=0.0 jobs passed --cf_scores_file for a nonexistent file and
    all 8 GPUs crashed with FileNotFoundError. Now numeric (a+0>0).
  - mind_eval component gained a use_subcategory passthrough so the P1.7
    subcat checkpoint evals faithfully.

Run ID naming: display shows the target checkpoint label (P1.x / model).
Usage:
    python resubmit_eval_sub2b.py
"""
import sys
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from run_eval_pipeline import mind_eval_pipeline, VC_ARM_ID, UAI_RESOURCE_ID
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes
from azure.identity import DefaultAzureCredential

ML = MLClient(DefaultAzureCredential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

DATA = "shares/users/wuc/data/MIND_small"

# (label, checkpoint_path, max_history, use_subcategory)
JOBS = [
    ("qwen3 P1.5 evalonly r2",
     "shares/users/wuc/output_dir/sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat/final_checkpoint",
     50, 0),
    ("gemma P1.2 evalonly r2",
     "shares/users/wuc/output_dir/sft_mind_pointwise_small_gemma-2-2b-it_bs256_ep7_neg2.0_hist30_chat/final_checkpoint",
     30, 0),
    ("gemma P1.4 evalonly r2",
     "shares/users/wuc/output_dir/sft_mind_pointwise_small_gemma-2-2b-it_bs256_ep5_neg2.0_hist50_chat/final_checkpoint",
     50, 0),
    ("gemma P1.5 evalonly r2",
     "shares/users/wuc/output_dir/sft_mind_pointwise_small_gemma-2-2b-it_bs256_ep7_neg3.0_hist50_chat/final_checkpoint",
     50, 0),
    ("gemma P1.7 evalonly r2 (subcat)",
     "shares/users/wuc/output_dir/sft_mind_pointwise_small_gemma-2-2b-it_bs256_ep5_neg2.0_hist30_subcat_chat/final_checkpoint",
     30, 1),
]


def submit():
    for label, ckpt, mh, subcat in JOBS:
        job = mind_eval_pipeline(
            msndni_input=Input(type="uri_folder",
                               path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                               mode=InputOutputModes.RW_MOUNT),
            model_path=ckpt,
            data_root=DATA,
            eval_type="pointwise", split="dev",
            use_chat_template=1, use_abstract=0,
            use_subcategory=subcat,
            max_history=mh,
            batch_size=8, num_gpus=8,
            cf_alpha=0.0,
            max_impressions=0,
        )
        job.settings.default_compute = VC_ARM_ID
        job.experiment_name = "sub2b_evalonly_cffix"
        job.display_name = f"eval {label} (cf-fix)"
        created = ML.jobs.create_or_update(job)
        print(f"{label:<28} {created.name}")
        sys.stdout.flush()


if __name__ == "__main__":
    submit()
