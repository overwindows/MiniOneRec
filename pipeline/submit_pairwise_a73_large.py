"""Submit A7.3 hybrid pointwise+pairwise (Bradley-Terry) training on MIND_large.

Validated on MIND_small: pw0.5 title-only = 0.6669 vs pw0.0 control = 0.6562
(+1.07 AUC); pw0.5 abstract = 0.6563 (canceled the gain). So the promotable
config is pw=0.5, TITLE-ONLY (abs=0), per the A7.3 pilot read.

Usage:
    HF_TOKEN=hf_xxx python submit_pairwise_a73_large.py
"""
import os
import sys
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes
from run_pipeline import mind_train_pipeline, VC_ARM_ID, UAI_RESOURCE_ID

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# MIND_large A7.3: pw=0.5 TITLE-ONLY (abs=0) — the validated winning config.
RUNS = [
    # (label, model, data_root, pairwise_weight, use_abstract)
    ("A7.3L-pw0.5-abs0", "Qwen/Qwen3-1.7B",
     "shares/users/wuc/data/MIND_large", 0.5, 0),
]


def submit(label, model, data_root, pw, use_abstract):
    job = mind_train_pipeline(
        msndni_input=Input(type="uri_folder",
                           path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                           mode=InputOutputModes.RW_MOUNT),
        model_path=model,
        data_root=data_root,
        batch_size=256, micro_batch_size=4,
        num_epochs=5, neg_ratio=2.0, hard_neg_ratio=0.5,
        max_history=30, cutoff_len=8192,
        use_chat_template=1,
        use_abstract=use_abstract,
        use_subcategory=0,
        pointwise_ratio=1.0, ranking_neg_ratio=4.0,
        run_eval=0, eval_split="dev",
        early_stopping_patience=3, train_sample=0,
        trainer_script="sft_mind_pointwise_pairwise.sh",
        pairwise_loss_weight=pw,
        hf_token=HF_TOKEN,
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = "a73_pairwise_large"
    job.display_name = f"{label} pw{pw} abs{use_abstract} (large)"
    created = ML.jobs.create_or_update(job)
    print(f"{label:<18} {created.name:<26} pw={pw} abs={use_abstract}")
    sys.stdout.flush()
    return created.name


def main():
    for label, model, data_root, pw, use_abstract in RUNS:
        submit(label, model, data_root, pw, use_abstract)


if __name__ == "__main__":
    main()
