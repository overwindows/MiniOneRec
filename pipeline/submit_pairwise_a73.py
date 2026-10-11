"""Submit A7.3 hybrid pointwise+pairwise (Bradley-Terry) training runs.

A7.3 (EXPERIMENT_PLAN.md): add L_BT = -log(sigmoid(score_pos - score_neg)) to
the pointwise CE, ranking the Yes-logit higher within same-impression pos/neg
pairs. Expected +0.2-0.5% AUC over the pointwise baseline.

Follows the L1.3 winning config (Qwen3-1.7B, abstract, ep5, neg2.0, hist30,
chat). Research-first on MIND_small before scaling to MINDlarge.

Usage:
    HF_TOKEN=hf_xxx python submit_pairwise_a73.py
"""
import os
import sys
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes
from run_pipeline import mind_train_pipeline, VC_ARM_ID, UAI_RESOURCE_ID

ML = MLClient(DefaultAzureCredential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# A7.3 sweep on MIND_small: vary pairwise weight + abstract (L1.3-style).
# Baseline (control) pw=0.0 == plain pointwise; pw>0 activates the BT term.
RUNS = [
    # (label, model, data_root, pairwise_weight, use_abstract)
    ("A7.3-pw0.0-abs0", "Qwen/Qwen3-1.7B", "shares/users/wuc/data/MIND_small", 0.0, 0),
    ("A7.3-pw0.5-abs0", "Qwen/Qwen3-1.7B", "shares/users/wuc/data/MIND_small", 0.5, 0),
    ("A7.3-pw0.5-abs1", "Qwen/Qwen3-1.7B", "shares/users/wuc/data/MIND_small", 0.5, 1),
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
        run_eval=0, eval_split="dev",          # eval via standalone pipeline (output name differs)
        early_stopping_patience=3, train_sample=0,
        trainer_script="sft_mind_pointwise_pairwise.sh",
        pairwise_loss_weight=pw,
        hf_token=HF_TOKEN,
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = "a73_pairwise_small"
    job.display_name = f"{label} pw{pw} abs{use_abstract} (small)"
    created = ML.jobs.create_or_update(job)
    print(f"{label:<18} {created.name:<26} pw={pw} abs={use_abstract}")
    sys.stdout.flush()
    return created.name


def main():
    if not HF_TOKEN:
        # not a gated model (Qwen3-1.7B is open) — token optional, continue
        pass
    for label, model, data_root, pw, use_abstract in RUNS:
        submit(label, model, data_root, pw, use_abstract)


if __name__ == "__main__":
    main()
