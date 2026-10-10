"""Resubmit failed sub-2B P-series sweep runs.

Fixes applied since the original submission:
  - gemma2-2b: google/gemma-2-2b-it is a GATED HF repo -> all 7 failed with
    GatedRepoError 401. Now pass HF_TOKEN so the gated model resolves.
  - qwen3-1p7b: 5 runs died to a transient pyarrow/NumPy env mismatch on a
    node -> retry (fresh node avoids the bad env). P1.0 was still running so
    it is retried here too (idempotent rerun).

Usage:
    python resubmit_failed_sub2b.py
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

DATA = "shares/users/wuc/data/MIND_small"

# HF token for gated gemma-2-2b-it access. MUST come from the environment;
# never commit a token. Set it before running:
#     HF_TOKEN=hf_xxx python resubmit_failed_sub2b.py
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODELS = [
    ("gemma2-2b", "google/gemma-2-2b-it", ["P1.0", "P1.1", "P1.2", "P1.3", "P1.4", "P1.5", "P1.7"]),
]

# (exp, display, **overrides) -- must match original CONFIGS order/labels
CONFIGS = {
    "P1.0": ("ep5 base", dict()),
    "P1.1": ("neg3.0", dict(neg_ratio=3.0)),
    "P1.2": ("ep7", dict(num_epochs=7)),
    "P1.3": ("abstract ckpt4096", dict(use_abstract=1, cutoff_len=4096)),
    "P1.4": ("hist50", dict(max_history=50)),
    "P1.5": ("neg3 ep7 hist50", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    "P1.7": ("subcategory", dict(use_subcategory=1)),
}


def submit():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN env var not set; cannot fetch gated gemma-2-2b-it.")
        print("Set it, e.g.  HF_TOKEN=hf_xxx python resubmit_failed_sub2b.py")
        sys.exit(1)
    n = 0
    for mtag, hf_id, exps in MODELS:
        for exp in exps:
            disp, ov = CONFIGS[exp]
            job = mind_train_pipeline(
                msndni_input=Input(type="uri_folder",
                                   path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                                   mode=InputOutputModes.RW_MOUNT),
                model_path=hf_id,
                data_root=DATA,
                batch_size=256, micro_batch_size=4,
                num_epochs=ov.get("num_epochs", 5),
                neg_ratio=ov.get("neg_ratio", 2.0),
                hard_neg_ratio=ov.get("hard_neg_ratio", 0.5),
                max_history=ov.get("max_history", 30),
                cutoff_len=ov.get("cutoff_len", 8192),
                use_chat_template=ov.get("use_chat_template", 1),
                use_abstract=ov.get("use_abstract", 0),
                use_subcategory=ov.get("use_subcategory", 0),
                pointwise_ratio=1.0, ranking_neg_ratio=4.0,
                run_eval=1, eval_split="dev",
                early_stopping_patience=3, train_sample=0,
                hf_token=HF_TOKEN,
            )
            job.settings.default_compute = VC_ARM_ID
            job.experiment_name = f"sub2b_{mtag}_{exp.lower().replace('.', '-')}_r2"
            job.display_name = f"{mtag} {exp} {disp} (small, r2)"
            created = ML.jobs.create_or_update(job)
            print(f"{mtag:<12} {exp:<6} {created.name:<26} {disp}")
            sys.stdout.flush()
            n += 1
    print(f"\nSubmitted {n} rerun jobs.")


if __name__ == "__main__":
    submit()
