"""Submit L-series + F1.1 + H1.1 (MIND_small — RESEARCH phase) on Qwen3.5-2B.

RESEARCH FIRST strategy: run the L1.x sweep on MIND_small for a fast validation
loop. Promote only winners to MIND_large afterward.

Usage:
    python submit_2b_l1x_lseries.py
"""
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
CHAT = "Qwen/Qwen3.5-2B"
BASE = "Qwen/Qwen3.5-2B-Base"

# (exp, display, model, **overrides)
RUNS = [
    ("L1.1", "2B L1.1 neg3.0 (large)", CHAT, dict(neg_ratio=3.0)),
    ("L1.2", "2B L1.2 ep7 (large)", CHAT, dict(num_epochs=7)),
    ("L1.3", "2B L1.3 abstract (large)", CHAT, dict(use_abstract=1)),
    ("L1.4", "2B L1.4 hist50 (large)", CHAT, dict(max_history=50)),
    ("L1.5", "2B L1.5 neg3 ep7 hist50 (large)", CHAT, dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    ("L1.7", "2B-Base L1.7 (large, no chat)", BASE, dict(use_chat_template=0)),
    ("L1.9", "2B-Base L1.9 abstract (large)", BASE, dict(use_chat_template=0, use_abstract=1)),
    ("L1.13", "2B L1.13 abstract ep7 neg3 (large)", CHAT, dict(num_epochs=7, neg_ratio=3.0, use_abstract=1)),
    ("F1.1", "2B F1.1 abstract+subcat (large)", CHAT, dict(use_abstract=1, use_subcategory=1)),
    ("H1.1", "2B H1.1 hard-neg 100% (large)", CHAT, dict(use_abstract=1, hard_neg_ratio=1.0)),
]

for exp, disp, model, ov in RUNS:
    job = mind_train_pipeline(
        msndni_input=Input(type="uri_folder",
                           path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                           mode=InputOutputModes.RW_MOUNT),
        model_path=model,
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
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = f"2b_l1x_{exp.lower().replace('.', '-')}"
    job.display_name = disp
    created = ML.jobs.create_or_update(job)
    print(f"{exp:<6} {created.name:<26} {disp}")
    sys.stdout.flush()
