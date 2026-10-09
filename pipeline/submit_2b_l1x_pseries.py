"""Submit P-series (MIND_small) experiments on Qwen3.5-2B.

Mirrors the historical Qwen3-1.7B P1.x configs EXCEPT model_path -> Qwen3.5-2B.
All other settings identical. Uses run_pipeline.build + ml_client directly so
we stay inside the same pipeline wiring (VC=ranking, UAI, force_rerun).

Usage:
    python submit_2b_l1x_pseries.py
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
MODEL = "Qwen/Qwen3.5-2B"

# (exp, display, key, **overrides)
RUNS = [
    ("P1.0", "2B P1.0 ep5 base (small)", dict()),
    ("P1.1", "2B P1.1 neg3.0 (small)", dict(neg_ratio=3.0)),
    ("P1.2", "2B P1.2 ep7 (small)", dict(num_epochs=7)),
    ("P1.3", "2B P1.3 abstract ckpt4096 (small)", dict(use_abstract=1, cutoff_len=4096)),
    ("P1.4", "2B P1.4 hist50 (small)", dict(max_history=50)),
    ("P1.5", "2B P1.5 neg3 ep7 hist50 (small)", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    ("P1.7", "2B P1.7 subcategory (small)", dict(use_subcategory=1)),
]

for exp, disp, ov in RUNS:
    job = mind_train_pipeline(
        msndni_input=Input(type="uri_folder",
                           path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                           mode=InputOutputModes.RW_MOUNT),
        model_path=MODEL,
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
