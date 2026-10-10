"""Resubmit the 3 gemma-2-2b r3 runs that failed with pyarrow/numpy drift.

Same root cause as qwen3: requirements.txt pinned numpy<2.0.0 but unbounded
datasets>=3.2.0 let pip resolve pyarrow>=19 (needs numpy 2.0) -> ImportError
at Trainer init, node-image-cache dependent. Now pinned (datasets==3.2.0 +
pyarrow==17.0.0) and pushed to dev, which these jobs clone fresh. The gated
gemma access is NOT the issue here (that 403 is fixed by the licensed token).

Usage:
    HF_TOKEN=hf_xxx python resubmit_gemma_pyarrow.py
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
HF_TOKEN = os.environ.get("HF_TOKEN", "")

CONFIGS = {
    "P1.0": ("ep5 base", dict()),
    "P1.4": ("hist50", dict(max_history=50)),
    "P1.5": ("neg3 ep7 hist50", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
}

# gemma r3 runs that FAILED with pyarrow/numpy drift
FAILED = [
    ("mighty_kale_6j7jh9tbk7", "P1.0"),
    ("musing_spring_ynv7vcd7ds", "P1.4"),
    ("musing_avocado_8g37lg0lx9", "P1.5"),
]


def submit():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN env var not set (gated gemma-2-2b-it).")
        sys.exit(1)
    n = 0
    for run_id, exp in FAILED:
        disp, ov = CONFIGS[exp]
        job = mind_train_pipeline(
            msndni_input=Input(type="uri_folder",
                               path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                               mode=InputOutputModes.RW_MOUNT),
            model_path="google/gemma-2-2b-it",
            data_root=DATA,
            batch_size=256, micro_batch_size=4,
            num_epochs=ov.get("num_epochs", 5),
            neg_ratio=ov.get("neg_ratio", 2.0),
            hard_neg_ratio=0.5,
            max_history=ov.get("max_history", 30),
            cutoff_len=ov.get("cutoff_len", 8192),
            use_chat_template=1,
            use_abstract=ov.get("use_abstract", 0),
            use_subcategory=ov.get("use_subcategory", 0),
            pointwise_ratio=1.0, ranking_neg_ratio=4.0,
            run_eval=1, eval_split="dev",
            early_stopping_patience=3, train_sample=0,
            hf_token=HF_TOKEN,
        )
        job.settings.default_compute = VC_ARM_ID
        job.experiment_name = f"sub2b_gemma2-2b_{exp.lower().replace('.', '-')}_r4"
        job.display_name = f"gemma2-2b {exp} {disp} (small, r4-pyarrowfix)"
        created = ML.jobs.create_or_update(job)
        print(f"{exp:<6} {created.name:<26} {disp}")
        sys.stdout.flush()
        n += 1
    print(f"\nSubmitted {n} gemma rerun jobs (pyarrow fix, licensed token).")


if __name__ == "__main__":
    submit()
