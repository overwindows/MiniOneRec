"""Resubmit the 6 qwen3-1p7b r2 runs that failed with pyarrow/numpy drift.

Root cause (deterministic once diagnosed): requirements.txt pinned numpy<2.0.0
but the unbounded "datasets>=3.2.0" let pip resolve a pyarrow>=19 wheel that
REQUIRES numpy 2.0 -> "ImportError: pyarrow requires NumPy 2.0 or newer,
found 1.26.4" at transformers.Trainer() init. Whether a node hits it depends on
its image-cache resolve, so retries kept derailing. requirements.txt now pins
datasets==3.2.0 + pyarrow==17.0.0 (numpy-1.26-compatible), pushed to dev branch,
which these jobs clone fresh.

Usage:
    python resubmit_qwen3_pyarrow.py
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

# (exp, display, **overrides)
CONFIGS = {
    "P1.0": ("ep5 base", dict()),
    "P1.1": ("neg3.0", dict(neg_ratio=3.0)),
    "P1.2": ("ep7", dict(num_epochs=7)),
    "P1.4": ("hist50", dict(max_history=50)),
    "P1.5": ("neg3 ep7 hist50", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    "P1.7": ("subcategory", dict(use_subcategory=1)),
}

# The 6 qwen3-1p7b r2 runs that FAILED with the pyarrow/numpy drift.
# (run_id, exp)
FAILED = [
    ("jovial_mangos_b3vnpz62gw", "P1.0"),
    ("strong_snail_gcyswk8n9n", "P1.1"),
    ("ashy_bee_91p17lxbnl", "P1.2"),
    ("silver_shampoo_s24fl6tjvl", "P1.4"),
    ("mighty_yuca_4ql3d1h4k7", "P1.5"),
    ("olden_pot_bvb2nl0c21", "P1.7"),
]


def submit():
    n = 0
    for run_id, exp in FAILED:
        disp, ov = CONFIGS[exp]
        job = mind_train_pipeline(
            msndni_input=Input(type="uri_folder",
                               path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                               mode=InputOutputModes.RW_MOUNT),
            model_path="Qwen/Qwen3-1.7B",
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
        )
        job.settings.default_compute = VC_ARM_ID
        job.experiment_name = f"sub2b_qwen3-1p7b_{exp.lower().replace('.', '-')}_r3"
        job.display_name = f"qwen3-1p7b {exp} {disp} (small, r3-pyarrowfix)"
        created = ML.jobs.create_or_update(job)
        print(f"{exp:<6} {created.name:<26} {disp}")
        sys.stdout.flush()
        n += 1
    print(f"\nSubmitted {n} qwen3 rerun jobs (pyarrow fix).")


if __name__ == "__main__":
    submit()
