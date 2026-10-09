"""Submit full P-series (MIND_small) sweep across sub-2B rival models.

3 models x 7 P-series configs = 21 runs on MIND_small (train_sample=0, no cap).
Models are resolved at runtime by bare HF ID — no datastore staging needed.

Grid (identical settings to the historical Qwen3-1.7B P1.x sweep):
    P1.0 ep5 base | P1.1 neg3.0 | P1.2 ep7 | P1.3 abstract ckpt4096 |
    P1.4 hist50 | P1.5 neg3 ep7 hist50 | P1.7 subcategory

Model choice notes:
    - Qwen/Qwen3-1.7B, Qwen/Qwen2.5-1.5B: apply_chat_template available (chat_idx=1).
    - google/gemma-2-2b-it: base gemma-2-2b has NO chat template; P-series uses
      use_chat_template=1, so we use the -it (instruct) variant which defines one.

Usage:
    python submit_sub2b_pseries.py
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

# 3 rival models (sub-2B size class). Qwen3-1.7B is our historical best single.
MODELS = [
    ("qwen3-1p7b", "Qwen/Qwen3-1.7B"),
    ("qwen25-1p5b", "Qwen/Qwen2.5-1.5B"),
    ("gemma2-2b", "google/gemma-2-2b-it"),
]

# (exp, display, **overrides)
CONFIGS = [
    ("P1.0", "ep5 base", dict()),
    ("P1.1", "neg3.0", dict(neg_ratio=3.0)),
    ("P1.2", "ep7", dict(num_epochs=7)),
    ("P1.3", "abstract ckpt4096", dict(use_abstract=1, cutoff_len=4096)),
    ("P1.4", "hist50", dict(max_history=50)),
    ("P1.5", "neg3 ep7 hist50", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    ("P1.7", "subcategory", dict(use_subcategory=1)),
]


def submit():
    n = 0
    for mtag, hf_id in MODELS:
        for exp, disp, ov in CONFIGS:
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
            )
            job.settings.default_compute = VC_ARM_ID
            job.experiment_name = f"sub2b_{mtag}_{exp.lower().replace('.', '-')}"
            job.display_name = f"{mtag} {exp} {disp} (small)"
            created = ML.jobs.create_or_update(job)
            print(f"{mtag:<12} {exp:<6} {created.name:<26} {disp}")
            sys.stdout.flush()
            n += 1
    print(f"\nSubmitted {n} runs.")


if __name__ == "__main__":
    submit()
