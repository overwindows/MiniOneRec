"""Resubmit the sub-2B rival P-series sweep as r5 with the corrected deps.

History:
  r1 (orig)      gemma: GatedRepoError 401 (anon) -> qwen3: pyarrow/numpy drift
  r2 gemma / r3  gemma: 403 no-license (first token) -> fixed with licensed token
  r3 (qwen3) / r4: exact pyarrow==17.0.0 pin broke -r requirements against
                 verl/ray -> `fire` never installed -> ModuleNotFoundError.
Now dev pins pyarrow<19 (upper bound) which caps the numpy-2-only wheels while
still resolving verl/ray. But the mono `-r` resolve STILL aborted on cold node
image-caches, leaving `import fire` missing (sft_mind_pointwise_ds.py:43). r6
adds an explicit independent install of the SFT-critical deps (fire, datasets,
pyarrow, safetensors, tqdm, einops) in setup_multi_node.sh ahead of the now
best-effort `-r` resolve. r5 died 9/10 to the missing-fire crash; r6 should be
deterministic across nodes.

NOTE: these jobs `git clone` dev at spawn and `git checkout dev`, so they must
be submitted AFTER dev has the pyarrow<19 fix (already pushed). The token must
come from the HF_TOKEN env var (never committed).

Usage:
    HF_TOKEN=hf_xxx python resubmit_sub2b_r5.py
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
    "P1.1": ("neg3.0", dict(neg_ratio=3.0)),
    "P1.2": ("ep7", dict(num_epochs=7)),
    "P1.3": ("abstract ckpt4096", dict(use_abstract=1, cutoff_len=4096)),
    "P1.4": ("hist50", dict(max_history=50)),
    "P1.5": ("neg3 ep7 hist50", dict(neg_ratio=3.0, num_epochs=7, max_history=50)),
    "P1.7": ("subcategory", dict(use_subcategory=1)),
}

# qwen3-1p7b (all 6 of P1.0-P1.7 except P1.3 which isn't in the qwen3 set)
QWEN3 = ["P1.0", "P1.1", "P1.2", "P1.4", "P1.5", "P1.7"]
GEMMA = ["P1.0", "P1.1", "P1.2", "P1.3", "P1.4", "P1.5", "P1.7"]


def submit(model, hf_id, exps, exp_round):
    n = 0
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
        job.experiment_name = f"sub2b_{model}_{exp.lower().replace('.', '-')}_{exp_round}"
        job.display_name = f"{model} {exp} {disp} (small, {exp_round})"
        created = ML.jobs.create_or_update(job)
        print(f"{model:<10} {exp:<6} {created.name:<26} {disp}")
        sys.stdout.flush()
        n += 1
    return n


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN env var not set (gated gemma-2-2b-it).")
        sys.exit(1)
    total = 0
    total += submit("qwen3-1p7b", "Qwen/Qwen3-1.7B", QWEN3, "r6")
    total += submit("gemma2-2b", "google/gemma-2-2b-it", GEMMA, "r6")
    print(f"\nSubmitted {total} r6 jobs (explicit SFT-deps + best-effort -r resolve).")


if __name__ == "__main__":
    main()
