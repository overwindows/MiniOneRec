"""Auto-submit A7.3 evals when each training job completes.

Polls the 3 A7.3 training runs (MIND_small staging) and, the first time one
reaches a terminal state (Completed/Failed) with its final_checkpoint present,
submits a standalone eval for it. Idempotent: tracks which train rids are
already evaluated in a state file so we never double-submit after a restart.

Usage:
    python auto_eval_a73.py            # one pass, returns
    # or loop it: while true; do python auto_eval_a73.py; sleep 600; done
"""
import os
import sys
import json
import datetime
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure.ai.ml import MLClient, Input
from azure.ai.ml.constants import InputOutputModes
from azure_auth import get_credential
from run_eval_pipeline import mind_eval_pipeline, VC_ARM_ID
import resubmit_eval_a73 as EV

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

STATE_FILE = "Q:/MiniOneRec/pipeline/.a73_eval_state.json"

# train rid -> (label, checkpoint_path, max_history, abstract, subcat)
TRAIN_JOBS = {
    "silver_ring_l47fyrmkx0": ("A7.3 pw0.0 abs0 control",
        "shares/users/wuc/output_dir/sft_mind_pointwise_pairwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_pw0.0_chat/final_checkpoint",
        30, 0, 0),
    "serene_bell_nmnm20cn4v": ("A7.3 pw0.5 abs0 title-only",
        "shares/users/wuc/output_dir/sft_mind_pointwise_pairwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_pw0.5_chat/final_checkpoint",
        30, 0, 0),
    "bubbly_screw_j68j6vw7rs": ("A7.3 pw0.5 abs1 abstract",
        "shares/users/wuc/output_dir/sft_mind_pointwise_pairwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_pw0.5_abstract_chat/final_checkpoint",
        30, 1, 0),
}


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


def status(rid):
    j = ML.jobs.get(rid)
    return str(j.status)


def submit_eval(label, ckpt, mh, abstract, subcat):
    job = mind_eval_pipeline(
        msndni_input=Input(type="uri_folder",
                           path="azureml://datastores/adls_msn_dni_09_rankfun/paths/",
                           mode=InputOutputModes.RW_MOUNT),
        model_path=ckpt, data_root=EV.DATA,
        eval_type="pointwise", split="dev",
        use_chat_template=1, use_abstract=abstract,
        use_subcategory=subcat, max_history=mh,
        batch_size=8, num_gpus=8, cf_alpha=0.0, max_impressions=0,
    )
    job.settings.default_compute = VC_ARM_ID
    job.experiment_name = "a73_pairwise_small_eval"
    job.display_name = f"eval {label}"
    created = ML.jobs.create_or_update(job)
    print(f"{datetime.datetime.now():%F %T}  eval submitted {label:<28} {created.name}")
    sys.stdout.flush()
    return created.name


def main():
    state = load_state()
    for rid, (label, ckpt, mh, abstract, subcat) in TRAIN_JOBS.items():
        if state.get(rid):
            continue  # already eval submitted
        s = status(rid)
        s_upper = s.upper()
        if s_upper in ("JOBSTATUS.COMPLETED", "JOBSTATUS.FAILED"):
            print(f"{datetime.datetime.now():%F %T}  {label}: training {s} -> submit eval")
            ev = submit_eval(label, ckpt, mh, abstract, subcat)
            state[rid] = {"train_status": s, "eval_rid": ev,
                          "time": datetime.datetime.now().isoformat()}
            save_state(state)
        else:
            print(f"{datetime.datetime.now():%F %T}  {label}: training {s}")


if __name__ == "__main__":
    main()
