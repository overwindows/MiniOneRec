"""Collect final dev AUC from completed sub-2B / A7.3 eval jobs.

Fetches the std_log of each eval child, greps the final calc_mind_metrics AUC
block, and prints a compact summary. Use after jobs reach Completed status.

Usage:
    python collect_eval_aucs.py
"""
import sys, io, urllib.request, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from azure.ai.ml import MLClient

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")

# parent run id -> display label
JOBS = {
    "olive_leek_6k8svsx0lw": "qwen3 P1.5",
    "frank_cord_yby9j1rl5h": "qwen3 P1.4",
    "gifted_diamond_zf9wq2wc33": "qwen3 P1.7 subcat",
    "strong_basket_q5jzrgqrbd": "gemma P1.2",
    "boring_ghost_15449mswxn": "gemma P1.4",
    "mango_leather_m2t42q6bqr": "gemma P1.5",
    "gentle_salt_5hpqn3nqmy": "gemma P1.7 subcat",
    "quiet_giraffe_3zk1gxdxgj": "A7.3 pw0.0 abs0 control",
    "ashy_corn_rdvd7kf9r4": "A7.3 pw0.5 abs0 title",
    "polite_pig_mnty8f5v9f": "A7.3 pw0.5 abs1 abstract",
}


def child_of(rid):
    children = list(ML.jobs.list(parent_job_name=rid))
    for c in children:
        if c.name:  # the eval_node step
            return c.name
    return None


def fetch_tail(url, n=8000):
    data = urllib.request.urlopen(urllib.request.Request(url), timeout=90).read().decode("utf-8", "replace")
    return data[-n:]


def main():
    for rid, label in JOBS.items():
        try:
            status = str(ML.jobs.get(rid).status)
        except Exception as e:
            print(f"{label:<16} status ERR {e}")
            continue
        if status.upper() != "JOBSTATUS.COMPLETED":
            print(f"{label:<16} {status}")
            continue
        c = child_of(rid)
        if not c:
            print(f"{label:<16} (no child)")
            continue
        try:
            rd = ML.jobs._runs_operations.get_run_details(c)
        except Exception as e:
            print(f"{label:<16} child details ERR {e}")
            continue
        url = None
        for k in (rd.log_files or {}):
            if k.endswith("std_log.txt"):
                url = rd.log_files[k]; break
        if not url:
            print(f"{label:<16} (no std_log)")
            continue
        try:
            tail = fetch_tail(url)
        except Exception as e:
            print(f"{label:<16} fetch ERR {e}")
            continue
        # final AUC line (format: "AUC:      0.6582")
        m = re.findall(r"AUC\s*[:=]\s*([\d.]+)", tail)
        aucs = m[-3:] if m else []
        print(f"{label:<16} COMPLETED  final AUC={aucs}")


if __name__ == "__main__":
    main()
