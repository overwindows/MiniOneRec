"""Reusable monitor for AML MIND jobs.

Usage:
    python monitor_jobs.py --status [RUN_ID ...]
    python monitor_jobs.py --logs RUN_ID [--lines N] [--search PATTERN] [--tail]
    python monitor_jobs.py --cancel RUN_ID [RUN_ID ...]
    python monitor_jobs.py --loop [--interval 300]  # poll all known runs forever
"""
import argparse
import io
import os
import sys
import time

# Windows console is often cp1252/latin-1; force UTF-8 for output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

SUB = "b6dc87f3-c479-49c8-8cb5-7896da3ff895"
RG = "AMLStudio"
WS = "NewsFeedL2_AML"

KNOWN = {
    "coral_yak_h5pzn92w33": "Qwen3.5-2B MINDlarge abstract (r4, viewer probe failed)",
    "bubbly_dress_zx4969kxzv": "Qwen3.5-2B-Base MINDlarge abstract (r4, viewer probe failed)",
    "blue_basil_zt8d7w4j8d": "Qwen3.5-2B MINDlarge abstract (r3, wandb crash)",
    "maroon_pea_r3502z5vhs": "Qwen3.5-2B-Base MINDlarge abstract (r3, canceled->wandbfix)",
    "tender_airport_bsn0ky6hv6": "Qwen3.5-2B (r2, failed group_by_length)",
    "wheat_snake_64srk3qwcs": "Qwen3.5-2B-Base (r2, failed group_by_length)",
    "neat_band_393lt71y7t": "r7 Qwen3.5-2B chat MINDlarge abstract gckpt (RUNNING)",
    "happy_energy_zrvmghx41r": "r7 Qwen3.5-2B-Base MINDlarge abstract gckpt (RUNNING)",
    "shy_fowl_14fkx19tyj": "M4.1 multitask 1.7B large abstract EP5 PW0.5 (queued)",
    "bold_foot_75cv39dfth": "L1.3x pointwise 1.7B large abstract EP5 (queued)",
}


def client():
    return MLClient(DefaultAzureCredential(), subscription_id=SUB,
                    resource_group_name=RG, workspace_name=WS)


def get_child_steps(ml, run_id):
    """Return list of child step job objects for a pipeline run."""
    try:
        return list(ml.jobs.list(parent_job_name=run_id))
    except Exception:
        return []


def status_all(ml, run_ids):
    print(f"{'RUN_ID':<26} {'STATUS':<14} NAME")
    for rid in run_ids:
        try:
            j = ml.jobs.get(rid)
            name = KNOWN.get(rid, j.display_name or "")
            print(f"{rid:<26} {str(j.status):<14} {name}")
            # pipeline: show children
            for child in get_child_steps(ml, rid):
                print(f"  └─ {child.name:<22} {str(child.status):<14} "
                      f"{child.display_name or ''}")
        except Exception as e:
            print(f"{rid:<26} ERROR {type(e).__name__}: {e}")


def fetch_logs(ml, run_id, lines=60, search=None, tail=False):
    """Fetch the most relevant stream (70_driver_log.txt or azureml-logs)."""
    try:
        # Preferred runtime log stream on Singularity nodes
        logs = ml.jobs._jobs_operations.get_logs(run_id, name="70_driver_log")
    except Exception:
        logs = ""
    if not logs:
        try:
            text = io.StringIO()
            ml.jobs._jobs_operations.stream(run_id, text)
            logs = text.getvalue()
        except Exception:
            logs = ""
    if not logs:
        print(f"No logs available for {run_id}")
        return
    lines_list = logs.splitlines()
    if search:
        matched = [l for l in lines_list if search.lower() in l.lower()]
        results = matched if not tail else matched[-int(tail):]
        print(f"[{run_id}] {len(matched)} matches for '{search}':")
        print("\n".join(results[-lines:]))
        return
    shown = lines_list if not tail else lines_list[-int(tail):]
    print(f"[{run_id}] showing last {len(shown)} lines:")
    print("\n".join(shown[-lines:]))


def cancel(ml, run_ids):
    for rid in run_ids:
        try:
            ml.jobs.begin_cancel(rid)
            print(f"Cancelling {rid}")
        except Exception as e:
            print(f"{rid}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", nargs="*", default=None)
    ap.add_argument("--logs", metavar="RUN_ID")
    ap.add_argument("--lines", type=int, default=60)
    ap.add_argument("--search")
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--cancel", nargs="+", default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    ml = client()

    if args.cancel:
        cancel(ml, args.cancel)
    if args.logs:
        fetch_logs(ml, args.logs, args.lines, args.search, args.tail or 0)
    if args.status is not None:
        rids = args.status if args.status else list(KNOWN.keys())
        status_all(ml, rids)
    if args.loop:
        while True:
            print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            status_all(ml, list(KNOWN.keys()))
            sys.stdout.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
