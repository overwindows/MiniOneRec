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
    "sleepy_pasta_g9hytyykq9": "r8 Qwen3.5-2B chat abstract EP7 (L1.2-repro)",
    "olden_drain_3b6psyybwj": "r9 Qwen3.5-2B chat abstract HIST50",
    "upbeat_leaf_43yqnzr0h8": "r10 Qwen3.5-2B chat L1.3-winner-repro abstract EP5",
    "sharp_napa_myhlzwqvh3": "r7-ev2 Qwen3.5-2B chat abstract EP3 dev BS1",
    "maroon_brush_xvk6c59278": "r7-ev2 Qwen3.5-2B-Base abstract EP3 dev BS1",
    "boring_snake_zwwk3r461s": "JEV9B-download CPU (APUS-OpenJev-v1-9B staging)",
    "neat_stone_3427s7bjrr": "JEV9B dev@5000 eval",
    "affable_fork_57j81ydc2b": "C1 JEV4B SFT+distill MINDlarge abstract EP3 KL0.1 (FAILED num_items_in_batch)",
    "frank_chicken_fct2hw878v": "C1 JEV4B SFT+distill MINDlarge abstract EP3 KL0.1 (retry, CANCELLED zero-loss idx bug)",
    "stoic_hook_xdc9sbfbk6": "C1b JEV4B SFT+distill MINDlarge abstract EP3 KL0.1 (idx-fix, zombie canceled)",
    "lemon_ship_f2nfshcfh9": "C1c JEV4B SFT+distill abstract EP3 KL0.1 (force_rerun, fresh)",
    "upbeat_nerve_xhyfgsv0sz": "r7-ev2 chat eval resub (bs1, NoIdentity fix)",
    "yellow_yak_xh87jbrdmg": "r7-ev2 Base eval resub (bs1, NoIdentity fix)",
    "quiet_drop_wfpvxsgwx3": "JEV9B dev@5000 eval resub (OOM fix)",
    "joyful_juice_zh8gm2z86r": "r9 JEVhist50 resub (force_rerun fix, fresh)",
    "placid_horse_fb32d274zt": "r7-ev2 chat eval bs1 std (FAILED NoIdentity, replaced by brave_rhythm)",
    "bright_beard_677p5gz563": "r7-ev2 Base eval bs1 std (FAILED NoIdentity, replaced by goofy_beet)",
    "witty_tree_dmkbfy3xkc": "JEV9B dev@5000 eval std (FAILED list.update bug, replaced by tender_carrot)",
    "tender_carrot_sbxpx7q8xv": "JEV9B dev@5000 eval v3 std (auth+OOM fix)",
    "brave_rhythm_grkhnk42yf": "r7-ev2 chat eval bs1 v3 std (FAILED OOM, replaced by loving_fig)",
    "goofy_beet_q8cg4r2mr0": "r7-ev2 Base eval bs1 v3 std (FAILED OOM, replaced by nifty_rabbit)",
    "loving_fig_d6sw5hr69d": "r7-ev2 chat eval bs1 v4 std (expandable_segments fix)",
    "nifty_rabbit_8d4hr31sxd": "r7-ev2 Base eval bs1 v4 std (expandable_segments fix)",
    "neat_dress_jsqx6fnngc": "r7 chat eval bs8 8gpu FULL (scaled, 28d-ETA singleGPU fix)",
    "honest_pig_qdvc9123ck": "r7 Base eval bs8 8gpu FULL (scaled, 28d-ETA singleGPU fix)",
    "shy_branch_8tsybcl6h0": "fla-smoke chat LARGE mp400 1gpu (fla fast-path validation, FAILED causal-conv1d>= abort)",
    "upbeat_wolf_wsz4qrz3kk": "fla-smoke2 chat LARGE 1gpu mp400 (causal-conv1d==1.4.0 pin fix)",
    "dynamic_sock_3t6byzn5r8": "r7 chat dev@5000 bs1 flafix (pre-heredoc, FAILED)",
    "witty_planet_p9lhjl99n9": "r7 Base dev@5000 bs1 flafix (pre-heredoc, FAILED)",
    "goofy_dinner_xwt8c0gt7j": "r7 chat dev@5000 bs1 heredocfix (CANCELED, duplicate)",
    "serene_ear_0xcn7yync7": "r7 chat dev@5000 bs1 heredocfix (KEEP→CANCELED full-dev)",
    "serene_bee_5hk1vvq8wp": "r7 Base dev@5000 bs1 heredocfix2 (KEEP→CANCELED full-dev)",
    "affable_jicama_j5dfcwhg8p": "r7 chat dev@5000 bs1 mi-fix (max_impressions positional fix)",
    "elated_worm_s7t33tl09b": "r7 Base dev@5000 bs1 mi-fix (max_impressions positional fix)",
    "quirky_pig_mrd62sv4xf": "r11 2B chat ckpt4096 s300k L1.3-repro (TRAIN_SAMPLE bound)",
    "serene_rice_88lx58jm5j": "r11b 2B chat ckpt4096 s300k resub (TRAIN_SAMPLE=300000)",
    "magenta_pasta_7j0yzmwxhw": "r11b 2B Base ckpt4096 s300k resub (TRAIN_SAMPLE=300000, chat_tpl=0)",
    "coral_lion_qrt71styxv": "r11b 2B chat ckpt4096 s300k dev@5000 sdpa eval (eval-fix, CANCELED quota)",
    "olive_steelpan_spqkjfn91t": "r11b 2B chat ckpt4096 s300k dev@5000 sdpa eval (eval-fix, ranking-VC)",
    "tender_bread_w3zgfpgcsv": "r11b Base eval (FAILED wrong ckpt path, replaced by frosty_whistle)",
    "frosty_whistle_tlztv5jz9h": "r11b 2B Base ckpt4096 s300k dev@5000 sdpa eval (correct ckpt path, ranking-VC)",
    "icy_school_n62sqfsycd": "JEV-C1 SFT+distill large abstract EP3 KL0.1 dev@5000 eval (ranking-VC)",
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
