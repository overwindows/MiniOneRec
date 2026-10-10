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
    "sincere_garlic_7t99m7nxw6": "JEV-C1fix small SFT CE EP2 MINDsmall no-distill (ranking-VC)",
    "zen_sun_11q5r489j4": "r11b 2B chat FULL-DEV eval 376k (sdpa, ranking-VC)",
    "nice_plate_8v2hbbvb5y": "r11b 2B Base FULL-DEV eval 376k (sdpa, ranking-VC)",
    "green_salt_0m0w4lhcgz": "r12 2B chat FULL MIND_small (no sample cap) abstract EP5",
    "tender_box_q646hppnl4": "r12 2B-Base FULL MIND_small (no sample cap) abstract EP5",
    "affable_hamster_dfksgqqfjm": "2B P1.0 ep5 base (small)",
    "dreamy_monkey_rb7j50sqc3": "2B P1.1 neg3.0 (small)",
    "olden_fennel_zvl1kjpwnz": "2B P1.2 ep7 (small)",
    "keen_drain_9d0g467ccp": "2B P1.3 abstract ckpt4096 (small)",
    "serene_station_c7g9br7ph1": "2B P1.4 hist50 (small)",
    "neat_potato_7myz9f6k4w": "2B P1.5 neg3 ep7 hist50 (small)",
    "gentle_root_d1c59cx392": "2B P1.7 subcategory (small)",
    "elated_gas_bhqnfn9jm9": "2B L1.1 neg3.0 (large) CANCELED",
    "gentle_toe_0j452vlrjz": "2B L1.2 ep7 (large) CANCELED",
    "placid_king_x117qmb9v4": "2B L1.3 abstract (large) CANCELED",
    "willing_prune_sf8y0p6s33": "2B L1.4 hist50 (large) CANCELED",
    "helpful_cat_nx40whzcb0": "2B L1.5 neg3 ep7 hist50 (large) CANCELED",
    "olden_brake_f3h7dmdgmq": "2B-Base L1.7 (large, no chat) FAILED",
    "careful_lamp_rp23ybf62v": "2B-Base L1.9 abstract (large) CANCELED",
    "cyan_lock_bvhq9qlnf7": "2B L1.13 abstract ep7 neg3 (large) CANCELED",
    "bright_tiger_cssqctnlx9": "2B F1.1 abstract+subcat (large) CANCELED",
    "green_potato_b51jnh1tk0": "2B H1.1 hard-neg 100% (large) CANCELED",
    "polite_vulture_6qxy937lfx": "2B L1.1 neg3.0 (small research)",
    "sleepy_cassava_4qtqckylh0": "2B L1.2 ep7 (small research)",
    "gentle_soca_k0yplhlqdz": "2B L1.3 abstract (small research)",
    "placid_nut_mvf48tgbm3": "2B L1.4 hist50 (small research)",
    "lemon_pasta_t8ct4b0nkx": "2B L1.5 neg3 ep7 hist50 (small research)",
    "cool_bottle_9hwv3v9tl4": "2B-Base L1.7 (small, no chat)",
    "nice_peach_wzg73q6dl8": "2B-Base L1.9 abstract (small research)",
    "sweet_cassava_33crdfhp1d": "2B L1.13 abstract ep7 neg3 (small research)",
    "blue_plum_21l227nncx": "2B F1.1 abstract+subcat (small research)",
    "goofy_box_nghcbmkp1z": "2B H1.1 hard-neg 100% (small research)",
    "salmon_squash_t0b7yt5zls": "2B H1.1 hard-neg 100% (small) rerun hardneg-fix",
    # --- Sub-2B rivals P-series sweep (21 runs, MIND_small) ---
    "great_beard_00zq6m1kcw": "qwen3-1p7b P1.0 ep5 base (small)",
    "kind_quill_7mm72qs2k5": "qwen3-1p7b P1.1 neg3.0 (small)",
    "heroic_feast_vpx3b57sd9": "qwen3-1p7b P1.2 ep7 (small)",
    "magenta_coconut_998v4105wc": "qwen3-1p7b P1.3 abstract ckpt4096 (small)",
    "happy_worm_y0m288lqqz": "qwen3-1p7b P1.4 hist50 (small)",
    "salmon_shark_blpj4qn2kb": "qwen3-1p7b P1.5 neg3 ep7 hist50 (small)",
    "elated_plum_6pf77ghcb2": "qwen3-1p7b P1.7 subcategory (small)",
    "nice_rose_2jq09f0361": "qwen25-1p5b P1.0 ep5 base (small)",
    "plum_king_v75p4sjgzy": "qwen25-1p5b P1.1 neg3.0 (small)",
    "silly_receipt_l3mgzrk40h": "qwen25-1p5b P1.2 ep7 (small)",
    "boring_vulture_pmzjggct0d": "qwen25-1p5b P1.3 abstract ckpt4096 (small)",
    "clever_room_lfb0n3zwv1": "qwen25-1p5b P1.4 hist50 (small)",
    "musing_oyster_h5nkkw3tvq": "qwen25-1p5b P1.5 neg3 ep7 hist50 (small)",
    "stoic_button_vx1f4n02h4": "qwen25-1p5b P1.7 subcategory (small)",
    "magenta_shelf_vbkv8snltt": "gemma2-2b P1.0 ep5 base (small)",
    "kind_turnip_lj6pk3hcwf": "gemma2-2b P1.1 neg3.0 (small)",
    "funny_rocket_h1tn68wfdq": "gemma2-2b P1.2 ep7 (small)",
    "amiable_parang_pth6htlzb8": "gemma2-2b P1.3 abstract ckpt4096 (small)",
    "neat_feather_nk1dky1l3y": "gemma2-2b P1.4 hist50 (small)",
    "silver_needle_vnttdxcyqq": "gemma2-2b P1.5 neg3 ep7 hist50 (small)",
    "tough_lunch_rp8063c5wd": "gemma2-2b P1.7 subcategory (small)",
    # --- r2 retries: qwen3 (transient pyarrow env) + gemma (HF token/license fix) ---
    "jovial_mangos_b3vnpz62gw": "qwen3-1p7b P1.0 ep5 base (small, r2)",
    "strong_snail_gcyswk8n9n": "qwen3-1p7b P1.1 neg3.0 (small, r2)",
    "ashy_bee_91p17lxbnl": "qwen3-1p7b P1.2 ep7 (small, r2)",
    "silver_shampoo_s24fl6tjvl": "qwen3-1p7b P1.4 hist50 (small, r2)",
    "mighty_yuca_4ql3d1h4k7": "qwen3-1p7b P1.5 neg3 ep7 hist50 (small, r2)",
    "olden_pot_bvb2nl0c21": "qwen3-1p7b P1.7 subcategory (small, r2)",
    "dreamy_root_1cy02qhc1d": "gemma2-2b P1.0 ep5 base (small, r2-401)",
    "joyful_foot_x4z0lcxv4q": "gemma2-2b P1.1 neg3.0 (small, r2-401)",
    "jolly_chin_9xwnc8kl6p": "gemma2-2b P1.2 ep7 (small, r2-401)",
    "quirky_vase_pc75nzy6sy": "gemma2-2b P1.3 abstract ckpt4096 (small, r2-401)",
    "salmon_machine_1x233rfk2y": "gemma2-2b P1.4 hist50 (small, r2-401)",
    "dynamic_cat_ybt0ly8dwy": "gemma2-2b P1.5 neg3 ep7 hist50 (small, r2-401)",
    "good_yak_y6t0vwm341": "gemma2-2b P1.7 subcategory (small, r2-401)",
    "sincere_root_48xlvknh42": "gemma2-2b P1.1 neg3.0 (small, r3-licensed)",
    "lemon_hand_dnb7c9srtp": "gemma2-2b P1.2 ep7 (small, r3-licensed)",
    "helpful_bread_ltfmydxw4j": "gemma2-2b P1.3 abstract ckpt4096 (small, r3-licensed)",
    "musing_spring_ynv7vcd7ds": "gemma2-2b P1.4 hist50 (small, r3-licensed)",
    "musing_avocado_8g37lg0lx9": "gemma2-2b P1.5 neg3 ep7 hist50 (small, r3-licensed)",
    "hungry_leek_vmldssjx7m": "gemma2-2b P1.7 subcategory (small, r3-licensed)",
    "mighty_kale_6j7jh9tbk7": "gemma2-2b P1.0 ep5 base (small, r3-licensed)",
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
