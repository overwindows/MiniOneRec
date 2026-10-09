"""Regenerate progress.json from AML job states and upload to the static site.

Usage:
    update_progress.py [-o out.json] [--campaign JSON] [--note 'text']

Reads live AML status for the JEV / MIND job catalog in monitor_jobs.KNOWN,
under the campaign wiring below (id -> display name, metric from KNOWN).
Writes progress.json locally and uploads it to $web on
privacypshowcase12114 (published at privacypshowcase12114.z19.web.core.windows.net).

Any --campaign overrides extend/replace the builtin wiring. Notes per campaign
passed via --note become banner text.
"""
import argparse, datetime, json, os, subprocess, sys, tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
import monitor_jobs as M

STORE = "privacypshowcase12114"
WEB = "$web"

# campaign -> dict with display fields + run list (known_key, display_name, metric)
# motivation/settings/data/eval are static prose shown on the live page.
CAMPAIGNS = [
    {
        "name": "JEV-4B Campaign-1: SFT (MIND small → large)",
        "motivation": "Stock OpenJev is a frozen 'no-UL' ranking model (0.6432 AUC) that never saw MIND news. "
                      "Fine-tune it to P(yes) on MIND pointwise samples to build a news-specialized ranking "
                      "model — the foundation the RLCD / RL / Teacher-OPD campaigns build on.",
        "settings": "Qwen3.5 stacked linear-attn JEV-4B · noul A/B prompt · plain pointwise SFT CE on the "
                    "answer token (R-Drop self-KL collapsed to chance 0.4975 and was removed) · sdpa · "
                    "micro-batch 2 · lr 1e-4 · DS ZeRO-2 · abstract, hist30, neg1.0, cutoff4096",
        "data": "MIND_small first (fast validation loop) → re-scale to MINDlarge once signal is confirmed",
        "eval": "run_jev_eval.py noul scorer on dev@5000 → recompute pointwise AUC vs stock JEV 0.6432",
        "runs": [
            ("sincere_garlic_7t99m7nxw6", "C1fix SFT CE (MIND_small) train", ""),
            ("boring_car_qdw46wzq1k", "C1fix SFT CE (MIND_small) dev@5000 eval", "0.4959"),
        ],
    },
    {
        "name": "JEV-4B Campaign-2: RLCD",
        "motivation": "Distill click preferences without an online reward model: score candidates with the "
                      "Campaign-1 model, build contrastive prefs, fine-tune to raise P(A) on preferred vs "
                      "dispreferred (DPO-style).",
        "settings": "DPO-style contrastive distillation from SFT-JEV self-scores · noul A/B contract",
        "data": "MIND_small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "JEV-4B Campaign-3: RL (GRPO)",
        "motivation": "Reinforcement learning from a pointwise reward over noul A/B responses on the "
                      "Campaign-1 checkpoint, reusing the existing verl/GRPO + vLLM infra.",
        "settings": "verl GRPO · reward_type=pointwise_binary / mind_auc · vLLM rollout · on C1 checkpoint",
        "data": "MIND small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "JEV-4B Campaign-4: Teacher OPD",
        "motivation": "Offline preference distillation from a strong external teacher (our L1.3 1.7B best "
                      "pointwise ranker) into JEV-4B — a real, distinct teacher signal (unlike the failed "
                      "within-model R-Drop).",
        "settings": "KL-to-frozen-teacher logits term on JEV noul format · β≈0.1",
        "data": "MIND small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "Qwen3.5-2B reference pointwise (r11b/r12)",
        "motivation": "Same linear-attention family as JEV; reference pointwise ranker for comparison.",
        "settings": "abstract gckpt · r12: FULL MIND_small no sample cap (EP5 ckpt4096) · r11b: MINDlarge s300k (30万样本上限)",
        "data": "r12: MIND_small (no cap) · r11b: MINDlarge (s300k truncated)",
        "eval": "dev@5000 AUC",
        "runs": [
            ("neat_band_393lt71y7t", "r7 chat abstract gckpt", ""),
            ("happy_energy_zrvmghx41r", "r7 Base abstract gckpt", ""),
            ("serene_rice_88lx58jm5j", "r11b chat SFT (MINDlarge s300k) train", ""),
            ("olive_steelpan_spqkjfn91t", "r11b chat dev@5000 eval", "0.6542"),
            ("magenta_pasta_7j0yzmwxhw", "r11b Base SFT (MINDlarge s300k) train", ""),
            ("frosty_whistle_tlztv5jz9h", "r11b Base dev@5000 eval", "0.6272"),
            ("zen_sun_11q5r489j4", "r11b chat FULL-DEV eval (376k)", ""),
            ("nice_plate_8v2hbbvb5y", "r11b Base FULL-DEV eval (376k)", ""),
            ("green_salt_0m0w4lhcgz", "r12 chat FULL MIND_small (no cap) train", ""),
            ("tender_box_q646hppnl4", "r12 Base FULL MIND_small (no cap) train", ""),
        ],
    },
    {
        "name": "Qwen3.5-2B P-series rerun (MIND_small L1.x sweep)",
        "motivation": "Rerun the historical Qwen3-1.7B MIND_small P1.x sweep on Qwen3.5-2B — identical settings, only model changes.",
        "settings": "bs256 micro4 · train_sample=0 (full MIND_small) · chat template · gckpt · cutoff8192 (P1.3=4096)",
        "data": "MIND_small",
        "eval": "dev@5000 AUC",
        "runs": [
            ("affable_hamster_dfksgqqfjm", "2B P1.0 ep5 base (small)", "0.6360"),
            ("dreamy_monkey_rb7j50sqc3", "2B P1.1 neg3.0 (small)", "0.6287"),
            ("olden_fennel_zvl1kjpwnz", "2B P1.2 ep7 (small)", "0.6424"),
            ("keen_drain_9d0g467ccp", "2B P1.3 abstract ckpt4096 (small)", "0.6385"),
            ("serene_station_c7g9br7ph1", "2B P1.4 hist50 (small)", "0.6429"),
            ("neat_potato_7myz9f6k4w", "2B P1.5 neg3 ep7 hist50 (small)", "0.6189"),
            ("gentle_root_d1c59cx392", "2B P1.7 subcategory (small)", "0.6253"),
        ],
    },
    {
        "name": "Qwen3.5-2B L-series rerun (MIND_small — research phase)",
        "motivation": "Rerun the historical Qwen3-1.7B L1.x sweep (incl. F1.1/H1.1) on Qwen3.5-2B — identical settings, only model changes. RESEARCH FIRST on MIND_small; winners promote to MIND_large.",
        "settings": "bs256 micro4 · train_sample=0 (full MIND_small) · chat template (L1.7/L1.9 Base no-chat) · gckpt · cutoff8192",
        "data": "MIND_small",
        "eval": "dev@5000 AUC",
        "runs": [
            ("polite_vulture_6qxy937lfx", "2B L1.1 neg3.0 (small research)", "0.6287"),
            ("sleepy_cassava_4qtqckylh0", "2B L1.2 ep7 (small research)", "0.6424"),
            ("gentle_soca_k0yplhlqdz", "2B L1.3 abstract (small research)", ""),
            ("placid_nut_mvf48tgbm3", "2B L1.4 hist50 (small research)", "0.6429"),
            ("lemon_pasta_t8ct4b0nkx", "2B L1.5 neg3 ep7 hist50 (small research)", "0.6189"),
            ("cool_bottle_9hwv3v9tl4", "2B-Base L1.7 (small, no chat)", ""),
            ("nice_peach_wzg73q6dl8", "2B-Base L1.9 abstract (small research)", ""),
            ("sweet_cassava_33crdfhp1d", "2B L1.13 abstract ep7 neg3 (small research)", ""),
            ("blue_plum_21l227nncx", "2B F1.1 abstract+subcat (small research)", ""),
            ("salmon_squash_t0b7yt5zls", "2B H1.1 hard-neg 100% (small) rerun hardneg-fix", ""),
            ("goofy_box_nghcbmkp1z", "2B H1.1 (OLD — collision w/ P1.0, unreliable)", ""),
        ],
    },
]

HEADLINE = (
    "Target: beat MIND SOTA zetik 0.7326 (currently 0.7144, #14). "
    "JEV Campaign-1 SFT trained on MIND_small (COMPLETED); dev eval running."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json"))
    ap.add_argument("--note", action="append", default=[], help="campaign:note")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    # merge --note into a {campaign: note} map
    note_map = {}
    for ent in args.note:
        if ":" in ent:
            k, v = ent.split(":", 1)
            note_map[k] = v

    ml = M.client()
    campaigns = []
    for camp in CAMPAIGNS:
        cname, runs = camp["name"], camp["runs"]
        cl = []
        for key, disp, metric in runs:
            status = None
            try:
                j = ml.jobs.get(key)
                status = str(j.status).replace("JobStatus.", "")
            except Exception:
                status = "unknown"
            cl.append({"id": key, "name": disp, "metric": metric or "", "status": status})
        campaigns.append({
            "name": cname,
            "runs": cl,
            "motivation": camp.get("motivation", ""),
            "settings": camp.get("settings", ""),
            "data": camp.get("data", ""),
            "eval": camp.get("eval", ""),
            "notes": note_map.get(cname, ""),
        })

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = {"generated": generated, "headline": HEADLINE, "campaigns": campaigns}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    if not args.no_upload:
        cmd = (f"az storage blob upload --account-name {STORE} "
               f"--container-name {WEB} --file {args.out} --name progress.json --overwrite")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("UPLOAD FAILED:", r.stderr[-800:])
        else:
            print("uploaded progress.json ->", gen_url(), flush=True)

    print("generated", args.out)


def gen_url():
    return f"https://{STORE}.z19.web.core.windows.net/progress.json"


if __name__ == "__main__":
    main()
