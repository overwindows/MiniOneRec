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
        "name": "JEV-4B frozen-backbone readout recovery",
        "motivation": "Full-model decision SFT remains below frozen JEV. Test a small residual readout "
                      "without changing pretrained representations; prior ranking drift is observed, "
                      "but its cause is not established.",
        "settings": "Pilot source bceaab345e8e116810dc6292b7d8a189d7d1c452. Exact conditional/causal/"
                    "collated-loss parity on four real prompts. Frozen backbone; 16383 training pairs; "
                    "binary_l2_0.01 selected solely on 256 excluded training impressions. "
                    "Confirmation source a88c920ebc4c232a51df266bbb4dfc44ec477ab5; "
                    "same checksum-pinned head, no fitting or reselection. "
                    "One Standard ranking A100; six-hour timeout; no new full-model training.",
        "data": "MIND_small, legacy prompt, hist30, abstracts, prompt budget4095. "
                "Pilot: dev5001-6000,1000 impressions/36802 candidates. "
                "Fixed-head confirmation: fresh dev6001-11000,5000 impressions/187881 candidates. "
                "Neither is the locked first5000 comparison; do not pool the two intervals.",
        "eval": "Verified pilot AUC 0.63367586 versus matched frozen 0.61530684, "
                "delta +0.01836902; 95% paired CI [+0.00029635,+0.03614217] only narrowly clears zero. "
                "MRR/nDCG point values improve but their intervals include zero. "
                "All raw coverage/hashes/eight metrics/paired intervals independently rebuilt. "
                "Promising pilot, not a robust or SOTA claim. Confirmation submitted; no result yet. "
                "Initial wrong-VC queued submission was canceled.",
        "runs": [
            ("great_oven_88h8j3jgjl", "Fixed-head fresh5000 confirmation parent", ""),
            ("ae73f51f-9d3c-496e-a402-8ffb37f0c5ed", "Fixed-head confirmation worker", ""),
            ("ivory_star_h4z4j4tn84", "Frozen-backbone readout pilot dev5001-6000", "0.63367586"),
            ("240ffc68-83e6-4289-985b-48ab9b966045", "One-A100 readout pilot worker", ""),
            ("teal_hamster_1nnbmw8xy9", "Canceled initial wrong-VC parent", ""),
            ("4778df36-1bc5-42b6-b61f-418d51d448c3", "Initial wrong-VC worker; cancellation requested", ""),
        ],
    },
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
        "name": "JEV-4B matched frozen prompt controls",
        "motivation": "Separate prompt effects from weight adaptation before evaluating the candidate-only loss.",
        "settings": "Same frozen weights, hist30, abstracts, 4095 prompt-token budget, one ranking Standard A100; "
                    "immutable source 10b4379407d11e3e3a00c6cae3a75042eeaf731c",
        "data": "Same first 5000 MIND_small dev impressions / 187009 candidates; data hashes and raw labels checked",
        "eval": "Legacy AUC 0.63623517; native-click AUC 0.63313036. Paired AUC delta -0.00310481, "
                "95% impression-bootstrap interval [-0.00711694, +0.00112362] includes zero "
                "(2048 resamples, seed42; no multiplicity adjustment). No prompt-only gain established. "
                "All four metrics independently reconstructed from raw scores.",
        "runs": [
            ("bright_ear_vnr75xb8tw", "Frozen legacy prompt dev@5000", "0.63623517"),
            ("bright_ear_vnr75xb8tw", "Frozen native-click prompt dev@5000", "0.63313036"),
        ],
    },
    {
        "name": "JEV-4B full-data candidate-only loss ablation",
        "motivation": "Isolate the loss change after frozen controls found no native-prompt AUC gain.",
        "settings": "decision BCE(z_A-z_B) + legacy prompt; no KL; seed42; lr1e-4; "
                    "global batch128/micro2; three epochs; bf16/ZeRO-2; one ranking Standard "
                    "eight-A100 node; 72-hour command timeout; immutable source ec86d37eb4b6",
        "data": "All 469264 deterministic sampled MIND_small training pairs; hist30, abstracts, "
                "cutoff4096; full audit passed without truncation",
        "eval": "Training Completed: all 11001 steps/3 epochs, 44.47 training hours; final checkpoint "
                "metadata/tokenizer/427-tensor extent verified. Fresh matched old full-vocabulary "
                "AUC 0.54519484 independently reconstructed with full raw-score coverage and input hashes. "
                "New decision AUC 0.56365540: +0.01846056 versus old SFT "
                "(95% paired CI [+0.00735828,+0.02897442]), but -0.07257978 versus matched frozen "
                "(CI [-0.08375560,-0.06140528]). All 5000/187009 coverage, hashes, four metrics "
                "and paired intervals independently verified; one training seed, no multiplicity adjustment. "
                "Partial recovery, not successful adaptation or SOTA; downstream campaigns remain gated.",
        "runs": [
            ("mighty_milk_0znks9vwg7", "Full decision-loss + legacy-prompt training", ""),
            ("2c74fe4d-5b03-4df1-b2a3-502003a390eb", "Full decision-loss training worker", ""),
            ("purple_flag_yjyhnhtpyn", "Old full-vocabulary matched legacy dev@5000", "0.54519484"),
            ("helpful_lizard_77ghb5kb5b", "New decision-loss matched legacy dev@5000", "0.56365540"),
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
    "Positive frozen-readout pilot on new dev5001-6000: AUC 0.61530684 -> 0.63367586, "
    "+0.01836902 (95% paired CI [+0.00029635,+0.03614217]); narrow pilot evidence, not SOTA. "
    "Unchanged-head confirmation submitted on fresh dev6001-11000: no retraining/retuning, "
    "one ranking A100, six-hour cap; no confirmation result yet. "
    "Recorded competition target: 0.7326; best submitted ensemble: 0.7144. "
    "Matched frozen controls completed 2026-10-20: legacy AUC 0.6362 vs native-click 0.6331; "
    "paired AUC interval includes zero, so no prompt-only improvement is established. "
    "Full input audit passed all 469264 training pairs and 187009 dev candidates without truncation. "
    "Full-data decision-loss + legacy-prompt experiment Completed and independently verified: "
    "decision AUC 0.56365540 versus old SFT 0.54519484 and matched frozen 0.63623517. "
    "Decision-minus-old +0.01846056 (95% paired CI [+0.00735828,+0.02897442]); "
    "still below frozen by 0.07257978. Partial recovery, not successful adaptation or SOTA. "
    "One training seed; impression-bootstrap intervals unadjusted; downstream campaigns remain gated."
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
