"""Match W&B runs to EXPERIMENT_PLAN.md experiment IDs and output the mapping."""
import json

with open("wandb_analysis/wandb_mind_runs_wuchen.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

# Build lookup: name -> list of {url, state, created_at, run_id, project}
by_name = {}
for r in runs:
    by_name.setdefault(r["name"], []).append(r)

# Define experiment -> expected run name pattern mapping
# For each exp, we list possible W&B run names (exact or substring)
EXP_MAP = {
    # Phase 1: MINDsmall pointwise
    "P1.B": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_chat"],
    "P1.1": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg3.0_hist30_chat"],
    "P1.2": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat"],
    "P1.3": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat"],
    "P1.4": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist50_chat"],
    "P1.5": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat"],
    "P1.7": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_subcat_chat"],

    # Phase 1L: MINDlarge pointwise
    "L1.1": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg3.0_hist30_chat"],
    "L1.2": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_chat"],
    "L1.3": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist30_abstract_chat"],
    "L1.4": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep5_neg2.0_hist50_chat"],
    "L1.5": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg3.0_hist50_chat"],
    "L1.6": ["sft_mind_pointwise_large_Qwen3-8B_bs256_ep5_neg2.0_hist30_chat"],
    "L1.7": ["pointwise_1.7B_Base_large"],
    "L1.8": ["sft_mind_pointwise_large_Qwen3-1.7B_bs256_ep7_neg2.0_hist30_abstract_chat"],
    "L1.9": ["sft_mind_pointwise_large_Qwen3-1.7B-Base_bs256_ep5_neg2.0_hist30_abstract"],

    # Older MIND project naming (small data, pre-pipeline)
    "P1.B_old": ["pointwise_1.7B_small_NEW"],
    "L1.7_old": ["pointwise_1.7B_large"],

    # Multi-task
    "M4.0": ["sft_mind_multitask_small_Qwen3-1.7B_bs256_ep3_pw0.7_chat"],

    # RL
    "R_ndcg": ["rl_mind_small_mind_ndcg"],
    "R_pw_weighted": ["rl_mind_pointwise_small_pointwise_weighted"],

    # CoT RL
    "A5.1_cot": ["simple_cot_mind_cot_binary_cand10_gen4_bs32_lr1e-7_kl0.5_resp256"],

    # Ranking (older, from MIND project)
    "RK_small_NEW": ["ranking_1.7B_small_NEW"],
    "RK_small": ["ranking_1.7B_small"],

    # Additional small pointwise runs (MIND project, older naming)
    "PW_base_small_neg1": ["sft_mind_pointwise_small_Qwen3-1.7B-Base_bs1024_ep3_neg1.0"],
    "PW_base_small_abs": ["sft_mind_pointwise_small_Qwen3-1.7B-Base_bs1024_ep3_neg1.0_abs"],
    "PW_small_neg1": ["sft_mind_pointwise_small_Qwen3-1.7B_bs1024_ep3_neg1.0"],

    # Ranking models with various configs
    "RK_base_neg8_hist50": ["sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3_neg8.0_hist50"],
    "RK_base_neg10_hist50": ["sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3_neg10.0_hist50"],
    "RK_base_neg6_hist50": ["sft_mind_ranking_small_Qwen3-1.7B-Base_bs1024_ep3_neg6.0_hist50"],
    "RK_1.7B_neg8_hist50": ["sft_mind_ranking_small_Qwen3-1.7B_bs1024_ep3_neg8.0_hist50"],
    "RK_1.7B_neg6_hist50": ["sft_mind_ranking_small_Qwen3-1.7B_bs1024_ep3_neg6.0_hist50"],

    # Early pointwise runs
    "PW_ep3_chat_1": ["sft_mind_pointwise_Qwen3-1.7B_bs256_ep3_neg2.0_hist30_chat"],
    "PW_small_ep3_chat": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep3_neg2.0_hist30_chat"],
    "PW_small_ep3_chat_old": ["sft_mind_pointwise_small_Qwen3-1.7B_bs256_ep3_neg2.0_hist30_chat"],
}

print("=" * 120)
print(f"{'Exp ID':<25} {'State':<10} {'W&B Run Name':<70} {'URL'}")
print("=" * 120)

matched_urls = {}

for exp_id, patterns in EXP_MAP.items():
    found = []
    for pat in patterns:
        if pat in by_name:
            found.extend(by_name[pat])
    
    if not found:
        # Try substring match
        for pat in patterns:
            for name, rlist in by_name.items():
                if pat in name:
                    found.extend(rlist)
    
    if found:
        # Pick the best: prefer finished, then most recent
        finished = [r for r in found if r["state"] == "finished"]
        if finished:
            best = sorted(finished, key=lambda r: r["created_at"], reverse=True)[0]
        else:
            best = sorted(found, key=lambda r: r["created_at"], reverse=True)[0]
        
        matched_urls[exp_id] = best["url"]
        # Show all runs for this experiment
        for r in sorted(found, key=lambda r: r["created_at"], reverse=True):
            marker = " <-- BEST" if r["run_id"] == best["run_id"] else ""
            print(f"{exp_id:<25} {r['state']:<10} {r['name']:<70} {r['url']}{marker}")
    else:
        print(f"{exp_id:<25} {'N/A':<10} {'(no matching W&B run found)':<70}")

print("\n" + "=" * 120)
print("\nMAPPING FOR EXPERIMENT_PLAN.md (copy-paste ready):")
print("=" * 120)
for exp_id, url in sorted(matched_urls.items()):
    print(f"  {exp_id}: {url}")

# Save as JSON for programmatic use
with open("wandb_analysis/exp_to_wandb_mapping.json", "w", encoding="utf-8") as f:
    json.dump(matched_urls, f, indent=2, ensure_ascii=False)
print(f"\nSaved mapping to wandb_analysis/exp_to_wandb_mapping.json")
