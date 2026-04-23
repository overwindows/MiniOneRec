"""
Fetch step-level training histories from W&B and generate training/eval loss curves.
Uses run.history(samples=N) for efficiency and run.scan_history() for full data.
"""
import os, json, warnings
warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = "fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b"

import wandb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

DATA_DIR = os.path.dirname(__file__)
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Load run metadata
with open(os.path.join(DATA_DIR, "wandb_mind_runs_wuchen.json"), "r", encoding="utf-8") as f:
    all_runs_meta = json.load(f)

finished = [r for r in all_runs_meta if r["state"] == "finished"]
print(f"Finished MIND runs: {len(finished)}")

api = wandb.Api(timeout=120)

# ── Classify runs ────────────────────────────────────────────────────────────
def classify(name):
    n = name.lower()
    info = {}
    if "multitask" in n: info["task"] = "multitask"
    elif "ranking" in n: info["task"] = "ranking"
    elif "pointwise" in n: info["task"] = "pointwise"
    elif "rl_" in n: info["task"] = "rl"
    else: info["task"] = "sft_other"
    info["size"] = "large" if "large" in n else ("small" if "small" in n else "unknown")
    if "8b" in n: info["model"] = "8B"
    elif "4b" in n and "reranker" not in n: info["model"] = "4B"
    elif "reranker-4b" in n: info["model"] = "Reranker-4B"
    elif "reranker-0.6b" in n: info["model"] = "Reranker-0.6B"
    elif "base" in n: info["model"] = "1.7B-Base"
    else: info["model"] = "1.7B"
    info["abstract"] = "abstract" in n or "_abs" in n
    info["chat"] = "chat" in n
    return info

for r in finished:
    r["info"] = classify(r["name"])

# ── Fetch histories (with caching) ──────────────────────────────────────────
CACHE_FILE = os.path.join(DATA_DIR, "wandb_mind_histories_v2.json")

if os.path.exists(CACHE_FILE):
    print("Loading cached histories...")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        histories = json.load(f)
else:
    histories = {}

# Only fetch SFT runs (RL handled separately)
sft_runs = [r for r in finished if r["info"]["task"] != "rl"]
to_fetch = [r for r in sft_runs if r["run_id"] not in histories]
print(f"SFT histories to fetch: {len(to_fetch)} (cached: {len(histories)})")

for i, r in enumerate(to_fetch, 1):
    run_path = r["run_path"]
    run_id = r["run_id"]
    print(f"  [{i}/{len(to_fetch)}] {r['name']}...")
    try:
        run = api.run(run_path)
        # Fetch all logged rows (train + eval are in separate rows)
        all_rows = []
        for row in run.scan_history(page_size=10000):
            entry = {}
            entry["_step"] = row.get("_step")
            entry["train_loss"] = row.get("train/loss")
            entry["train_lr"] = row.get("train/learning_rate")
            entry["train_epoch"] = row.get("train/epoch")
            entry["train_grad_norm"] = row.get("train/grad_norm")
            entry["eval_loss"] = row.get("eval/loss")
            entry["global_step"] = row.get("train/global_step")
            # Keep only rows that have at least one metric
            if entry["train_loss"] is not None or entry["eval_loss"] is not None:
                all_rows.append(entry)
        histories[run_id] = all_rows
        print(f"    -> {len(all_rows)} data points")
    except Exception as e:
        print(f"    [ERROR] {e}")
        histories[run_id] = []

# Save cache
with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(histories, f, ensure_ascii=False)
print(f"Cached to {CACHE_FILE}")

# ── Build DataFrames ─────────────────────────────────────────────────────────
train_rows = []
eval_rows = []

for r in sft_runs:
    rid = r["run_id"]
    info = r["info"]
    rows = histories.get(rid, [])
    for row in rows:
        base = {
            "run_id": rid, "name": r["name"],
            "task": info["task"], "model": info["model"],
            "size": info["size"], "abstract": info["abstract"],
            "chat": info["chat"],
        }
        if row.get("train_loss") is not None:
            train_rows.append({
                **base,
                "step": row["global_step"] or row["_step"],
                "epoch": row.get("train_epoch"),
                "loss": row["train_loss"],
                "lr": row.get("train_lr"),
                "grad_norm": row.get("train_grad_norm"),
            })
        if row.get("eval_loss") is not None:
            eval_rows.append({
                **base,
                "step": row["global_step"] or row["_step"],
                "epoch": row.get("train_epoch"),
                "loss": row["eval_loss"],
            })

df_train = pd.DataFrame(train_rows)
df_eval = pd.DataFrame(eval_rows)
print(f"\nTrain data points: {len(df_train)}, Eval data points: {len(df_eval)}")

def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

# ── Helper: short name for legend ────────────────────────────────────────────
def short_name(name):
    """Shorten run name for plot legend."""
    name = name.replace("sft_mind_", "").replace("_small_", " S/").replace("_large_", " L/")
    name = name.replace("Qwen3-", "").replace("_bs1024", "").replace("_bs256", "")
    return name

print("\n=== Generating training curve plots ===")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Training loss curves — Pointwise runs (by step)
# ══════════════════════════════════════════════════════════════════════════════
pw_runs = [r for r in sft_runs if r["info"]["task"] == "pointwise"]
if pw_runs and len(df_train) > 0:
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw = df_train[df_train["run_id"].isin(pw_ids)].copy()
    
    if len(df_pw) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_pw["run_id"].unique():
            sub = df_pw[df_pw["run_id"] == rid].sort_values("step")
            name = short_name(sub["name"].iloc[0])
            # Smooth: rolling mean window
            if len(sub) > 100:
                sub = sub.copy()
                sub["loss_smooth"] = sub["loss"].rolling(window=max(1, len(sub)//100), min_periods=1).mean()
            else:
                sub["loss_smooth"] = sub["loss"]
            ax.plot(sub["step"], sub["loss_smooth"], label=name, alpha=0.85, linewidth=1.3)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Train Loss (smoothed)")
        ax.set_title("Training Loss — Pointwise SFT Runs")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "01_train_loss_pointwise.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Training loss curves — Ranking runs (by step)
# ══════════════════════════════════════════════════════════════════════════════
rk_runs = [r for r in sft_runs if r["info"]["task"] == "ranking"]
if rk_runs and len(df_train) > 0:
    rk_ids = {r["run_id"] for r in rk_runs}
    df_rk = df_train[df_train["run_id"].isin(rk_ids)].copy()
    
    if len(df_rk) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_rk["run_id"].unique():
            sub = df_rk[df_rk["run_id"] == rid].sort_values("step")
            name = short_name(sub["name"].iloc[0])
            if len(sub) > 100:
                sub = sub.copy()
                sub["loss_smooth"] = sub["loss"].rolling(window=max(1, len(sub)//100), min_periods=1).mean()
            else:
                sub["loss_smooth"] = sub["loss"]
            ax.plot(sub["step"], sub["loss_smooth"], label=name, alpha=0.85, linewidth=1.3)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Train Loss (smoothed)")
        ax.set_title("Training Loss — Ranking SFT Runs")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "02_train_loss_ranking.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Eval loss curves — Pointwise (by step)
# ══════════════════════════════════════════════════════════════════════════════
if len(df_eval) > 0:
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw_eval = df_eval[df_eval["run_id"].isin(pw_ids)].copy()
    if len(df_pw_eval) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_pw_eval["run_id"].unique():
            sub = df_pw_eval[df_pw_eval["run_id"] == rid].sort_values("step")
            name = short_name(sub["name"].iloc[0])
            ax.plot(sub["step"], sub["loss"], marker="o", markersize=5, label=name, alpha=0.85, linewidth=1.5)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Eval Loss")
        ax.set_title("Eval Loss — Pointwise SFT Runs")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        save_fig(fig, "03_eval_loss_pointwise.png")

    # Ranking eval
    rk_ids = {r["run_id"] for r in rk_runs}
    df_rk_eval = df_eval[df_eval["run_id"].isin(rk_ids)].copy()
    if len(df_rk_eval) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_rk_eval["run_id"].unique():
            sub = df_rk_eval[df_rk_eval["run_id"] == rid].sort_values("step")
            name = short_name(sub["name"].iloc[0])
            ax.plot(sub["step"], sub["loss"], marker="o", markersize=5, label=name, alpha=0.85, linewidth=1.5)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Eval Loss")
        ax.set_title("Eval Loss — Ranking SFT Runs")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        save_fig(fig, "04_eval_loss_ranking.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Training loss by epoch (normalized x-axis) — Pointwise
# ══════════════════════════════════════════════════════════════════════════════
if len(df_train) > 0:
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw_ep = df_train[df_train["run_id"].isin(pw_ids) & df_train["epoch"].notna()].copy()
    if len(df_pw_ep) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_pw_ep["run_id"].unique():
            sub = df_pw_ep[df_pw_ep["run_id"] == rid].sort_values("epoch")
            name = short_name(sub["name"].iloc[0])
            if len(sub) > 100:
                sub = sub.copy()
                sub["loss_smooth"] = sub["loss"].rolling(window=max(1, len(sub)//100), min_periods=1).mean()
            else:
                sub["loss_smooth"] = sub["loss"]
            ax.plot(sub["epoch"], sub["loss_smooth"], label=name, alpha=0.85, linewidth=1.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss (smoothed)")
        ax.set_title("Training Loss vs Epoch — Pointwise SFT")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "07_train_loss_by_epoch_pointwise.png")

    # Ranking by epoch
    rk_ids = {r["run_id"] for r in rk_runs}
    df_rk_ep = df_train[df_train["run_id"].isin(rk_ids) & df_train["epoch"].notna()].copy()
    if len(df_rk_ep) > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        for rid in df_rk_ep["run_id"].unique():
            sub = df_rk_ep[df_rk_ep["run_id"] == rid].sort_values("epoch")
            name = short_name(sub["name"].iloc[0])
            if len(sub) > 100:
                sub = sub.copy()
                sub["loss_smooth"] = sub["loss"].rolling(window=max(1, len(sub)//100), min_periods=1).mean()
            else:
                sub["loss_smooth"] = sub["loss"]
            ax.plot(sub["epoch"], sub["loss_smooth"], label=name, alpha=0.85, linewidth=1.3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss (smoothed)")
        ax.set_title("Training Loss vs Epoch — Ranking SFT")
        ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "07b_train_loss_by_epoch_ranking.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Learning rate schedule
# ══════════════════════════════════════════════════════════════════════════════
if len(df_train) > 0 and "lr" in df_train.columns:
    df_lr = df_train[df_train["lr"].notna()].copy()
    if len(df_lr) > 0:
        # Pick a few representative runs
        rep_ids = []
        for task in ["pointwise", "ranking"]:
            task_runs = [r for r in sft_runs if r["info"]["task"] == task]
            if task_runs:
                rep_ids.append(task_runs[0]["run_id"])
        
        if rep_ids:
            fig, ax = plt.subplots(figsize=(14, 5))
            for rid in rep_ids:
                sub = df_lr[df_lr["run_id"] == rid].sort_values("step")
                if len(sub) > 0:
                    name = short_name(sub["name"].iloc[0])
                    ax.plot(sub["step"], sub["lr"], label=name, linewidth=1.2)
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule")
            ax.legend(fontsize=7)
            save_fig(fig, "11_lr_schedule.png")

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Gradient norm over training
# ══════════════════════════════════════════════════════════════════════════════
if len(df_train) > 0 and "grad_norm" in df_train.columns:
    df_gn = df_train[df_train["grad_norm"].notna()].copy()
    pw_ids = {r["run_id"] for r in pw_runs}
    df_gn_pw = df_gn[df_gn["run_id"].isin(pw_ids)]
    if len(df_gn_pw) > 0:
        fig, ax = plt.subplots(figsize=(16, 6))
        for rid in df_gn_pw["run_id"].unique():
            sub = df_gn_pw[df_gn_pw["run_id"] == rid].sort_values("step")
            name = short_name(sub["name"].iloc[0])
            if len(sub) > 100:
                sub = sub.copy()
                sub["gn_smooth"] = sub["grad_norm"].rolling(window=max(1, len(sub)//50), min_periods=1).mean()
            else:
                sub["gn_smooth"] = sub["grad_norm"]
            ax.plot(sub["step"], sub["gn_smooth"], label=name, alpha=0.8, linewidth=1.0)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Gradient Norm (smoothed)")
        ax.set_title("Gradient Norm — Pointwise SFT Runs")
        ax.legend(fontsize=6, loc="upper right", ncol=1)
        ax.set_yscale("log")
        save_fig(fig, "12_grad_norm_pointwise.png")

print("\n=== Done! ===")
for f in sorted(os.listdir(PLOT_DIR)):
    if f.endswith(".png"):
        print(f"  {f}")
