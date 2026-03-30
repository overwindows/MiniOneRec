"""
Comprehensive W&B MIND experiment analysis:
  1. Fetch step-level training history for all finished runs
  2. Plot training loss curves (grouped by task type, data size, model)
  3. Plot eval loss curves
  4. Bar chart comparison of final metrics
  5. Save all plots to wandb_analysis/plots/
"""
import os, json, sys
import warnings
warnings.filterwarnings("ignore")

os.environ["WANDB_API_KEY"] = "fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b"

import wandb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

DATA_DIR = os.path.dirname(__file__)

# ── 1. Load run metadata ────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, "wandb_mind_runs_wuchen.json"), "r", encoding="utf-8") as f:
    all_runs_meta = json.load(f)

finished = [r for r in all_runs_meta if r["state"] == "finished"]
print(f"Finished MIND runs: {len(finished)}")

api = wandb.Api(timeout=120)

# ── 2. Classify runs ────────────────────────────────────────────────────────
def classify(name, config_str):
    cfg = json.loads(config_str) if config_str else {}
    info = {}
    n = name.lower()
    # Task type
    if "multitask" in n:
        info["task"] = "multitask"
    elif "ranking" in n:
        info["task"] = "ranking"
    elif "pointwise" in n:
        info["task"] = "pointwise"
    elif "rl_" in n or "rl " in n:
        info["task"] = "rl"
    else:
        info["task"] = "sft_other"
    # Data size
    if "large" in n:
        info["size"] = "large"
    elif "small" in n:
        info["size"] = "small"
    else:
        info["size"] = "unknown"
    # Model
    if "8b" in n.lower():
        info["model"] = "8B"
    elif "4b" in n.lower():
        info["model"] = "4B"
    elif "reranker-4b" in n.lower():
        info["model"] = "Reranker-4B"
    elif "reranker-0.6b" in n.lower():
        info["model"] = "Reranker-0.6B"
    elif "base" in n.lower():
        info["model"] = "1.7B-Base"
    elif "1.7b" in n.lower():
        info["model"] = "1.7B"
    else:
        info["model"] = "1.7B"  # default
    # Epochs
    info["epochs"] = cfg.get("num_train_epochs", "?")
    # Abstract
    info["abstract"] = "abstract" in n
    # Chat template
    info["chat"] = "chat" in n
    # Short label for legend
    parts = []
    parts.append(info["task"])
    parts.append(info["model"])
    parts.append(info["size"])
    if info["abstract"]:
        parts.append("abs")
    if info["chat"]:
        parts.append("chat")
    parts.append(f"ep{info['epochs']}")
    info["label"] = "_".join(parts)
    return info


for r in finished:
    r["info"] = classify(r["name"], r.get("config", "{}"))

# ── 3. Fetch step-level histories ───────────────────────────────────────────
HISTORY_CACHE = os.path.join(DATA_DIR, "wandb_mind_histories.json")

if os.path.exists(HISTORY_CACHE):
    print("Loading cached histories...")
    with open(HISTORY_CACHE, "r", encoding="utf-8") as f:
        histories = json.load(f)
else:
    histories = {}

to_fetch = [r for r in finished if r["run_id"] not in histories]
print(f"Histories to fetch: {len(to_fetch)} (cached: {len(histories)})")

for i, r in enumerate(to_fetch, 1):
    run_path = r["run_path"]
    run_id = r["run_id"]
    print(f"  [{i}/{len(to_fetch)}] Fetching {r['name']}...")
    try:
        run = api.run(run_path)
        # Get train/loss and eval/loss over steps
        hist = run.scan_history(
            keys=["train/loss", "eval/loss", "train/epoch", "_step"],
            page_size=10000,
        )
        rows = []
        for row in hist:
            rows.append({
                "step": row.get("_step"),
                "train_loss": row.get("train/loss"),
                "eval_loss": row.get("eval/loss"),
                "epoch": row.get("train/epoch"),
            })
        histories[run_id] = rows
        print(f"    -> {len(rows)} data points")
    except Exception as e:
        print(f"    [ERROR] {e}")
        histories[run_id] = []

# Save cache
with open(HISTORY_CACHE, "w", encoding="utf-8") as f:
    json.dump(histories, f, ensure_ascii=False)
print(f"Histories cached to {HISTORY_CACHE}")

# ── 4. Build DataFrames ─────────────────────────────────────────────────────
all_train = []
all_eval = []

for r in finished:
    rid = r["run_id"]
    info = r["info"]
    label = info["label"]
    rows = histories.get(rid, [])
    for row in rows:
        if row.get("train_loss") is not None:
            all_train.append({
                "run_id": rid,
                "name": r["name"],
                "label": label,
                "task": info["task"],
                "model": info["model"],
                "size": info["size"],
                "step": row["step"],
                "epoch": row.get("epoch"),
                "train_loss": row["train_loss"],
            })
        if row.get("eval_loss") is not None:
            all_eval.append({
                "run_id": rid,
                "name": r["name"],
                "label": label,
                "task": info["task"],
                "model": info["model"],
                "size": info["size"],
                "step": row["step"],
                "epoch": row.get("epoch"),
                "eval_loss": row["eval_loss"],
            })

df_train = pd.DataFrame(all_train)
df_eval = pd.DataFrame(all_eval)
print(f"\nTrain data points: {len(df_train)}, Eval data points: {len(df_eval)}")

# ── Helper ───────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

# ── 5. PLOT: Training loss curves — Pointwise runs ─────────────────────────
print("\n=== Generating plots ===")

# 5a. Pointwise SFT runs (1.7B variants, small + large)
pw_runs = [r for r in finished if r["info"]["task"] == "pointwise" and r["info"]["model"] in ("1.7B", "1.7B-Base")]
if pw_runs and len(df_train) > 0:
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw = df_train[df_train["run_id"].isin(pw_ids)].copy()
    
    if len(df_pw) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        for rid in df_pw["run_id"].unique():
            sub = df_pw[df_pw["run_id"] == rid].sort_values("step")
            lbl = sub["label"].iloc[0]
            name = sub["name"].iloc[0]
            # Downsample for readability
            if len(sub) > 500:
                sub = sub.iloc[::max(1, len(sub)//500)]
            ax.plot(sub["step"], sub["train_loss"], label=name, alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Train Loss")
        ax.set_title("Training Loss — Pointwise SFT (1.7B / 1.7B-Base)")
        ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "01_train_loss_pointwise.png")

# 5b. Ranking SFT runs
rk_runs = [r for r in finished if r["info"]["task"] == "ranking"]
if rk_runs and len(df_train) > 0:
    rk_ids = {r["run_id"] for r in rk_runs}
    df_rk = df_train[df_train["run_id"].isin(rk_ids)].copy()
    
    if len(df_rk) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        for rid in df_rk["run_id"].unique():
            sub = df_rk[df_rk["run_id"] == rid].sort_values("step")
            name = sub["name"].iloc[0]
            if len(sub) > 500:
                sub = sub.iloc[::max(1, len(sub)//500)]
            ax.plot(sub["step"], sub["train_loss"], label=name, alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Train Loss")
        ax.set_title("Training Loss — Ranking SFT")
        ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "02_train_loss_ranking.png")

# ── 6. PLOT: Eval loss curves ───────────────────────────────────────────────
if len(df_eval) > 0:
    # Pointwise eval
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw_eval = df_eval[df_eval["run_id"].isin(pw_ids)].copy()
    if len(df_pw_eval) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        for rid in df_pw_eval["run_id"].unique():
            sub = df_pw_eval[df_pw_eval["run_id"] == rid].sort_values("step")
            name = sub["name"].iloc[0]
            ax.plot(sub["step"], sub["eval_loss"], marker="o", markersize=4, label=name, alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Eval Loss")
        ax.set_title("Eval Loss — Pointwise SFT (1.7B / 1.7B-Base)")
        ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
        save_fig(fig, "03_eval_loss_pointwise.png")

    # Ranking eval
    rk_ids = {r["run_id"] for r in rk_runs}
    df_rk_eval = df_eval[df_eval["run_id"].isin(rk_ids)].copy()
    if len(df_rk_eval) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        for rid in df_rk_eval["run_id"].unique():
            sub = df_rk_eval[df_rk_eval["run_id"] == rid].sort_values("step")
            name = sub["name"].iloc[0]
            ax.plot(sub["step"], sub["eval_loss"], marker="o", markersize=4, label=name, alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Eval Loss")
        ax.set_title("Eval Loss — Ranking SFT")
        ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
        save_fig(fig, "04_eval_loss_ranking.png")

# ── 7. PLOT: Final eval loss bar chart comparison ───────────────────────────
# Build summary table of finished runs
summary_rows = []
for r in finished:
    cfg = json.loads(r.get("config", "{}"))
    row = {
        "name": r["name"],
        "task": r["info"]["task"],
        "model": r["info"]["model"],
        "size": r["info"]["size"],
        "epochs": r["info"]["epochs"],
        "abstract": r["info"]["abstract"],
        "chat": r["info"]["chat"],
        "eval_loss": r.get("eval/loss"),
        "train_loss_final": r.get("train/loss"),
        "train_loss_avg": r.get("train_loss"),
        "lr": cfg.get("learning_rate"),
        "neg_ratio": cfg.get("neg_ratio"),
        "max_history": cfg.get("max_history"),
    }
    # RL metrics
    if "val-core/mind/acc/mean@1" in r:
        row["acc@1"] = r["val-core/mind/acc/mean@1"]
    if "val-core/mind_pointwise/acc/mean@1" in r:
        row["acc@1"] = r["val-core/mind_pointwise/acc/mean@1"]
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(DATA_DIR, "wandb_mind_summary.csv"), index=False)
print(f"\nSummary table saved ({len(df_summary)} runs)")

# Bar chart: eval loss by run (pointwise only, sorted)
df_pw_summary = df_summary[(df_summary["task"] == "pointwise") & df_summary["eval_loss"].notna()].copy()
df_pw_summary = df_pw_summary.sort_values("eval_loss")

if len(df_pw_summary) > 0:
    fig, ax = plt.subplots(figsize=(14, max(6, len(df_pw_summary) * 0.45)))
    colors = []
    for _, row in df_pw_summary.iterrows():
        if row["size"] == "large":
            colors.append("#2196F3")
        elif row["model"] == "1.7B-Base":
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")
    bars = ax.barh(range(len(df_pw_summary)), df_pw_summary["eval_loss"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_pw_summary)))
    ax.set_yticklabels(df_pw_summary["name"], fontsize=7)
    ax.set_xlabel("Eval Loss (lower is better)")
    ax.set_title("Final Eval Loss — Pointwise SFT Runs\n(Blue=large, Orange=1.7B-Base, Green=1.7B)")
    # Add value labels
    for i, (_, row) in enumerate(df_pw_summary.iterrows()):
        ax.text(row["eval_loss"] + 0.002, i, f'{row["eval_loss"]:.4f}', va="center", fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, "05_eval_loss_bar_pointwise.png")

# Bar chart: ranking runs
df_rk_summary = df_summary[(df_summary["task"] == "ranking") & df_summary["eval_loss"].notna()].copy()
df_rk_summary = df_rk_summary.sort_values("eval_loss")

if len(df_rk_summary) > 0:
    fig, ax = plt.subplots(figsize=(14, max(6, len(df_rk_summary) * 0.45)))
    colors = []
    for _, row in df_rk_summary.iterrows():
        if "4b" in row["model"].lower() or "4B" in row["model"]:
            colors.append("#9C27B0")
        elif "reranker" in row["model"].lower():
            colors.append("#E91E63")
        elif row["model"] == "1.7B-Base":
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")
    bars = ax.barh(range(len(df_rk_summary)), df_rk_summary["eval_loss"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df_rk_summary)))
    ax.set_yticklabels(df_rk_summary["name"], fontsize=7)
    ax.set_xlabel("Eval Loss (lower is better)")
    ax.set_title("Final Eval Loss — Ranking SFT Runs\n(Green=1.7B, Orange=1.7B-Base, Purple=4B, Pink=Reranker)")
    for i, (_, row) in enumerate(df_rk_summary.iterrows()):
        ax.text(row["eval_loss"] + 0.01, i, f'{row["eval_loss"]:.4f}', va="center", fontsize=7)
    ax.invert_yaxis()
    save_fig(fig, "06_eval_loss_bar_ranking.png")

# ── 8. PLOT: Training loss by epoch (normalized x-axis) ────────────────────
# Pointwise with epoch-normalized x
if len(df_train) > 0:
    pw_ids = {r["run_id"] for r in pw_runs}
    df_pw_ep = df_train[df_train["run_id"].isin(pw_ids) & df_train["epoch"].notna()].copy()
    
    if len(df_pw_ep) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        for rid in df_pw_ep["run_id"].unique():
            sub = df_pw_ep[df_pw_ep["run_id"] == rid].sort_values("epoch")
            name = sub["name"].iloc[0]
            if len(sub) > 500:
                sub = sub.iloc[::max(1, len(sub)//500)]
            ax.plot(sub["epoch"], sub["train_loss"], label=name, alpha=0.8, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train Loss")
        ax.set_title("Training Loss vs. Epoch — Pointwise SFT (1.7B / 1.7B-Base)")
        ax.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)
        ax.set_ylim(bottom=0)
        save_fig(fig, "07_train_loss_by_epoch_pointwise.png")

# ── 9. PLOT: Effect of hyperparameters on eval loss ─────────────────────────
# Scatterplot: lr vs eval_loss
df_with_lr = df_summary[df_summary["lr"].notna() & df_summary["eval_loss"].notna()].copy()
if len(df_with_lr) > 3:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # lr vs eval_loss
    ax = axes[0]
    for task in df_with_lr["task"].unique():
        sub = df_with_lr[df_with_lr["task"] == task]
        ax.scatter(sub["lr"], sub["eval_loss"], label=task, s=60, alpha=0.7)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Learning Rate vs Eval Loss")
    ax.set_xscale("log")
    ax.legend()
    
    # epochs vs eval_loss
    ax = axes[1]
    df_ep = df_summary[df_summary["eval_loss"].notna() & (df_summary["epochs"] != "?")].copy()
    df_ep["epochs"] = df_ep["epochs"].astype(float)
    for task in df_ep["task"].unique():
        sub = df_ep[df_ep["task"] == task]
        ax.scatter(sub["epochs"], sub["eval_loss"], label=task, s=60, alpha=0.7)
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Epochs vs Eval Loss")
    ax.legend()
    
    # model vs eval_loss (box plot)
    ax = axes[2]
    df_box = df_summary[df_summary["eval_loss"].notna()].copy()
    models_order = ["1.7B", "1.7B-Base", "Reranker-0.6B", "4B", "Reranker-4B"]
    models_present = [m for m in models_order if m in df_box["model"].values]
    if models_present:
        df_box_filtered = df_box[df_box["model"].isin(models_present)]
        sns.boxplot(data=df_box_filtered, x="model", y="eval_loss", order=models_present, ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Eval Loss")
        ax.set_title("Model vs Eval Loss")
        ax.tick_params(axis="x", rotation=30)
    
    fig.suptitle("Hyperparameter Impact on Eval Loss", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, "08_hyperparam_analysis.png")

# ── 10. PLOT: RL runs ───────────────────────────────────────────────────────
rl_runs = [r for r in finished if r["info"]["task"] == "rl"]
if rl_runs:
    rl_ids = {r["run_id"] for r in rl_runs}
    # Check if we have any RL-specific metrics in histories
    rl_data = []
    for r in rl_runs:
        rid = r["run_id"]
        rows = histories.get(rid, [])
        has_rl = any(row.get("train_loss") is not None for row in rows)
        rl_data.append({"name": r["name"], "run_id": rid, "points": len(rows), "has_train": has_rl})
    
    # For RL, also try to fetch reward/acc metrics
    rl_hist_rows = []
    for r in rl_runs:
        rid = r["run_id"]
        try:
            run = api.run(r["run_path"])
            hist = run.scan_history(page_size=10000)
            for row in hist:
                step = row.get("_step", 0)
                entry = {"run_id": rid, "name": r["name"], "step": step}
                for k, v in row.items():
                    if v is not None and isinstance(v, (int, float)):
                        entry[k] = v
                rl_hist_rows.append(entry)
        except:
            pass
    
    if rl_hist_rows:
        df_rl = pd.DataFrame(rl_hist_rows)
        # Find interesting columns
        metric_cols = [c for c in df_rl.columns if any(x in c.lower() for x in ["reward", "acc", "loss", "kl"])]
        metric_cols = [c for c in metric_cols if df_rl[c].notna().sum() > 2]
        
        if metric_cols:
            n_cols = min(len(metric_cols), 4)
            fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
            if n_cols == 1:
                axes = [axes]
            for idx, col in enumerate(metric_cols[:n_cols]):
                ax = axes[idx]
                for rid in df_rl["run_id"].unique():
                    sub = df_rl[(df_rl["run_id"] == rid) & df_rl[col].notna()].sort_values("step")
                    if len(sub) > 0:
                        name = sub["name"].iloc[0]
                        ax.plot(sub["step"], sub[col], label=name, alpha=0.8, linewidth=1.2)
                ax.set_xlabel("Step")
                ax.set_ylabel(col)
                ax.set_title(col)
                ax.legend(fontsize=6)
            fig.suptitle("RL Training Metrics", fontsize=14)
            fig.tight_layout()
            save_fig(fig, "09_rl_metrics.png")

# ── 11. PLOT: Small vs Large data comparison ────────────────────────────────
df_size = df_summary[(df_summary["task"] == "pointwise") & df_summary["eval_loss"].notna()].copy()
df_small = df_size[df_size["size"] == "small"]
df_large = df_size[df_size["size"] == "large"]

if len(df_small) > 0 and len(df_large) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        [df_small["eval_loss"].dropna(), df_large["eval_loss"].dropna()],
        labels=["MIND Small", "MIND Large"],
        widths=0.5,
    )
    # Overlay individual points
    for i, (df_sub, label) in enumerate([(df_small, "small"), (df_large, "large")], 1):
        jitter = (pd.Series(range(len(df_sub))) - len(df_sub)/2) * 0.01
        ax.scatter([i] * len(df_sub), df_sub["eval_loss"], alpha=0.6, s=40, zorder=5)
        for _, row in df_sub.iterrows():
            ax.annotate(row["name"].split("_")[-1] if "_" in row["name"] else row["name"],
                       (i + 0.15, row["eval_loss"]), fontsize=5, alpha=0.7)
    ax.set_ylabel("Eval Loss")
    ax.set_title("Pointwise SFT: Small vs Large Data")
    save_fig(fig, "10_small_vs_large.png")

print("\n=== All plots generated! ===")
print(f"Output directory: {PLOT_DIR}")
for f in sorted(os.listdir(PLOT_DIR)):
    print(f"  {f}")
