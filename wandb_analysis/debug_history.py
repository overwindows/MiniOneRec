"""Debug: check what keys are actually in W&B run histories."""
import os
os.environ["WANDB_API_KEY"] = "fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b"
import wandb

api = wandb.Api(timeout=120)

# Pick a known finished run with good data
test_paths = [
    "wuchen/MIND/pointwise_1.7B_Base_large",  # might not work - use run_path
]

# Use actual run paths from our JSON
import json
with open("wandb_analysis/wandb_mind_runs_wuchen.json", "r") as f:
    runs = json.load(f)

finished = [r for r in runs if r["state"] == "finished" and r.get("eval/loss")]
r = finished[0]
print(f"Testing run: {r['name']} ({r['run_path']})")

run = api.run(r["run_path"])

# Method 1: run.history()
print("\n--- run.history() ---")
h = run.history(samples=20)
print(f"Shape: {h.shape}")
print(f"Columns: {list(h.columns)}")
if len(h) > 0:
    print(h.head(5).to_string())

# Method 2: scan_history without key filter
print("\n--- scan_history (no key filter, first 5 rows) ---")
count = 0
for row in run.scan_history(page_size=5):
    print(f"  keys: {list(row.keys())}")
    print(f"  sample: { {k: row[k] for k in list(row.keys())[:10]} }")
    count += 1
    if count >= 3:
        break
