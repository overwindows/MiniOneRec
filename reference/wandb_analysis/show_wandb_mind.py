import json

with open("wandb_mind_runs_wuchen.json", "r", encoding="utf-8") as f:
    runs = json.load(f)

finished = [r for r in runs if r["state"] == "finished"]
print(f"Finished runs: {len(finished)} / {len(runs)} total\n")

if finished:
    print("Available columns:", list(finished[0].keys()))
    print()

std = {"entity","project","run_id","run_path","url","name","state","created_at","heartbeat_at","tags","group","job_type","notes","config"}

for i, r in enumerate(finished, 1):
    metrics = {k: v for k, v in r.items() if k not in std and v not in [None, "", 0]}
    print(f"--- #{i}  {r['name']}  ({r['project']})  [{r['created_at']}]  state={r['state']}")
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")
    else:
        print("    (no summary metrics)")

    cfg = json.loads(r.get("config", "{}"))
    keys_of_interest = [
        "model_name_or_path", "num_train_epochs", "per_device_train_batch_size",
        "learning_rate", "neg_ratio", "max_history", "use_abstract",
        "use_chat_template", "data_root", "task_type",
    ]
    interesting = {k: cfg[k] for k in keys_of_interest if k in cfg}
    if interesting:
        print(f"    config: {interesting}")
    print()
