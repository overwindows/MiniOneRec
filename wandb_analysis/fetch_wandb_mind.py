"""Fetch all MIND-related W&B runs for entity wuchen across all projects."""
import os, json, csv, wandb

os.environ["WANDB_API_KEY"] = "fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b"

api = wandb.Api(timeout=120)

ENTITIES = ["wuchen", "devdivchina"]
KEYWORD = "mind"

all_runs = []
total_projects = 0
total_runs = 0

for entity in ENTITIES:
    try:
        projects = api.projects(entity=entity)
    except Exception as e:
        print(f"[WARN] Could not list projects for entity '{entity}': {e}")
        continue

    for proj in projects:
        pname = proj.name
        total_projects += 1
        try:
            runs = api.runs(f"{entity}/{pname}", per_page=300)
        except Exception as e:
            print(f"  [WARN] Could not list runs for {entity}/{pname}: {e}")
            continue

        for run in runs:
            total_runs += 1
            # Check if MIND-related
            searchable = " ".join([
                str(run.name or ""),
                str(getattr(run, "display_name", "") or ""),
                str(run.group or ""),
                str(run.job_type or ""),
                str(run.notes or ""),
                str(run.tags or ""),
                pname,
            ]).lower()

            if KEYWORD not in searchable:
                continue

            # Extract summary metrics of interest
            summary = dict(run.summary) if run.summary else {}
            metric_keys = [k for k in summary if any(
                m in k.lower() for m in ["acc", "ndcg", "mrr", "auc", "loss", "hit", "recall", "precision", "map", "f1"]
            )]
            metrics = {k: summary[k] for k in sorted(metric_keys) if not isinstance(summary[k], (dict, list))}

            # Extract key config items
            config = dict(run.config) if run.config else {}
            config_keys = [k for k in config if not k.startswith("_")]
            config_clean = {}
            for k in sorted(config_keys):
                v = config[k]
                if not isinstance(v, (dict, list)):
                    config_clean[k] = v

            row = {
                "entity": entity,
                "project": pname,
                "run_id": run.id,
                "run_path": "/".join(run.path),
                "url": run.url,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "heartbeat_at": getattr(run, "heartbeat_at", ""),
                "tags": ", ".join(run.tags) if run.tags else "",
                "group": run.group or "",
                "job_type": run.job_type or "",
                "notes": (run.notes or "")[:200],
                **metrics,
                "config": json.dumps(config_clean, ensure_ascii=False),
            }
            all_runs.append(row)

        mind_count_in_proj = sum(1 for r in all_runs if r["project"] == pname and r["entity"] == entity)
        if mind_count_in_proj > 0:
            print(f"  {entity}/{pname}: {mind_count_in_proj} MIND run(s)")

print(f"\n=== Summary ===")
print(f"Entities scanned : {ENTITIES}")
print(f"Total projects   : {total_projects}")
print(f"Total runs       : {total_runs}")
print(f"MIND runs matched: {len(all_runs)}")

# --- Save JSON ---
json_path = "wandb_mind_runs_wuchen.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_runs, f, indent=2, ensure_ascii=False, default=str)
print(f"Saved {json_path}")

# --- Save CSV ---
if all_runs:
    all_keys = list(dict.fromkeys(k for row in all_runs for k in row))
    csv_path = "wandb_mind_runs_wuchen.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_runs)
    print(f"Saved {csv_path}")

# --- Print compact table ---
print(f"\n{'#':>3} | {'State':<10} | {'Created':<20} | {'Name'}")
print("-" * 120)
for i, r in enumerate(all_runs, 1):
    print(f"{i:>3} | {r['state']:<10} | {r['created_at']:<20} | {r['name']}")
