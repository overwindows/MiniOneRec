import csv
import json
from pathlib import Path

import wandb

PRIMARY_ENTITY = "wuchen"
FALLBACK_ENTITY = "devdivchina"
QUERY = "mind"

OUT_JSON = Path(r"c:\Users\wuc\MiniOneRec\wandb_mind_runs_wuchen.json")
OUT_CSV = Path(r"c:\Users\wuc\MiniOneRec\wandb_mind_runs_wuchen.csv")


def to_text(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return " ".join(to_text(x) for x in v)
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(v)
    return str(v)


def flatten_summary(summary):
    out = {}
    if summary is None:
        return out
    try:
        items = dict(summary).items()
    except Exception:
        try:
            items = summary.items()
        except Exception:
            return out
    for k, v in items:
        ks = str(k)
        low = ks.lower()
        if any(tok in low for tok in ["acc", "ndcg", "mrr", "auc", "loss"]):
            out[f"summary.{ks}"] = v
    return out


def run_matches(run, project_name):
    fields = []
    fields.append(getattr(run, "name", None))
    fields.append(getattr(run, "display_name", None))
    fields.append(getattr(run, "group", None))
    fields.append(getattr(run, "job_type", None))
    fields.append(getattr(run, "notes", None))
    fields.append(project_name)

    tags = getattr(run, "tags", None)
    if tags is not None:
        fields.append(tags)

    cfg = getattr(run, "config", None)
    if cfg:
        try:
            for _, v in dict(cfg).items():
                fields.append(v)
        except Exception:
            fields.append(cfg)

    blob = " ".join(to_text(x) for x in fields if x is not None).lower()
    return QUERY in blob


def list_projects(api, entity):
    return list(api.projects(entity=entity))


def main():
    api = wandb.Api()
    projects = []
    active_entity = PRIMARY_ENTITY
    errors = []

    try:
        projects = list_projects(api, PRIMARY_ENTITY)
    except Exception as e:
        errors.append(f"Failed listing projects for entity '{PRIMARY_ENTITY}': {e}")
        try:
            projects = list_projects(api, FALLBACK_ENTITY)
            active_entity = FALLBACK_ENTITY
        except Exception as e2:
            errors.append(f"Failed listing projects for entity '{FALLBACK_ENTITY}': {e2}")
            projects = []

    total_projects_scanned = len(projects)
    total_runs_scanned = 0
    matched = []

    for p in projects:
        project_name = getattr(p, "name", None) or ""
        entity_name = getattr(p, "entity", None) or active_entity
        path = f"{entity_name}/{project_name}"
        try:
            runs = api.runs(path=path)
        except Exception as e:
            errors.append(f"Failed listing runs for {path}: {e}")
            continue

        for run in runs:
            total_runs_scanned += 1
            try:
                if not run_matches(run, project_name):
                    continue

                rec = {
                    "entity": entity_name,
                    "project": project_name,
                    "run_id": getattr(run, "id", ""),
                    "run_path": "/".join(getattr(run, "path", []) or []),
                    "url": getattr(run, "url", ""),
                    "name": getattr(run, "name", ""),
                    "display_name": getattr(run, "display_name", ""),
                    "state": getattr(run, "state", ""),
                    "created_at": getattr(run, "created_at", ""),
                    "heartbeat_at": getattr(run, "heartbeat_at", ""),
                    "host": getattr(run, "host", ""),
                    "tags": to_text(getattr(run, "tags", [])),
                    "group": getattr(run, "group", ""),
                    "job_type": getattr(run, "job_type", ""),
                }
                rec.update(flatten_summary(getattr(run, "summary", None)))
                matched.append(rec)
            except Exception as e:
                errors.append(f"Error processing run in {path}: {e}")

    OUT_JSON.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")

    base_cols = [
        "entity", "project", "run_id", "run_path", "url", "name", "display_name", "state",
        "created_at", "heartbeat_at", "host", "tags", "group", "job_type",
    ]
    metric_cols = sorted({k for r in matched for k in r.keys() if k.startswith("summary.")})
    cols = base_cols + metric_cols

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in matched:
            w.writerow(r)

    print(f"Active entity used for scanning: {active_entity}")
    print(f"Total projects scanned: {total_projects_scanned}")
    print(f"Total runs scanned: {total_runs_scanned}")
    print(f"Mind runs matched: {len(matched)}")
    print(f"JSON output: {OUT_JSON}")
    print(f"CSV output: {OUT_CSV}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"- {e}")


if __name__ == "__main__":
    main()
