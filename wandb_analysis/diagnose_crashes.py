"""Fetch crash/failure details for all non-finished MIND runs."""
import os, json, sys
os.environ["WANDB_API_KEY"] = "fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b"
import wandb

api = wandb.Api(timeout=120)

with open("wandb_mind_runs_wuchen.json", "r", encoding="utf-8") as f:
    runs_meta = json.load(f)

failed_runs = [r for r in runs_meta if r["state"] in ("crashed", "failed")]
print(f"Non-finished MIND runs: {len(failed_runs)}\n")

results = []

for i, meta in enumerate(failed_runs, 1):
    run_path = meta["run_path"]
    name = meta["name"]
    state = meta["state"]
    created = meta["created_at"]
    project = meta["project"]
    
    print(f"{'='*100}")
    print(f"#{i}  {name}")
    print(f"    project={project}  state={state}  created={created}")
    print(f"    url={meta['url']}")
    
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"    [ERROR] Could not fetch run: {e}")
        results.append({"name": name, "state": state, "error": str(e)})
        continue

    # Get system metadata
    sys_metrics = {}
    try:
        sm = run.summary._json_dict if hasattr(run.summary, '_json_dict') else dict(run.summary)
        # Look for OOM / error indicators
        for k, v in sm.items():
            kl = k.lower()
            if any(x in kl for x in ["error", "crash", "exception", "oom", "memory", "gpu", "exit"]):
                sys_metrics[k] = v
            # Also capture last training step info
            if any(x in kl for x in ["train/loss", "train/epoch", "train/global_step"]):
                sys_metrics[k] = v
    except:
        pass

    if sys_metrics:
        print(f"    Summary signals: {json.dumps(sys_metrics, default=str)}")

    # Try to get the last lines of the run log (stderr/stdout)
    try:
        log_lines = []
        for logfile in run.files():
            fname = logfile.name
            if fname in ("output.log", "wandb-summary.json"):
                continue
        
        # Fetch output.log which typically has the crash traceback
        try:
            logfile = run.file("output.log")
            content = logfile.download(replace=True, exist_ok=True)
            with open(content.name, "r", encoding="utf-8", errors="replace") as lf:
                all_lines = lf.readlines()
            
            # Get last 60 lines which usually contain the error
            tail = all_lines[-60:]
            # Filter for error-relevant lines
            error_lines = []
            capture = False
            for line in tail:
                ll = line.lower()
                if any(x in ll for x in ["error", "traceback", "exception", "oom", "cuda", "killed", "segfault", "signal", "abort", "memory", "raise", "fatal"]):
                    capture = True
                if capture:
                    error_lines.append(line.rstrip())
            
            if error_lines:
                print(f"    --- Error from output.log (last relevant lines) ---")
                for el in error_lines[-30:]:
                    print(f"    {el}")
            else:
                # Just show last 15 lines
                print(f"    --- Last 15 lines of output.log ---")
                for el in all_lines[-15:]:
                    print(f"    {el.rstrip()}")
                    
        except Exception as e2:
            pass

        # Also try wandb-debug.log or other log files
        try:
            for fname in ["wandb-debug.log"]:
                logfile = run.file(fname)
                content = logfile.download(replace=True, exist_ok=True)
                with open(content.name, "r", encoding="utf-8", errors="replace") as lf:
                    all_lines = lf.readlines()
                error_lines = [l.rstrip() for l in all_lines[-30:] 
                               if any(x in l.lower() for x in ["error", "exception", "crash", "killed", "oom", "signal", "fatal", "abort"])]
                if error_lines:
                    print(f"    --- Errors from {fname} ---")
                    for el in error_lines[-10:]:
                        print(f"    {el}")
        except:
            pass

    except Exception as e:
        print(f"    [WARN] Could not fetch logs: {e}")

    # Config summary
    cfg = json.loads(meta.get("config", "{}"))
    interesting = {k: cfg[k] for k in ["model_name_or_path", "num_train_epochs", 
                   "per_device_train_batch_size", "learning_rate", "neg_ratio",
                   "max_history", "use_abstract", "use_chat_template", "data_root",
                   "deepspeed"] if k in cfg}
    if interesting:
        print(f"    config: {json.dumps(interesting, default=str)}")
    
    print()

print(f"\n{'='*100}")
print("DONE - analyzed all crashed/failed runs")
