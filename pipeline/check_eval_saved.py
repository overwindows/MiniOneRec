"""Find the eval child's std_log and grep for the saved checkpoints/output path."""
import sys, io, urllib.request, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from azure.ai.ml import MLClient

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")
RID = "purple_malanga_09mg6l4ncq"

def children(rid):
    out = []
    for c in ML.jobs.list(parent_job_name=rid):
        if c.name:
            out.append(c.name)
    return out

def main():
    for c in children(RID):
        print("=== child", c)
        try:
            rd = ML.jobs._runs_operations.get_run_details(c)
        except Exception as e:
            print("  (get_run_details err)", e); continue
        url = None
        for k in (rd.log_files or {}):
            if k.endswith("std_log.txt"):
                url = rd.log_files[k]; break
        if not url:
            print("  (no std_log)"); continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(url), timeout=180).read().decode("utf-8","replace")
        except Exception as e:
            print("  (urlopen err)", e); continue
        # grep for relevant keywords
        for line in data.splitlines():
            low = line.lower()
            if any(x in low for x in ("saved to", "output_dir", "mind_small", "mind_large",
                                      "eval_split", "eval results", "predictions", "checkpoint")):
                print("   ", line.strip()[:300])
        print(f"   --- log bytes {len(data)}")

if __name__ == "__main__":
    main()
