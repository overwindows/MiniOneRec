"""Grep JEV-C1 small SFT child logs for dev AUC."""
import sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from azure.ai.ml import MLClient

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")
RID = "sincere_garlic_7t99m7nxw6"

def main():
    for c in ML.jobs.list(parent_job_name=RID):
        if not c.name:
            continue
        try:
            rd = ML.jobs._runs_operations.get_run_details(c.name)
        except Exception as e:
            print(c.name, "get_run_details err", e); continue
        url = None
        for k in (rd.log_files or {}):
            if k.endswith("std_log.txt"):
                url = rd.log_files[k]; break
        if not url:
            print(c.name, "(no std_log)", c.display_name); continue
        try:
            data = urllib.request.urlopen(urllib.request.Request(url), timeout=180).read().decode("utf-8","replace")
        except Exception as e:
            print(c.name, "urlopen err", e); continue
        aucs = []
        for line in data.splitlines():
            if "AUC" in line and ("nDCG" in line or "MRR" in line or "Training" in line or "best" in line.lower()):
                aucs.append(line.strip()[:200])
            if "best" in line.lower() and ("auc" in line.lower() or "0." in line):
                aucs.append(line.strip()[:200])
        print(f"=== {c.display_name} ({c.name}) bytes={len(data)}")
        for a in aucs[-15:]:
            print("   ", a)

if __name__ == "__main__":
    main()
