"""Poll the A7.3L title-only dev eval; when Completed, grep std_log AUC."""
import sys, io, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
from azure_auth import get_credential
from azure.ai.ml import MLClient

ML = MLClient(get_credential(),
              subscription_id="b6dc87f3-c479-49c8-8cb5-7896da3ff895",
              resource_group_name="AMLStudio",
              workspace_name="NewsFeedL2_AML")
RID = "purple_malanga_09mg6l4ncq"

def child_of(rid):
    for c in ML.jobs.list(parent_job_name=rid):
        if c.name:
            return c.name
    return None

def main():
    st = str(ML.jobs.get(RID).status)
    print(f"parent {RID}: {st}")
    if st.upper() != "JOBSTATUS.COMPLETED":
        return
    c = child_of(RID)
    if not c:
        print("(no child yet)"); return
    rd = ML.jobs._runs_operations.get_run_details(c)
    url = None
    for k in (rd.log_files or {}):
        if k.endswith("std_log.txt"):
            url = rd.log_files[k]; break
    if not url:
        print("(no std_log)"); return
    data = urllib.request.urlopen(urllib.request.Request(url), timeout=120).read().decode("utf-8","replace")
    for line in data[-4000:].splitlines():
        if any(x in line for x in ("AUC","MRR","nDCG")):
            print(line)
    print(f"--- log tail bytes {len(data)}")

if __name__ == "__main__":
    main()
