import os,sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pipeline.monitor_jobs import client
import time, requests

IDS = {
    'quiet_wheel_d6dplq44x4': 'M3B 2B multitask (env fix)',
    'shy_yuca_7qmtcsf96q': 'M3BL 2BBase multitask (env fix)',
    'crimson_date_k8mk9cm6z9': 'JEV4B download to cosmos',
}
logs = {
    'quiet_wheel_d6dplq44x4': 'scripts/m3b_std_log.txt',
    'shy_yuca_7qmtcsf96q': 'scripts/m3bl_std_log.txt',
    'crimson_date_k8mk9cm6z9': 'scripts/jev_dl_cosmos_std_log.txt',
}

def running_child(children):
    for c in children:
        if c.status and 'RUNNING' in str(c.status).upper():
            return c
    return children[0] if children else None

ml = client()
while True:
    print(f"\n=== {time.strftime('%H:%M:%S')} ===", flush=True)
    for rid, label in IDS.items():
        try:
            j = ml.jobs.get(rid)
            print(f"{rid} [{str(j.status)}] {label}", flush=True)
            for c in list(ml.jobs.list(parent_job_name=rid)):
                print(f"   child {c.name[:8]} {c.status}", flush=True)
            c = running_child(list(ml.jobs.list(parent_job_name=rid)))
            if c:
                try:
                    rd = ml.jobs._runs_operations.get_run_details(c.name)
                    uri = rd.log_files.get('user_logs/std_log.txt')
                    if uri:
                        r = requests.get(uri, timeout=60)
                        txt = r.text
                        open(logs[rid], 'w', encoding='utf-8').write(txt)
                        ln = txt.splitlines()
                        hit = ''
                        for line in reversed(ln[-40:]):
                            if ("loss" in line and "[" in line) or "eval_loss" in line or "AUC" in line or "MOUNT_LIST" in line or "DOWNLOAD_COMPLETE" in line or "MOUNT" in line:
                                hit = line; break
                        print(f"   [{len(ln)} lines] {hit[-250:]}", flush=True)
                except Exception as e:
                    print(f"   log ERR {e}", flush=True)
        except Exception as e:
            print(f"{rid}: ERR {type(e).__name__} {e}", flush=True)
    time.sleep(240)
