"""Persistent poll for r11b chat+Base + JEV. Writes to poll_r11c.log."""
import subprocess, time, sys, os, requests
os.environ.setdefault("PYTHONIOENCODING","utf-8")
CHAT="serene_rice_88lx58jm5j"; BASE="magenta_pasta_7j0yzmwxhw"; JEV="stoic_hook_xdc9sbfbk6"
OUT="Q:/MiniOneRec/pipeline/poll_r11c.log"; POLL=int(os.environ.get("POLL","240"))
tpl='''
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import requests, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ml=MLClient(DefaultAzureCredential(), subscription_id='b6dc87f3-c479-49c8-8cb5-7896da3ff895', resource_group_name='AMLStudio', workspace_name='NewsFeedL2_AML')
RID="{RID}"
try: children=list(ml.jobs.list(parent_job_name=RID))
except Exception as e:
    print("ERR",e); sys.exit()
if not children: print("NOCHILD"); sys.exit()
child=children[0]; cs=str(child.status)
d=ml.jobs._runs_operations.get_run_details(child.name)
if 'user_logs/std_log.txt' not in d.log_files:
    print(child.name, cs, "no std_log"); sys.exit()
txt=requests.get(d.log_files['user_logs/std_log.txt'], timeout=90).content.decode('utf-8','replace')
lines=txt.splitlines()
bar=[l for l in lines if 's/it' in l and 'Loading weights' not in l]
err=[l for l in lines if 'Traceback' in l or 'CUDA out of memory' in l or 'RuntimeError' in l]
load=[l for l in lines if 'Loaded' in l]
print(child.name, cs)
if load: print("  LOAD:", " | ".join(load[-2:])[:160])
if bar: print("  BAR:", bar[-1][:200])
if err: print("  ERR:", " || ".join(err[-2:])[:300])
print("  TAIL:", lines[-1][:140])
'''
def run(rid):
    try:
        p=subprocess.run(["python","-c",tpl.format(RID=rid)],cwd="Q:/MiniOneRec/pipeline",capture_output=True,timeout=150,env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        return ((p.stdout or b"").decode("utf-8","replace").strip() or (p.stderr or b"").decode("utf-8","replace").strip())
    except Exception as e:
        return f"ERR {e}"
with open(OUT,"w",encoding="utf-8") as f:
    for i in range(2000):
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} tick{i} ===\n"); f.flush()
        for tag,rid in [("chat",CHAT),("base",BASE),("jev",JEV)]:
            f.write(f"[{time.strftime('%H:%M:%S')}] {tag} :: {run(rid)}\n"); f.flush()
        time.sleep(POLL)
print("poll exited")
