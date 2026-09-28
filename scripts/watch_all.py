import requests, time, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pipeline.monitor_jobs import client

IDS = {
    'neat_band_393lt71y7t': 'r7 chat (Qwen3.5-2B)',
    'happy_energy_zrvmghx41r': 'r7 base (Qwen3.5-2B-Base)',
    'bold_foot_75cv39dfth': 'L1.3x pointwise 1.7B',
    'honest_cartoon_3r3jwfljql': 'M4.1c 1.7B multitask RN2 mb2',
    'great_pear_n8jrrxdbsp': 'M2B 2B multitask RN2 mb2',
    'frosty_zoo_8qlgqlc84z': 'M2B-Base 2BBase multitask RN2 mb2',
    'dreamy_pepper_xzt2wld734': 'R7c2 2B pointwise rerun',
    'keen_lychee_27zbpfkgkg': 'JEV4B download (CPU)',
    'amiable_wire_4rc378x92r': 'JEV4B download to cosmos v2 (CPU)',
    'quiet_wheel_d6dplq44x4': 'M3B 2B multitask PW0.5 RN2 mb2 (env fix)',
    'shy_yuca_7qmtcsf96q': 'M3BL 2BBase multitask PW0.5 RN2 mb2 (env fix)',
}
logs = {
    'neat_band_393lt71y7t': 'scripts/r7_chat_std_log.txt',
    'happy_energy_zrvmghx41r': 'scripts/r7_base_std_log.txt',
    'bold_foot_75cv39dfth': 'scripts/l13x_std_log.txt',
    'honest_cartoon_3r3jwfljql': 'scripts/m41c_std_log.txt',
    'great_pear_n8jrrxdbsp': 'scripts/m2b_std_log.txt',
    'frosty_zoo_8qlgqlc84z': 'scripts/m2b_base_std_log.txt',
    'dreamy_pepper_xzt2wld734': 'scripts/r7c2_std_log.txt',
    'keen_lychee_27zbpfkgkg': 'scripts/jev_dl_std_log.txt',
    'amiable_wire_4rc378x92r': 'scripts/jev_dl_cosmos_std_log.txt',
    'quiet_wheel_d6dplq44x4': 'scripts/m3b_std_log.txt',
    'shy_yuca_7qmtcsf96q': 'scripts/m3bl_std_log.txt',
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
            try:
                children = list(ml.jobs.list(parent_job_name=rid))
                for c in children:
                    print(f"   child {c.name[:8]} {c.status}", flush=True)
                c = running_child(children)
            except Exception:
                c = None
            if c:
                try:
                    rd = ml.jobs._runs_operations.get_run_details(c.name)
                    lf = rd.log_files
                    uri = lf.get('user_logs/std_log.txt')
                    if uri:
                        r = requests.get(uri, timeout=60)
                        txt = r.text
                        open(logs[rid], 'w', encoding='utf-8').write(txt)
                        ln = txt.splitlines()
                        # print training/eval line if present
                        hit = ''
                        for line in reversed(ln[-40:]):
                            if ("loss" in line and ("[" in line or "'loss'" in line)) or "eval_loss" in line or "AUC" in line:
                                hit = line; break
                        print(f"   [{len(ln)} lines] {hit[-300:]}", flush=True)
                except Exception as e:
                    print(f"   log ERR {e}", flush=True)
        except Exception as e:
            print(f"{rid}: ERR {type(e).__name__} {e}", flush=True)
    time.sleep(240)
