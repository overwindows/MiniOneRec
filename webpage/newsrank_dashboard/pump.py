"""Continuously refresh the NewsRank Live progress page.

Every INTERVAL seconds: regenerate progress.json from AML + upload.
Runs until killed. Logs a heartbeat line each refresh.
"""
import os, subprocess, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_progress.py")

while True:
    try:
        r = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True)
        tail = (r.stdout.strip() + r.stderr.strip()).splitlines()
        print(f"[{time.strftime('%H:%M:%S')}] refresh rc={r.returncode} {tail[-1] if tail else ''}", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ERR {type(e).__name__}: {e}", flush=True)
    time.sleep(INTERVAL)
