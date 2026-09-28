#!/bin/bash
cd /q/MiniOneRec
while true; do
  python -c "
from pipeline.monitor_jobs import client
ml = client()
for rid in ['neat_band_393lt71y7t','happy_energy_zrvmghx41r','shy_fowl_14fkx19tyj','bold_foot_75cv39dfth']:
    try:
        j = ml.jobs.get(rid)
        st = str(j.status)
        print(f'{rid} [{st}] {j.display_name}')
        if 'FAILED' in st.upper() or 'CANCEL' in st.upper():
            for child in ml.jobs.list(parent_job_name=rid):
                rd = ml.jobs._runs_operations.get_run_details(child.name)
                err=''
                if rd.error:
                    err=(rd.error.get('error',{}).get('message','') if isinstance(rd.error,dict) else str(rd.error))[:250]
                print(f'   child {child.name} {child.status} :: {err}')
    except Exception as e:
        print(f'{rid}: ERR {type(e).__name__} {e}')
" 2>&1 | grep -vi warning
  echo "--- $(date +%H:%M:%S) ---"
  sleep 240
done
