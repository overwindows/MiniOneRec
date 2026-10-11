#!/bin/bash
# Poll A7.3 r2 pairwise training + sub-2B eval-only recovery jobs
IDS="silver_ring_l47fyrmkx0 serene_bell_nmnm20cn4v bubbly_screw_j68j6vw7rs frank_cord_yby9j1rl5h ashy_cartoon_mdz49cwrh6 olive_leek_6k8svsx0lw helpful_planet_9ch6y47pr9 silver_spring_qf8hlzc0bn gentle_cumin_p0hk9n82m5 coral_kitchen_w35lvg2cxk"
while true; do
  { echo "=== $(date '+%F %T') ===";
    python -u monitor_jobs.py --status $IDS 2>&1 | grep -viE "warning|unknown field|Failed to invoke|Proceeding|experimental|AutoDelete|Intellectual|Protection|Class ";
  } >> poll_a73.log
  sleep 600
done
