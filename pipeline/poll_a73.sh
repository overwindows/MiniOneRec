#!/bin/bash
# Poll A7.3 r2 pairwise training + sub-2B eval-only + gemma/STM recovery jobs
IDS="silver_ring_l47fyrmkx0 serene_bell_nmnm20cn4v bubbly_screw_j68j6vw7rs patient_brick_b829btzn4v frank_cord_yby9j1rl5h ashy_cartoon_mdz49cwrh6 olive_leek_6k8svsx0lw strong_basket_q5jzrgqrbd boring_ghost_15449mswxn mango_leather_m2t42q6bqr gentle_salt_5hpqn3nqmy gifted_diamond_zf9wq2wc33"
while true; do
  { echo "=== $(date '+%F %T') ===";
    python -u monitor_jobs.py --status $IDS 2>&1 | grep -viE "warning|unknown field|Failed to invoke|Proceeding|experimental|AutoDelete|Intellectual|Protection|Class ";
  } >> poll_a73.log
  sleep 600
done
