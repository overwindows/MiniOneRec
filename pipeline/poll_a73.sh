#!/bin/bash
# Poll for A7.3 pairwise + sub-2B recovery jobs
IDS="magenta_ocean_ftdzs7pqt8 jovial_sun_48100pm3yb dreamy_apricot_ft6l3k5f9q quirky_date_dkv0tt8bk8 silly_airport_dy16bqyk4s frosty_scooter_7g4b0lz065 blue_rabbit_zsx7q7bqd7 cyan_stick_ysl99gl86l dreamy_cumin_sf41x5xjkz mango_pea_35h81yt951 silver_card_gkfbvh9vj7 olive_boniato_0jtf0jq8vc placid_yuca_yztdxpmpgm"
while true; do
  { echo "=== $(date '+%F %T') ===";
    python -u monitor_jobs.py --status $IDS 2>&1 | grep -viE "warning|unknown field|Failed to invoke|Proceeding";
  } >> poll_a73.log
  sleep 300
done
