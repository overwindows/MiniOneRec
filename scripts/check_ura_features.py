"""Check if discover-rk-ura feeds have feature scores in candidate_cards."""
import json
import os
import sys

# Need to query Databricks directly for raw candidate_cards with features
from databricks import sql as dbsql

token = os.environ.get("DATABRICKS_TOKEN")
if not token:
    print("Set DATABRICKS_TOKEN first")
    sys.exit(1)

conn = dbsql.connect(
    server_hostname="adb-3355567219430035.15.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/9c266703e61b038d",
    access_token=token,
)
cursor = conn.cursor()

# Query: sample URA vs non-URA feeds, check if features exist
query = """
SELECT 
    CASE WHEN user_flight_ids LIKE '%discover-rk-ura%' THEN 'URA' ELSE 'non-URA' END as group_name,
    COUNT(*) as cnt,
    SUM(CASE WHEN candidate_cards LIKE '%clickLikelihood%' THEN 1 ELSE 0 END) as has_features,
    SUM(CASE WHEN candidate_cards NOT LIKE '%clickLikelihood%' THEN 1 ELSE 0 END) as no_features
FROM mai_ws_discover.analytics.ods_doca_feed_grounded_v8_partitioned
WHERE bizdate = '20260420'
GROUP BY CASE WHEN user_flight_ids LIKE '%discover-rk-ura%' THEN 'URA' ELSE 'non-URA' END
"""

print("Querying Databricks...")
cursor.execute(query)
rows = cursor.fetchall()
print(f"\n{'Group':<10} {'Total':>8} {'Has Features':>14} {'No Features':>13} {'% Features':>12}")
print("-" * 60)
for row in rows:
    grp, total, has_f, no_f = row
    pct = has_f / total * 100 if total > 0 else 0
    print(f"{grp:<10} {total:>8} {has_f:>14} {no_f:>13} {pct:>11.1f}%")

cursor.close()
conn.close()
