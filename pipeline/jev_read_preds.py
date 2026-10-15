"""Read a dev_jev_predictions.txt from datalake via WebHDFS OPEN and print sample."""
import sys, requests, urllib.parse
sys.path.insert(0,'pipeline')
sys.stdout.reconfigure(encoding="utf-8",errors="replace")
from azure_auth import get_credential
cred=get_credential()
token=cred.get_token("https://datalake.azure.net/.default").token
store="msn-dni-c09"
base=f"https://{store}.azuredatalakestore.net/webhdfs/v1"
H={"Authorization":f"Bearer {token}"}
path=sys.argv[1]
q=urllib.parse.urlencode({"op":"OPEN","offset":0,"length":int(sys.argv[2]) if len(sys.argv)>2 else 2000})
r=requests.get(f"{base}/{path}?{q}",headers=H,timeout=120)
print("status", r.status_code)
print(r.text[:2000] if r.status_code==200 else r.text[:500])
