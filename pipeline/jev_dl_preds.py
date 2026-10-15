"""Download a datalake file fully to local via WebHDFS OPEN (offset loop)."""
import sys, requests, urllib.parse
sys.path.insert(0,'pipeline')
from azure_auth import get_credential
cred=get_credential()
token=cred.get_token("https://datalake.azure.net/.default").token
store="msn-dni-c09"
base=f"https://{store}.azuredatalakestore.net/webhdfs/v1"
H={"Authorization":f"Bearer {token}"}
path=sys.argv[1]; out=sys.argv[2]
chunk=4*1024*1024
buf=bytes()
offset=0
while True:
    q=urllib.parse.urlencode({"op":"OPEN","offset":offset,"length":chunk})
    r=requests.get(f"{base}/{path}?{q}",headers=H,timeout=180)
    if r.status_code!=200:
        print("err",r.status_code,r.text[:200]); break
    b=r.content
    if not b: break
    buf+=b
    if len(b)<chunk: break
    offset+=len(b)
open(out,"wb").write(buf)
print("wrote",len(buf),"bytes ->",out)
