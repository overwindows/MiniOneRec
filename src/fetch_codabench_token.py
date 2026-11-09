"""Mint a Codabench API token from account credentials and cache it.

WHY THIS FILE EXISTS
--------------------
scripts/submit_codabench.sh needs a token (`Authorization: Token <key>`).
The key was not present anywhere on this machine, so every submission was
blocked on a human step of copying it off another machine by hand.

The endpoint is `POST /api/api-token-auth/` with a JSON body
`{"username": ..., "password": ...}` and it returns `{"token": "<40 hex>"}`.
Note the double `api/` -- the OpenAPI schema is served from
`/api/api/schema/`, so the API's own paths are rooted at `/api/api/...`.
The single-slash guesses (`/api/auth/login/`, `/api/token/`) all 404, and
`/api/api-token-auth/` is the one that answers 200.

Two gotchas found the hard way:
  * Python's urllib gets served a SCRAMBLED anti-bot page (random prose with
    mangled links) whenever it sends a browser User-Agent. curl works. The
    token request below therefore uses urllib WITHOUT a UA header.
  * The password is never written to disk and never appears in a command
    line: it is read from $CODEBENCH_PASSWORD.

Usage:
    CODEBENCH_PASSWORD='...' python3 src/fetch_codabench_token.py
    CODEBENCH_PASSWORD='...' python3 src/fetch_codabench_token.py --print
"""
import argparse
import json
import os
import stat
import sys
import urllib.error
import urllib.request

TOKEN_URL = "https://www.codabench.org/api/api-token-auth/"
USER = os.environ.get("CODEBENCH_USERNAME", "overwindows")
TOKEN_PATH = os.path.expanduser("~/.codeguru/codabench_key")


def mint(username, password, timeout=30):
    """POST the credentials; return the token string or raise."""
    req = urllib.request.Request(
        TOKEN_URL,
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            status = r.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        status = e.code
    if status != 200:
        raise RuntimeError(f"{TOKEN_URL} -> HTTP {status}: {body[:200]}")
    obj = json.loads(body)
    tok = obj.get("token")
    if not tok:
        raise RuntimeError(f"no 'token' in response: {body[:200]}")
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="also print the token")
    ap.add_argument("--username", default=USER)
    args = ap.parse_args()

    pw = os.environ.get("CODEBENCH_PASSWORD")
    if not pw:
        sys.exit("FATAL: CODEBENCH_PASSWORD is not set (export it; it is never stored)")

    try:
        tok = mint(args.username, pw)
    except Exception as e:
        sys.exit(f"FATAL: {e}")

    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    os.chmod(os.path.dirname(TOKEN_PATH), 0o700)
    tmp = TOKEN_PATH + ".tmp"
    # Create the temp file with mode 0600 ATOMICALLY, so the token is never
    # world-readable in the window between creation and chmod.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    os.replace(tmp, TOKEN_PATH)
    os.chmod(TOKEN_PATH, stat.S_IRUSR | stat.S_IWUSR)
    print(f"token cached at {TOKEN_PATH} (mode 0600), {len(tok)} chars")
    if args.print:
        print(tok)


if __name__ == "__main__":
    main()
