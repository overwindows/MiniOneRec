"""Reliable Azure credential for on-prem Windows where `az` is a .cmd shim.

`DefaultAzureCredential`/`AzureCliCredential`'s AzureCliCredential calls `az`
via `subprocess.Popen(["az", ...])`, and on this machine there is no bare
`az` executable on the subprocess PATH (`az` only resolves via `cmd /c` or the
full az.cmd path). That makes AzureCliCredential intermittently raise
"Failed to invoke the Azure CLI", which silently killed the auto-eval watcher
loop all night (trainings Completed but no eval ever submitted).

This module shells out to `cmd /c az account get-access-token` (proven to
work) and wraps the resulting token in a minimal TokenCredential so MLClient
authenticates reliably. `get_credential()` returns a DeleteOnExpiry-compatible,
inert object; the real token is fetched lazily on first get_token().
"""
import json
import shlex
import subprocess

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential

_MANAGEMENT_SCOPE = "https://management.azure.com/.default"


def _pick_scope(scopes):
    """Choose the ARM scope to request. az identity APIs need an azure.com scope."""
    for s in scopes or []:
        if "://" in s:
            return s
    return _MANAGEMENT_SCOPE


def _fetch_token(*scopes) -> AccessToken:
    """Call az account get-access-token via cmd /c and parse the JSON."""
    scope = _pick_scope(scopes)
    args = "az account get-access-token --output json --scope " + scope
    proc = subprocess.run(
        ["cmd", "/c", args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "az get-access-token failed rc=" + str(proc.returncode) +
            " stderr=" + (proc.stderr or "")[:400]
        )
    data = json.loads(proc.stdout)
    # `expiresOn` may be a unix epoch (int) or an ISO datetime string; normalize.
    from datetime import datetime, timezone
    exp = data.get("expiresOn") or data.get("expires_on")
    if isinstance(exp, (int, float)):
        expires_on = int(exp)
    else:
        expires_on = int(datetime.strptime(str(exp), "%Y-%m-%d %H:%M:%S.%f")
                         .replace(tzinfo=timezone.utc).timestamp())
    return AccessToken(data["accessToken"], expires_on)


class _AzTokenCredential:
    """Returns a fresh token on each get_token call (fine for one-shot polling)."""

    def __init__(self):
        self._token = None

    def get_token(self, *scopes, **kwargs):
        self._token = _fetch_token(*scopes)
        return self._token


def get_credential():
    """Return a reliable TokenCredential backed by `cmd /c az`.

    The bare `az` is not on azure-identity's subprocess PATH on this machine,
    so AzureCliCredential/DefaultAzureCredential fail intermittently. Shelling
    to `cmd /c az account get-access-token` works deterministically.
    """
    return _AzTokenCredential()
