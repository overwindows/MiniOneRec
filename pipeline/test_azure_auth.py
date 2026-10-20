from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from azure_auth import _fetch_token


class AzureTokenExpiryTests(unittest.TestCase):
    def token(self, **expiry):
        result = subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"accessToken": "fixture-token", **expiry}),
        )
        with (patch("azure_auth._find_az", return_value="az"),
              patch("azure_auth.subprocess.run", return_value=result)):
            return _fetch_token()

    def test_epoch_takes_precedence_over_local_wall_time(self):
        token = self.token(expires_on=1792466595, expiresOn="2036-01-01 00:00:00.000000")
        self.assertEqual(token.expires_on, 1792466595)

    def test_string_epoch_is_accepted(self):
        self.assertEqual(self.token(expires_on="1792466595").expires_on, 1792466595)

    def test_legacy_wall_time_uses_the_local_timezone(self):
        value = "2026-10-20 11:23:15.000000"
        self.assertEqual(self.token(expiresOn=value).expires_on,
                         int(datetime.fromisoformat(value).timestamp()))


if __name__ == "__main__":
    unittest.main()
