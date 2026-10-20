import contextlib
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class JevEvalSubmissionTests(unittest.TestCase):
    def test_resource_selection(self):
        cases = [
            ([], "recall", "Singularity.ND96amrs_A100_v4", "Premium", 0),
            (
                [
                    "--virtual-cluster", "ranking",
                    "--instance-type", "Singularity.ND12am_A100_v4",
                    "--sla-tier", "Standard",
                    "--max-impressions", "5000",
                ],
                "ranking", "Singularity.ND12am_A100_v4", "Standard", 5000,
            ),
            (
                ["--prompt-contract", "native_click"],
                "recall", "Singularity.ND96amrs_A100_v4", "Premium", 0,
            ),
        ]
        script = Path(__file__).with_name("run_jev_eval.py")
        for extra, cluster, instance, tier, count in cases:
            with self.subTest(cluster=cluster, instance=instance):
                argv = [str(script), "--model-path", "shares/model/final_checkpoint", *extra]
                with (
                    patch("azure.ai.ml.MLClient") as client,
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    client.return_value.jobs.create_or_update.return_value = SimpleNamespace(
                        name="test-run", studio_url="test-studio",
                    )
                    runpy.run_path(str(script), run_name="__main__")
                    submit = client.return_value.jobs.create_or_update
                    submit.assert_called_once()
                    job = submit.call_args.args[0]

                node = job.jobs["node"]
                self.assertTrue(str(node.compute).endswith("/virtualClusters/" + cluster))
                self.assertEqual(node.compute, job.settings.default_compute)
                self.assertEqual(node.resources.instance_count, 1)
                self.assertEqual(node.resources.instance_type, instance)
                self.assertEqual(node.resources.properties["singularity"]["slaTier"], tier)
                self.assertTrue(
                    node.environment_variables["_AZUREML_SINGULARITY_JOB_UAI"].endswith(
                        "/rankfun_aml"
                    )
                )
                self.assertEqual(job._to_dict()["inputs"]["max_impressions"], count)
                self.assertEqual(
                    job._to_dict()["inputs"]["prompt_contract"],
                    "native_click" if "--prompt-contract" in extra else "auto",
                )
                self.assertTrue(job.settings.force_rerun)
                self.assertNotIn("settings", job._to_dict()["jobs"]["node"])


if __name__ == "__main__":
    unittest.main()
