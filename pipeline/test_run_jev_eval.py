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
                    "--timeout-seconds", "43200",
                    "--max-prompt-tokens", "4095",
                    "--output-root", "shares/results/frozen",
                ],
                "ranking", "Singularity.ND12am_A100_v4", "Standard", 5000,
            ),
            (
                ["--prompt-contract", "both"],
                "recall", "Singularity.ND96amrs_A100_v4", "Premium", 0,
            ),
        ]
        script = Path(__file__).with_name("run_jev_eval.py")
        for extra, cluster, instance, tier, count in cases:
            with self.subTest(cluster=cluster, instance=instance):
                argv = [str(script), "--model-path", "shares/model/final_checkpoint",
                        "--source-revision", "a" * 40, *extra]
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
                    "both" if "--prompt-contract" in extra else "auto",
                )
                self.assertEqual(job.tags["source_revision"], "a" * 40)
                self.assertEqual(job._to_dict()["inputs"]["source_revision"], "a" * 40)
                if count:
                    self.assertEqual(node.limits.timeout, 43200)
                    self.assertEqual(job._to_dict()["inputs"]["max_prompt_tokens"], 4095)
                    self.assertEqual(job._to_dict()["inputs"]["output_root"], "shares/results/frozen")
                self.assertTrue(job.settings.force_rerun)
                self.assertNotIn("settings", job._to_dict()["jobs"]["node"])

    def test_worker_uses_isolated_checkout_and_committed_source_bundle(self):
        import json
        path = Path(__file__).parent / "components" / "mind_jev_eval" / "config_mind_jev_eval.json"
        commands = json.loads(path.read_text(encoding="utf-8"))["commands"]
        self.assertIn("{code_dir}/MiniOneRec", commands[0]["command"])
        self.assertIn("git fetch {code_dir}/source.bundle HEAD", commands[1]["command"])
        self.assertIn("git checkout --detach {source_revision}", commands[1]["command"])
        self.assertFalse(any("rm -rf" in command["command"] for command in commands))


if __name__ == "__main__":
    unittest.main()
