import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class JevTrainingSubmissionTests(unittest.TestCase):
    def test_defaults_and_explicit_decision_modes(self):
        script = Path(__file__).with_name("run_jev_train.py")
        for loss, contract, extra in [
            ("full_vocab", "legacy", []),
            ("decision", "native_click", [
                "--loss-type", "decision", "--prompt-contract", "native_click", "--enable-kl", "0",
                "--instance-type", "Singularity.ND12am_A100_v4", "--timeout-seconds", "3600",
            ]),
        ]:
            with self.subTest(loss=loss, contract=contract):
                with (
                    patch("azure.ai.ml.MLClient") as client,
                    patch.object(sys, "argv", [
                        str(script), "--code-revision", "test-revision",
                        "--source-revision", "a" * 40, *extra,
                    ]),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    client.return_value.jobs.create_or_update.return_value = SimpleNamespace(
                        name="test-run", studio_url="test-studio", display_name="test-training",
                    )
                    runpy.run_path(str(script), run_name="__main__")
                    submit = client.return_value.jobs.create_or_update
                    submit.assert_called_once()
                    job = submit.call_args.args[0]
                serialized = job._to_dict()
                self.assertEqual(serialized["inputs"]["loss_type"], loss)
                self.assertEqual(serialized["inputs"]["prompt_contract"], contract)
                self.assertEqual(serialized["inputs"]["source_revision"], "a" * 40)
                self.assertEqual(job.tags["source_revision"], "a" * 40)
                self.assertTrue(job.settings.force_rerun)
                self.assertNotIn("settings", serialized["jobs"]["train_node"])
                self.assertTrue(str(job.jobs["train_node"].compute).endswith("/ranking"))
                node = job.jobs["train_node"]
                self.assertEqual(node.resources.instance_type, "Singularity.ND12am_A100_v4"
                                 if loss == "decision" else "Singularity.ND96amrs_A100_v4")
                if loss == "decision":
                    self.assertEqual(node.limits.timeout, 3600)

    def test_worker_clones_isolated_source_and_checks_out_the_pin(self):
        config = Path(__file__).parent / "components" / "jev_train" / "config_jev_train.json"
        commands = json.loads(config.read_text(encoding="utf-8"))["commands"]
        self.assertIn("{code_dir}/MiniOneRec", commands[0]["command"])
        self.assertIn("git fetch {code_dir}/source.bundle HEAD", commands[1]["command"])
        self.assertIn("git checkout --detach {source_revision} && git rev-parse HEAD",
                      commands[1]["command"])
        self.assertFalse(any("rm -rf" in command["command"] for command in commands))

    def test_decision_mode_rejects_unsupported_kl_before_submission(self):
        script = Path(__file__).with_name("run_jev_train.py")
        with (
            patch("azure.ai.ml.MLClient") as client,
            patch.object(sys, "argv", [str(script), "--loss-type", "decision"]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as error:
                runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(error.exception.code, 2)
            client.return_value.jobs.create_or_update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
