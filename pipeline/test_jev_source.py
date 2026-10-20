from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from jev_source import prepare_source_component


class SourceBundleTests(unittest.TestCase):
    def test_bundle_reconstructs_exact_committed_revision_from_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()

            def git(*args, cwd=repo):
                return subprocess.run(
                    ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                ).stdout.strip()

            git("init", "-q", "-b", "dev")
            source = Path(__file__).parent / "components" / "mind_jev_eval"
            target = repo / "pipeline" / "components" / "mind_jev_eval"
            target.mkdir(parents=True)
            for name in ("component.yaml", "config_mind_jev_eval.json", "pipeline_executor.py"):
                shutil.copy2(source / name, target / name)
            git("add", ".")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            git("update-ref", "refs/remotes/origin/dev", base)
            (repo / "source.txt").write_text("committed content\n", encoding="utf-8")
            git("add", "source.txt")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "new")
            expected = git("rev-parse", "HEAD")
            revision, component, upload = prepare_source_component(target / "component.yaml", bundle_source=True)
            self.addCleanup(upload.cleanup)
            self.assertEqual(revision, expected)
            self.assertEqual(component.version, "3-" + expected[:12])
            bundle = Path(upload.name) / "source.bundle"
            checkout = root / "checkout"
            checkout.mkdir()
            git("init", "-q", cwd=checkout)
            git("fetch", "-q", str(repo), base, cwd=checkout)
            git("fetch", "-q", str(bundle), "HEAD", cwd=checkout)
            git("checkout", "-q", "--detach", revision, cwd=checkout)
            self.assertEqual((checkout / "source.txt").read_text(), "committed content\n")
            with self.assertRaisesRegex(ValueError, "full commit SHA"):
                prepare_source_component(target / "component.yaml", "dev", False)
            with self.assertRaisesRegex(ValueError, "committed local HEAD"):
                prepare_source_component(target / "component.yaml", base, True)


if __name__ == "__main__":
    unittest.main()
