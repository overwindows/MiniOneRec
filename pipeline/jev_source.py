"""Deliver an immutable committed JEV source revision to an AML component."""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from azure.ai.ml import load_component


def prepare_source_component(component_path, revision=None, bundle_source=False):
    component_path = Path(component_path)
    repo = component_path.parents[3]

    def git(*args):
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=repo,
        ).stdout.strip()

    revision = revision or git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--source-revision must be a full commit SHA")
    component = load_component(source=str(component_path))
    if not bundle_source:
        return revision, component, None
    if revision != git("rev-parse", "HEAD"):
        raise ValueError("--bundle-source requires --source-revision to equal committed local HEAD")
    base = git("merge-base", "HEAD", "origin/dev")
    if base == revision:
        raise ValueError("--bundle-source requires a local commit beyond origin/dev")
    upload = tempfile.TemporaryDirectory(prefix="jev-source-")
    try:
        directory = Path(upload.name)
        for path in component_path.parent.iterdir():
            if path.suffix in (".yaml", ".json", ".py") and not path.name.startswith("test_"):
                shutil.copy2(path, directory / path.name)
        git("bundle", "create", str(directory / "source.bundle"), "HEAD", "^" + base)
        component = load_component(source=str(directory / component_path.name))
        component.version = str(component.version) + "-" + revision[:12]
        return revision, component, upload
    except BaseException:
        upload.cleanup()
        raise
