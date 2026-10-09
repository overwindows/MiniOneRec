#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch missing OpenJev repo files (chat_template.jinja) onto the shared mount.

Downloads only the small missing files directly from HuggingFace to a local
staging dir, then copies them onto the ADLS mount. Does NOT re-download the
~13GB weights (already staged by the full download job).
"""
import argparse
import io
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(f"patch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("jev_patch")

REPO_ID = "apus-ailab/APUS-OpenJev-v1-4B"
MISSING = ["chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mount-dir", required=True)
    ap.add_argument("--dest-path", required=True)
    ap.add_argument("--debug-mode", default="false")
    args = ap.parse_args()

    dest = os.path.join(args.mount_dir, args.dest_path)
    os.makedirs(dest, exist_ok=True)
    log.info("=== jev_patch dest: %s ===", dest)
    log.info("----- dest file list (before) -----")
    for fn in sorted(os.listdir(dest)):
        log.info("   %s", fn)

    staging = "/tmp/jev_patch"
    os.makedirs(staging, exist_ok=True)

    from huggingface_hub import hf_hub_download
    for name in MISSING:
        rel = os.path.join(dest, name)
        if os.path.exists(rel):
            log.info("SKIP already present: %s", name)
            continue
        log.info("Downloading %s ...", name)
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            local_dir=staging,
            local_dir_use_symlinks=False,
        )
        shutil.copy2(local, rel)
        log.info("COPIED %s (%d bytes)", name, os.path.getsize(rel))

    log.info("----- dest file list (after) -----")
    for fn in sorted(os.listdir(dest)):
        log.info("   %s", fn)

    have_tok = os.path.exists(os.path.join(dest, "tokenizer.json"))
    have_cfg = os.path.exists(os.path.join(dest, "tokenizer_config.json"))
    have_jinja = os.path.exists(os.path.join(dest, "chat_template.jinja"))
    log.info("tokenizer.json=%s tokenizer_config.json=%s chat_template.jinja=%s",
             have_tok, have_cfg, have_jinja)
    if have_tok and have_cfg and have_jinja:
        log.info("PATCH_COMPLETE")
    else:
        log.error("PATCH_INCOMPLETE tokenizer.json=%s tokenizer_config.json=%s chat_template.jinja=%s",
                  have_tok, have_cfg, have_jinja)
        sys.exit(1)


if __name__ == "__main__":
    main()
