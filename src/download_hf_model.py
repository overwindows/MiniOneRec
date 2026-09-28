"""Download a HuggingFace model snapshot to a local directory.

Used to pre-stage OpenJev onto the shared ADLS datastore mount so eval jobs on
GPU don't consume A100 hours doing the download.

Usage:
    python src/download_hf_model.py --repo_id apus-ailab/APUS-OpenJev-v1-4B \
        --dest /path/to/shares/users/wuc/models/APUS-OpenJev-v1-4B
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--dest", required=True, help="Absolute destination dir on the mounted datastore")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--staging", default="/tmp/jev_dl", help="Local writable staging dir (NOT on the FUSE mount)")
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    os.makedirs(args.staging, exist_ok=True)
    print(f"Downloading {args.repo_id}@{args.revision} -> staging {args.staging} then copy -> {args.dest}")

    from huggingface_hub import snapshot_download

    # Stage on the NODE's LOCAL writable disk first. HF's snapshot_download writes a
    # `.cache/huggingface` staging dir inside local_dir AND checks free disk space via
    # os.statvfs, which reports "0.00 MB free" on the ADLS/FUSE mount (df lies on FUSE),
    # stalling the download. Local node disk passes that check, then we copy to the mount.
    out = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.staging,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.py",
            "*.txt",
            "*.model",
            "*.tokenizer",
            "*.merges",
            "*.vocab",
            "*tokenizer*",
            "*depth_config*",
            "*openjet_runtime*",
        ],
    )
    print("Snapshot staged at:", out)

    print(f"Copying staged files -> {args.dest} ...")
    import shutil
    n_files = 0
    n_bytes = 0
    for root, _, files in os.walk(args.staging):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, args.staging)
            dst = os.path.join(args.dest, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n_files += 1
            n_bytes += os.path.getsize(src)
            print(f"  copied {rel} ({os.path.getsize(src)/1e9:.2f} GB)")
    print(f"COPIED {n_files} files, {n_bytes/1e9:.2f} GB total")

    # Sanity check the safetensors landed on the mount
    n_sf = 0
    for root, _, files in os.walk(args.dest):
        for fn in files:
            if fn.endswith(".safetensors"):
                n_sf += 1
                p = os.path.join(root, fn)
                print(f"  weight: {p} ({os.path.getsize(p)/1e9:.2f} GB)")
    if n_sf == 0:
        print("WARNING: no .safetensors files found on dest")
    else:
        print(f"DEST_OK {n_sf} safetensors on {args.dest}")
    print("DONE")


if __name__ == "__main__":
    main()
