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
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    print(f"Downloading {args.repo_id}@{args.revision} -> {args.dest}")

    from huggingface_hub import snapshot_download

    out = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.dest,
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
    print("Snapshot downloaded to:", out)

    # Sanity check the safetensors landed
    n_sf = 0
    for root, _, files in os.walk(args.dest):
        for fn in files:
            if fn.endswith(".safetensors"):
                n_sf += 1
                p = os.path.join(root, fn)
                print(f"  weight: {p} ({os.path.getsize(p)/1e9:.2f} GB)")
    if n_sf == 0:
        print("WARNING: no .safetensors files found in download")
    print("DONE")


if __name__ == "__main__":
    main()
