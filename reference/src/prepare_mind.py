#!/usr/bin/env python3
"""Download and extract the MIND dataset into MIND_ROOT/{train,dev,test}.

NOTE: Microsoft has restricted public access to the original Azure blob storage URLs.
You need to download MIND dataset from the official sources:

1. Kaggle: https://www.kaggle.com/datasets/arashnic/mind-news-dataset
2. Official MIND page: https://msnews.github.io/

This script will work with locally downloaded ZIP files or alternative URLs.
"""
import argparse
import os
import shutil
import sys
import zipfile

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request


# Alternative download URLs (these may also require authentication)
# Users should download manually from https://www.kaggle.com/datasets/arashnic/mind-news-dataset
MIND_URLS = {
    "small": {
        "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
        "dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
        "test": "https://mind201910small.blob.core.windows.net/release/MINDsmall_test.zip",
    },
    "large": {
        "train": "https://mind201910large.blob.core.windows.net/release/MINDlarge_train.zip",
        "dev": "https://mind201910large.blob.core.windows.net/release/MINDlarge_dev.zip",
        "test": "https://mind201910large.blob.core.windows.net/release/MINDlarge_test.zip",
    },
}

# Kaggle alternative (requires kaggle CLI: pip install kaggle)
KAGGLE_DATASET = "arashnic/mind-news-dataset"


def _progress_hook(label):
    def hook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(downloaded / total_size, 1.0) * 100
        sys.stdout.write(f"\r{label}: {pct:5.1f}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")
    return hook


def _download(url, dst, force=False):
    if os.path.exists(dst) and not force:
        print(f"Using cached file: {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Downloading {url}")

    try:
        if HAS_REQUESTS:
            # Try with requests library (better error handling)
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(dst, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pct = (downloaded / total_size) * 100
                            sys.stdout.write(f"\r{os.path.basename(dst)}: {pct:5.1f}%")
                            sys.stdout.flush()
                    sys.stdout.write("\n")
        else:
            # Fallback to urllib
            urllib.request.urlretrieve(url, dst, reporthook=_progress_hook(os.path.basename(dst)))
    except Exception as e:
        if os.path.exists(dst):
            os.remove(dst)
        raise RuntimeError(
            f"Failed to download {url}. Error: {e}\n\n"
            "NOTE: Microsoft has restricted public access to MIND dataset URLs.\n"
            "Please download manually from one of these sources:\n"
            "  1. Kaggle: https://www.kaggle.com/datasets/arashnic/mind-news-dataset\n"
            "  2. Official MIND page: https://msnews.github.io/\n"
            "  3. Use the --local flag to extract from locally downloaded ZIP files"
        )


def _find_mind_dir(root):
    for dirpath, _, filenames in os.walk(root):
        if "behaviors.tsv" in filenames and "news.tsv" in filenames:
            return dirpath
    return None


def _extract(zip_path, extract_root, target_dir, force=False):
    if os.path.isdir(target_dir) and os.listdir(target_dir) and not force:
        print(f"Skipping extract; target exists: {target_dir}")
        return
    if os.path.isdir(target_dir) and force:
        shutil.rmtree(target_dir)
    os.makedirs(extract_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)
    src_dir = _find_mind_dir(extract_root)
    if not src_dir:
        raise RuntimeError(f"Could not locate behaviors.tsv/news.tsv in {extract_root}")
    os.makedirs(target_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(target_dir, name)
        if os.path.isdir(src):
            if os.path.exists(dst) and force:
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    print(f"Extracted to: {target_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract MIND dataset",
        epilog="""
NOTE: Microsoft has restricted public access to MIND dataset.
Download manually from:
  - Kaggle: https://www.kaggle.com/datasets/arashnic/mind-news-dataset
  - Official MIND: https://msnews.github.io/

Then use --local to extract from downloaded ZIP files.
        """
    )
    parser.add_argument("--root", default="data/MIND", help="Output root for MIND data.")
    parser.add_argument("--size", choices=("small", "large"), default="small")
    parser.add_argument("--splits", default="train,dev,test", help="Comma-separated splits.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract.")
    parser.add_argument("--local", metavar="ZIP_DIR", help="Extract from local ZIP files instead of downloading.")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits provided.")

    download_dir = os.path.join(args.root, ".downloads")
    extract_dir = os.path.join(args.root, ".extracted")

    if args.local:
        # Extract from locally downloaded ZIP files
        print(f"Extracting from local ZIP files in: {args.local}")

        # Check for Kaggle combined archive first
        kaggle_combined = os.path.join(args.local, "mind-news-dataset.zip")
        if os.path.exists(kaggle_combined):
            print(f"Found Kaggle combined archive: {kaggle_combined}")
            print("Extracting combined archive (contains data directly, not nested ZIPs)...")

            # Extract the Kaggle combined archive directly
            with zipfile.ZipFile(kaggle_combined, "r") as zf:
                # Get all file names in the archive
                all_files = zf.namelist()

                for split in splits:
                    # Look for directories matching this split
                    split_prefix = f"MIND{args.size}_{split}/"
                    split_files = [f for f in all_files if f.startswith(split_prefix)]

                    if not split_files:
                        print(f"Warning: Could not find {split} data in Kaggle archive")
                        print(f"  Looked for: {split_prefix}")
                        continue

                    print(f"Found {len(split_files)} files for {split} split")
                    target_dir = os.path.join(args.root, split)
                    os.makedirs(target_dir, exist_ok=True)

                    # Extract files for this split
                    for file_path in split_files:
                        # Extract to target directory, removing the split prefix
                        if file_path.endswith('/'):
                            continue  # Skip directories
                        # Get just the filename (e.g., "behaviors.tsv" from "MINDsmall_train/behaviors.tsv")
                        filename = os.path.basename(file_path)
                        target_path = os.path.join(target_dir, filename)

                        # Extract the file
                        with zf.open(file_path) as source, open(target_path, 'wb') as target:
                            shutil.copyfileobj(source, target)

                    print(f"Extracted {split} to: {target_dir}")
        else:
            # Try individual split ZIPs (original behavior)
            for split in splits:
                # Try multiple naming conventions
                possible_names = [
                    f"MIND{args.size}_{split}.zip",
                    f"mind_{args.size}_{split}.zip",
                    f"{split}.zip",
                ]

                zip_path = None
                for name in possible_names:
                    candidate = os.path.join(args.local, name)
                    if os.path.exists(candidate):
                        zip_path = candidate
                        break

                if not zip_path:
                    print(f"Warning: Could not find ZIP for {split} in {args.local}")
                    print(f"  Tried: {', '.join(possible_names)}")
                    continue

                print(f"Found: {zip_path}")
                target_dir = os.path.join(args.root, split)
                split_extract = os.path.join(extract_dir, f"{args.size}_{split}")
                _extract(zip_path, split_extract, target_dir, force=args.force)
    else:
        # Try to download from URLs
        urls = MIND_URLS[args.size]
        for split in splits:
            if split not in urls:
                raise SystemExit(f"Unknown split: {split}")
            url = urls[split]
            zip_path = os.path.join(download_dir, f"mind_{args.size}_{split}.zip")
            _download(url, zip_path, force=args.force)
            target_dir = os.path.join(args.root, split)
            split_extract = os.path.join(extract_dir, f"{args.size}_{split}")
            _extract(zip_path, split_extract, target_dir, force=args.force)

    print("MIND dataset processing completed.")


if __name__ == "__main__":
    main()
