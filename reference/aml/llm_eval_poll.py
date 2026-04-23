import argparse
import json
import os
import subprocess
import sys
import time


def _load_seen(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("seen", []))


def _save_seen(path, seen):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"seen": sorted(seen)}, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, path)


def _list_checkpoints(root):
    if not os.path.isdir(root):
        return []
    names = []
    for entry in os.listdir(root):
        if entry.startswith("checkpoint-"):
            full = os.path.join(root, entry)
            if os.path.isdir(full):
                names.append(full)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description="Poll for new checkpoints and run LLM evals.")
    parser.add_argument("--checkpoints_dir", required=True, help="Directory with checkpoint-* folders.")
    parser.add_argument("--output_dir", required=True, help="Output directory for eval results.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval",
        help="Comma-separated lm-eval tasks.",
    )
    parser.add_argument("--poll_interval", type=int, default=600, help="Seconds between polls.")
    parser.add_argument("--once", action="store_true", help="Run a single scan then exit.")
    parser.add_argument("--batch_size", default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--device", default="cuda", help="Device for lm-eval.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples.")
    parser.add_argument("--limit", type=float, default=None, help="Optional eval limit.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seen_path = os.path.join(args.output_dir, "seen_checkpoints.json")
    seen = _load_seen(seen_path)

    while True:
        checkpoints = _list_checkpoints(args.checkpoints_dir)
        new_ckpts = [ckpt for ckpt in checkpoints if ckpt not in seen]

        for ckpt in new_ckpts:
            cmd = [
                sys.executable,
                "llm_eval.py",
                "--model_path",
                ckpt,
                "--tasks",
                args.tasks,
                "--output_dir",
                args.output_dir,
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--num_fewshot",
                str(args.num_fewshot),
            ]
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])

            print(f"Evaluating checkpoint: {ckpt}")
            subprocess.run(cmd, check=True)
            seen.add(ckpt)
            _save_seen(seen_path, seen)

        if args.once:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
