import argparse
import json
import os
import time


def _build_model_args(model_path, tokenizer_path, dtype, trust_remote_code, max_length):
    args = [f"pretrained={model_path}"]
    if tokenizer_path:
        args.append(f"tokenizer={tokenizer_path}")
    if dtype:
        args.append(f"dtype={dtype}")
    if trust_remote_code:
        args.append("trust_remote_code=true")
    if max_length is not None:
        args.append(f"max_length={max_length}")
    return ",".join(args)


def main():
    parser = argparse.ArgumentParser(description="Run LLM capability evals via lm-eval-harness.")
    parser.add_argument("--model_path", required=True, help="HF model path or checkpoint directory.")
    parser.add_argument("--tokenizer_path", default="", help="Optional tokenizer path.")
    parser.add_argument(
        "--tasks",
        default="mmlu,hellaswag,arc_challenge,winogrande,gsm8k,ifeval",
        help="Comma-separated lm-eval task names.",
    )
    parser.add_argument("--output_dir", default="llm_eval", help="Where to write results JSON.")
    parser.add_argument("--batch_size", default="auto", help="Batch size (or 'auto').")
    parser.add_argument("--device", default="cuda", help="Device for lm-eval (e.g., cuda, cpu).")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g., bfloat16, float16).")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Few-shot examples.")
    parser.add_argument("--limit", type=float, default=None, help="Optional eval limit.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--max_length", type=int, default=None, help="Optional max length.")
    args = parser.parse_args()

    try:
        from lm_eval import evaluator
    except Exception as exc:
        raise SystemExit(
            "lm-eval-harness is not installed. Install with: pip install -r requirements-eval.txt"
        ) from exc

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        raise SystemExit("No tasks provided. Use --tasks with comma-separated task names.")

    model_args = _build_model_args(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path or None,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
    )

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=args.batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(os.path.abspath(args.model_path))
    out_path = os.path.join(args.output_dir, f"{model_name}_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    print(f"Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
