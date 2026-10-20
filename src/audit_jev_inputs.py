"""Audit every sampled training pair and every evaluation candidate without truncation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from evaluate_mind_jev import build_jev_state, parse_behaviors_line
from jev_decision import PROMPT_CONTRACTS, encode_decision_prompt, render_noul_prompt
from jev_noul_data import JEVNoulPointwiseDataset
from mind_utils import load_news


def audit_states(states, tokenizer, budget, batch_size=128):
    lengths = {contract: [] for contract in PROMPT_CONTRACTS}
    fingerprint = hashlib.sha256()
    pending = []
    count = 0

    def consume():
        for contract in PROMPT_CONTRACTS:
            formatted = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": render_noul_prompt(state, contract)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for state in pending
            ]
            encoded = tokenizer(formatted, add_special_tokens=False, truncation=False)
            lengths[contract].extend(len(ids) for ids in encoded["input_ids"])

    for identity, state in states:
        fingerprint.update(json.dumps(identity, ensure_ascii=False).encode("utf-8") + b"\n")
        fingerprint.update(state.encode("utf-8") + b"\n")
        if count == 0:
            for contract in PROMPT_CONTRACTS:
                encode_decision_prompt(tokenizer, render_noul_prompt(state, contract))
        pending.append(state)
        count += 1
        if len(pending) == batch_size:
            consume()
            pending.clear()
            if count % 8192 == 0:
                print(f"Audited {count} candidate prompts in both modes", flush=True)
    if pending:
        consume()
    if not count:
        raise ValueError("Audit input contains no candidates")
    return {
        "candidate_count": count,
        "ordered_input_sha256": fingerprint.hexdigest(),
        "prompt_budget": budget,
        "contracts": {
            contract: {
                "min": min(values),
                "max": max(values),
                "percentiles_50_95_99": np.percentile(values, [50, 95, 99]).tolist(),
                "over_budget": sum(value > budget for value in values),
            }
            for contract, values in lengths.items()
        },
    }


def dev_states(behaviors_path, news, max_history, max_impressions, coverage):
    coverage.update(impressions=0, missing_candidates=0, missing_history=0)
    with open(behaviors_path, encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_behaviors_line(line)
            if parsed is None:
                raise ValueError("Malformed dev behavior row")
            impression_id, history_ids, candidates = parsed
            if not candidates:
                raise ValueError(f"Empty impression {impression_id}")
            if max_history > 0:
                history_ids = history_ids[-max_history:]
            coverage["missing_history"] += sum(nid not in news for nid in history_ids)
            history = [news[nid] for nid in history_ids if nid in news]
            for nid, label in candidates:
                if nid not in news:
                    coverage["missing_candidates"] += 1
                    raise ValueError(f"Dev candidate absent from news: {nid}")
                yield (impression_id, nid, label), build_jev_state(history, news[nid])
            coverage["impressions"] += 1
            if max_impressions and coverage["impressions"] >= max_impressions:
                break
    if max_impressions and coverage["impressions"] != max_impressions:
        raise ValueError("Dev data contains fewer impressions than requested")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--max-impressions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cutoff_len < 2 or args.max_history < 0 or args.max_impressions < 1:
        parser.error("Require cutoff >= 2, history >= 0 and impressions >= 1")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    root = Path(args.data_root)
    dataset = JEVNoulPointwiseDataset(
        str(root / "train" / "behaviors.tsv"), str(root / "train" / "news.tsv"),
        tokenizer, max_len=args.cutoff_len, sample=-1, seed=args.seed,
        max_history=args.max_history, neg_ratio=1.0, hard_neg_ratio=0.5,
        use_abstract=True, prompt_contract="legacy",
    )
    report = {
        "seed": args.seed, "max_history": args.max_history, "use_abstract": True,
        "cutoff_len": args.cutoff_len, "neg_ratio": 1.0, "hard_neg_ratio": 0.5,
        "tokenizer_files_sha256": {
            name: hashlib.sha256((Path(args.tokenizer) / name).read_bytes()).hexdigest()
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "data_sha256": {
            f"{split}/{name}": hashlib.sha256((root / split / name).read_bytes()).hexdigest()
            for split in ("train", "dev") for name in ("news.tsv", "behaviors.tsv")
        },
    }
    report["train"] = audit_states(
        (((sample["impression_id"], sample["candidate_id"], sample["label"]),
          build_jev_state(sample["history"], sample["candidate"]))
         for sample in dataset.samples),
        tokenizer, min(args.cutoff_len - 1, 8192),
    )
    report["train"]["positives"] = sum(sample["label"] for sample in dataset.samples)
    del dataset
    coverage = {}
    report["dev"] = audit_states(
        dev_states(root / "dev" / "behaviors.tsv", load_news(str(root / "dev" / "news.tsv"), True),
                   args.max_history, args.max_impressions, coverage),
        tokenizer, min(args.cutoff_len - 1, 8192),
    )
    report["dev"].update(coverage)
    report["all_prompts_fit"] = all(
        contract["over_budget"] == 0
        for split in ("train", "dev")
        for contract in report[split]["contracts"].values()
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_prompts_fit"]:
        raise SystemExit("Audit rejected shared prompt budget; no training should be submitted")


if __name__ == "__main__":
    main()
