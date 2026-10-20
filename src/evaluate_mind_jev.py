"""
Evaluate MIND using OpenJev (APUS-OpenJev) as a point-wise decision model.

OpenJev is TypeSafe's System-One decision model. Its OpenJet runtime exposes a
`noul` (binary yes/no) primitive: given a "shared state" plus a candidate, it
returns per-candidate probabilities over {yes, no}. For MIND news recommendation,
each (user history, candidate article) is scored by P(yes), and within an
impression we rank candidates by that P(yes). AUC is then computed exactly as in
calc_mind_metrics.py.

This scorer uses a direct full-depth HF forward in the repository's environment,
not the released OpenJet executor, which pins a different Transformers version.
It does not establish exact equivalence to that runtime or exercise its low exit.
The model is a pre-staged local snapshot, loaded with local_files_only.

Prompt mode auto restores checkpoint metadata, defaulting to legacy for historical
checkpoints. Legacy preserves the old yes/no descriptions; native_click uses a
click proposition and canonical Noul true/false criteria. Both use validated
single-token A/B labels at the actual chat answer boundary and rank by the
candidate-normalized P(A), not the full-vocabulary probability of A.

Usage:
  python src/evaluate_mind_jev.py \
      --model_path /path/to/APUS-OpenJev-v1-4B \
      --behaviors_path .../dev/behaviors.tsv \
      --news_path .../dev/news.tsv \
      --output_file dev_jev_predictions.txt
"""

import argparse
import math
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from mind_utils import auc_score, mrr_score, ndcg_score, load_news
from jev_decision import (
    PROMPT_CONTRACTS, decision_token_ids, encode_decision_prompt, render_noul_prompt,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


LABELS = tuple("ABCDEFGHIJKLMNOP")


def build_jev_state(history: List[Dict[str, str]], candidate: Dict[str, str]) -> str:
    """Build the "Shared state" block = user history + candidate, plain text.

    Mirrors the pointwise prompt content used across the repo (category tags,
    history list, candidate block) but rendered as the decision model's shared
    state rather than a Yes/No Q&A.
    """
    lines = ["A user read these news articles:"]
    if history:
        for i, h in enumerate(history, 1):
            cat = h.get("category", "General")
            lines.append(f"{i}. [{cat}] {h['text']}")
    else:
        lines.append("(No reading history)")
    lines.append("")
    lines.append("Candidate article:")
    cat = candidate.get("category", "General")
    lines.append(f"[{cat}] {candidate['text']}")
    return "\n".join(lines)


class JevScorer:
    """Full-depth Qwen3.5 candidate-token scorer with an explicit prompt mode."""

    def __init__(self, model_path: str, device: str = "cuda:0", dtype: str = "bfloat16",
                 prompt_contract: str = "auto"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.device = device
        torch_dtype = getattr(torch, dtype)
        local_kwargs = {"local_files_only": True}

        config = AutoConfig.from_pretrained(model_path, **local_kwargs)
        self.config = config
        self.prompt_contract = (
            getattr(config, "jev_prompt_contract", "legacy")
            if prompt_contract == "auto" else prompt_contract
        )
        if self.prompt_contract not in PROMPT_CONTRACTS:
            raise ValueError(f"Unknown JEV prompt contract: {self.prompt_contract}")
        print(f"JEV prompt contract: {self.prompt_contract}")
        loader = AutoModelForCausalLM
        if config.model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration
            loader = Qwen3_5ForConditionalGeneration

        self.model = loader.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        # OpenJev ships its chat template in chat_template.jinja (tokenizer_config has
        # no inline chat_template). If the tokenizer has none, load it from the file.
        if not getattr(self.tokenizer, "chat_template", None):
            import os as _os
            jinja = _os.path.join(model_path, "chat_template.jinja")
            if _os.path.exists(jinja):
                with open(jinja, encoding="utf-8") as f:
                    self.tokenizer.chat_template = f.read()

        self.cand_tokens = list(decision_token_ids(self.tokenizer))

    def _encode(self, state: str):
        return encode_decision_prompt(
            self.tokenizer, render_noul_prompt(state, self.prompt_contract),
            max_length=8192 if self.prompt_contract == "native_click" else None,
        )

    def _max_len(self) -> int:
        if sys.platform != "win32":
            return getattr(self.config, "max_position_embeddings", 262144)
        return 8192

    @torch.inference_mode()
    def score_yes(self, state: str) -> float:
        """Return P(yes)=softmax(logits)[A] for a single noul prompt."""
        ids = self._encode(state)
        if not ids or len(ids) > self._max_len():
            return -1.0  # too long; caller treats as abstain
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=torch.arange(len(ids), device=self.device).unsqueeze(0),
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = out.logits[0, -1].float()
        cand_logits = logits[self.cand_tokens]
        probs = torch.softmax(cand_logits, -1)
        return float(probs[0].item())

    def _score_chunk(self, seqs, idx):
        """One padded forward over a chunk of encoded sequences."""
        L = max(len(s) for s in seqs)
        batch = torch.zeros((len(seqs), L), dtype=torch.long, device=self.device)
        mask = torch.zeros((len(seqs), L), dtype=torch.long, device=self.device)
        pos_ids = torch.zeros((len(seqs), L), dtype=torch.long, device=self.device)
        for r, s in enumerate(seqs):
            off = L - len(s)
            batch[r, off:] = torch.tensor(s, dtype=torch.long, device=self.device)
            mask[r, off:] = 1
            pos_ids[r, off:] = torch.arange(len(s), device=self.device)
        out = self.model(
            input_ids=batch,
            attention_mask=mask,
            position_ids=pos_ids,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :].float()
        cand_logits = logits[:, self.cand_tokens]
        probs = torch.softmax(cand_logits, -1)
        return {r: float(probs[j, 0].item()) for j, r in enumerate(idx)}

    @torch.inference_mode()
    def score_yes_batch(self, states: List[str], chunk_size: int = 32) -> List[float]:
        """Score many candidate states in padded forwards, chunked by chunk_size.

        This avoids the serial per-candidate forward passes that made the naive
        loop ~10-30x slower, while keeping each forward small enough to avoid
        OOM on large-impression batches (the delta-rule linear attention
        materializes a workspace ~BxLxR; a whole-impression batch is wasteful).
        Paddings go on the LEFT so the final real token of each row carries its
        own answer-boundary logit (logits_to_keep=1). Returns P(yes) per state;
        -1.0 marks an oversized/abstained candidate.
        """
        max_len = self._max_len()
        seqs = []
        idx = []
        for i, s in enumerate(states):
            ids = self._encode(s)
            if not ids or len(ids) > max_len:
                continue
            seqs.append(ids)
            idx.append(i)
        if not seqs:
            return [-1.0] * len(states)
        result = [-1.0] * len(states)
        for start in range(0, len(seqs), chunk_size):
            chunk_seqs = seqs[start:start + chunk_size]
            chunk_idx = idx[start:start + chunk_size]
            for j, prob in self._score_chunk(chunk_seqs, chunk_idx).items():
                result[j] = prob
        return result


def parse_behaviors_line(line: str):
    parts = line.strip().split("\t")
    if len(parts) < 5:
        return None
    impression_id = parts[0]
    history_ids = parts[3].split()
    impressions = []
    for imp in parts[4].split():
        if "-" not in imp:
            continue
        nid, label = imp.rsplit("-", 1)
        impressions.append((nid, int(label)))
    return impression_id, history_ids, impressions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--behaviors_path", required=True)
    parser.add_argument("--news_path", required=True)
    parser.add_argument("--use_abstract", action="store_true")
    parser.add_argument("--max_history", type=int, default=30)
    parser.add_argument("--max_impressions", type=int, default=0)
    parser.add_argument("--prompt_contract", choices=("auto", *PROMPT_CONTRACTS), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", default="")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    news = load_news(args.news_path, args.use_abstract)
    print(f"Loaded {len(news)} news articles")

    scorer = JevScorer(args.model_path, device=args.device, prompt_contract=args.prompt_contract)
    print("JEV scorer ready")

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    aucs, mrrs, ndcg5, ndcg10 = [], [], [], []
    predictions = []
    count = 0

    with open(args.behaviors_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating impressions"):
            parsed = parse_behaviors_line(line)
            if parsed is None:
                continue
            impression_id, history_ids, impressions = parsed
            if args.max_history > 0:
                history_ids = history_ids[-args.max_history:]

            if not impressions:
                continue

            history_objs = [news[nid] for nid in history_ids if nid in news]

            labels = []
            candidate_ids = []
            states = []
            for nid, label in impressions:
                if nid not in news:
                    continue
                candidate_ids.append(nid)
                labels.append(label)
                states.append(build_jev_state(history_objs, news[nid]))
            # Batch all candidates of the impression into one padded forward.
            raw = scorer.score_yes_batch(states)
            scores = [0.5 if s < 0 else s for s in raw]  # abstain/too-long -> neutral

            if not candidate_ids:
                continue

            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5.append(ndcg_score(labels, scores, 5))
                ndcg10.append(ndcg_score(labels, scores, 10))

            if args.output_file:
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranked_ids = [candidate_ids[i] for i in ranked]
                predictions.append((impression_id, ranked_ids))

            count += 1
            if args.max_impressions and count >= args.max_impressions:
                break

    if args.output_file and predictions:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for imp_id, ranked in predictions:
                f.write(f"{imp_id}\t[{' '.join(ranked)}]\n")
        print(f"Saved predictions to {args.output_file}")

    print("\nFinal Results:")
    print(f"Processed {count} impressions")
    if aucs:
        print(f"AUC:     {_avg(aucs):.4f}")
        print(f"MRR:     {_avg(mrrs):.4f}")
        print(f"nDCG@5:  {_avg(ndcg5):.4f}")
        print(f"nDCG@10: {_avg(ndcg10):.4f}")
    else:
        print("No labeled impressions processed")


if __name__ == "__main__":
    main()
