"""
JEV noul pointwise dataset for training APUS-OpenJev on MIND.

Reuses MINDPointwiseSFTDataset's news loading, history limit and negative sampler.
Both prompt modes share no-thinking chat formatting with evaluate_mind_jev.py:
legacy preserves the historical yes/no descriptions and truncation; native_click
uses a click proposition and canonical Noul true/false descriptions, and checks
all sampled prompts before training rather than silently truncating them.

The appended target is one token: A for click or B for no-click. Full-vocabulary
CE supervises that token; candidate-only loss removes it before the forward pass
and compares final-prompt A/B logits. These are different objectives.
"""

import sys
import os

# Ensure `data` (repo root) is importable from this file under src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import build_jev_state
from jev_decision import (
    PROMPT_CONTRACTS, decision_token_ids, encode_decision_prompt, render_noul_prompt,
)


class JEVNoulPointwiseDataset(MINDPointwiseSFTDataset):
    """MINDPointwiseSFTDataset with the OpenJev noul prompt/target contract."""

    def __init__(self, *args, prompt_contract="legacy", **kwargs):
        if prompt_contract not in PROMPT_CONTRACTS:
            raise ValueError(f"Unknown JEV prompt contract: {prompt_contract}")
        self.prompt_contract = prompt_contract
        super().__init__(*args, **kwargs)
        if self.max_len < 2:
            raise ValueError("JEV cutoff must reserve a prompt token and an answer token")
        self.decision_token_ids = decision_token_ids(self.tokenizer)
        if self.prompt_contract == "native_click":
            for index in range(len(self.samples)):
                self[index]
            print(f"Validated {len(self.samples)} native click prompts before training")

    def _build_noul_pair(self, sample):
        """Return (prompt_prefix, answer_token) for a pointwise sample.

        Answer always "A" (yes/click) or "B" (no/no-click); P(A) is the rank
        score used by evaluate_mind_jev.py.
        """
        state = build_jev_state(sample['history'], sample['candidate'])
        prompt = render_noul_prompt(state, self.prompt_contract)
        answer = "A" if sample['label'] == 1 else "B"
        return prompt, answer

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, answer = self._build_noul_pair(sample)

        prompt_ids = encode_decision_prompt(
            self.tokenizer, prompt,
            max_length=min(self.max_len - 1, 8192) if self.prompt_contract == "native_click" else self.max_len - 1,
            truncate=self.prompt_contract == "legacy",
        )

        target_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if len(target_ids) != 1:
            # A/B must be single-token for next-token classification to match eval.
            raise ValueError(f"answer {answer!r} not single token: {target_ids}")

        input_ids = (prompt_ids + target_ids)[:self.max_len]
        train_labels = ([-100] * len(prompt_ids) + target_ids)[:self.max_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(train_labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
        }
