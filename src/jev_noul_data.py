"""
JEV noul pointwise dataset for training APUS-OpenJev on MIND.

Subclasses the repo's MINDPointwiseSFTDataset (data.py) so it reuses the exact
news loading, history truncation, positive/negative expansion and hard-negative
sampling. It only overrides prompt construction to emit the **OpenJev noul
contract** that src/evaluate_mind_jev.py scores with:

    Shared state:
    A user read these news articles:
    1. [cat] title abstract
    ...
    (No reading history)

    Candidate article:
    [cat] title abstract

    {"primitive": "noul", "instructions": "...", "criteria":
     [{"label": "A", "description": "yes"}, {"label": "B", "description": "no"}]}
    Return only the selected letter: A, B.
    Answer:<A/B>

Target token is single-token: "A" for a click, "B" for no-click -> P(A)=P(yes),
which is exactly the score evaluate_mind_jev.py ranks candidates by. Using the
same contract for training and eval keeps the metrics comparable.
"""

import sys
import os

# Ensure `data` (repo root) is importable from this file under src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import build_jev_state, render_noul_prompt


class JEVNoulPointwiseDataset(MINDPointwiseSFTDataset):
    """MINDPointwiseSFTDataset with the OpenJev noul prompt/target contract."""

    def _build_noul_pair(self, sample):
        """Return (prompt_prefix, answer_token) for a pointwise sample.

        Answer always "A" (yes/click) or "B" (no/no-click); P(A) is the rank
        score used by evaluate_mind_jev.py.
        """
        state = build_jev_state(sample['history'], sample['candidate'])
        prompt = render_noul_prompt(state)
        answer = "A" if sample['label'] == 1 else "B"
        return prompt, answer

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, answer = self._build_noul_pair(sample)

        # ALWAYS wrap the noul prompt in the JEV chat template as a user message,
        # matching evaluate_mind_jev.JevScorer._encode exactly (which does
        # apply_chat_template(..., add_generation_prompt=True,
        # enable_thinking=False)). If train and eval tokenize the prompt
        # differently, the model's P(A) is not comparable to the dev AUC.
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(
            formatted_prompt,
            max_length=self.max_len - 1,
            truncation=True,
            add_special_tokens=False,
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
