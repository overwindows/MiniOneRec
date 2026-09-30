"""
JEV noul pairwise (preferred/dispreferred) dataset for RLCD.

Builds ground-truth preference pairs from MIND behaviors.tsv: within each
impression, every clicked candidate (preferred) is paired with a sampled
unclicked candidate (dispreferred). Both are rendered in the OpenJev noul
contract (shared state + candidate + A/B task json). The pair shares the same
reading history; only the candidate article changes.

This is the "RL from Contrastive Distillation" (RLCD) data shape: the contrast
comes from the *model's own* P(A) gap between the pair (self-distillation), and
the preference itself (clicked > unclicked) is ground truth — no external
teacher is required.

Each example __getitem__ returns two tokenized sequences sharing the same
history portion:
    input_ids_pos / labels_pos / attention_mask_pos
    input_ids_neg / labels_neg / attention_mask_neg
with a single A/B answer token at the end of each (A=click for pos, B=no-click
for neg). A contrastive trainer reads both halves.

To keep memory bounded and match the pointwise trainer, both members share the
same history+task prefix; only the candidate slice differs. Batching is done by
a collator that pads pos and neg independently.
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset

from data import MINDPointwiseSFTDataset
from evaluate_mind_jev import build_jev_state, render_noul_prompt


class JEVNoulPairwiseDataset(Dataset):
    """Ground-truth (clicked, unclicked) candidate pairs in noul format.

    Reuses MINDPointwiseSFTDataset's news loading and hard-negative sampling to
    obtain candidate pairs, but instead of scoring each candidate separately it
    groups positives (preferred) with same-impression negatives (dispreferred).
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 30,
        neg_ratio: float = 1.0,
        hard_neg_ratio: float = 0.5,
        use_abstract: bool = False,
        use_subcategory: bool = False,
        use_chat_template: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed
        self.neg_ratio = neg_ratio
        self.hard_neg_ratio = hard_neg_ratio
        self.max_history = max_history if max_history > 0 else None

        # Reuse the pointwise loader purely for its candidate-pair sampling,
        # then group positives with negatives that share an impression.
        base = MINDPointwiseSFTDataset(
            behaviors_path=behaviors_path,
            news_path=news_path,
            tokenizer=tokenizer,
            max_len=max_len,
            sample=sample,
            seed=seed,
            max_history=max_history,
            neg_ratio=neg_ratio,
            hard_neg_ratio=hard_neg_ratio,
            use_abstract=use_abstract,
            use_subcategory=use_subcategory,
            use_chat_template=use_chat_template,
        )
        self.base = base
        self.samples = self._group_pairs(base)

        print(f"Loaded {len(self.samples)} preference pairs (pos/neg candidate)")
        print(f"  Max len: {max_len}, history cap: {max_history}")

    def _group_pairs(self, base):
        """Group positives and negatives by impression into (pos, neg) pairs."""
        # Index samples by impression id
        by_imp = {}
        for s in base.samples:
            by_imp.setdefault(s['impression_id'], []).append(s)

        rng = random.Random(self.seed)
        pairs = []
        for imp_id, samples in by_imp.items():
            poss = [s for s in samples if s['label'] == 1]
            negs = [s for s in samples if s['label'] == 0]
            if not poss or not negs:
                continue
            for pos in poss:
                # Hard-negative preference: prefer a same-category (hard) neg when available.
                hard = [n for n in negs if self._same_cat(pos, n)]
                pool = hard if hard else negs
                neg = rng.choice(pool)
                # Each pair reuses the SAME history (pos and neg share impression).
                pairs.append({
                    'impression_id': imp_id,
                    'history': pos['history'],
                    'pos_candidate': pos['candidate'],
                    'neg_candidate': neg['candidate'],
                })
        return pairs

    @staticmethod
    def _same_cat(pos, neg):
        return pos['candidate'].get('category', '') == neg['candidate'].get('category', '')

    def _encode_candidate(self, sample, candidate, answer):
        """Render one candidate in noul format and tokenize with a single A/B
        answer token, exactly matching evaluate_mind_jev's encoding."""
        state = build_jev_state(sample['history'], candidate)
        prompt = render_noul_prompt(state)
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tokenizer.encode(
            formatted,
            max_length=self.max_len - 1,
            truncation=True,
            add_special_tokens=False,
        )
        ans_ids = self.tokenizer.encode(answer, add_special_tokens=False)
        if len(ans_ids) != 1:
            raise ValueError(f"answer {answer!r} not single token: {ans_ids}")
        input_ids = (prompt_ids + ans_ids)[:self.max_len]
        labels = ([-100] * len(prompt_ids) + ans_ids)[:self.max_len]
        return input_ids, labels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_ids, pos_lab = self._encode_candidate(sample, sample['pos_candidate'], "A")
        neg_ids, neg_lab = self._encode_candidate(sample, sample['neg_candidate'], "B")
        return {
            'input_ids_pos': torch.tensor(pos_ids, dtype=torch.long),
            'labels_pos': torch.tensor(pos_lab, dtype=torch.long),
            'attention_mask_pos': torch.ones(len(pos_ids), dtype=torch.long),
            'input_ids_neg': torch.tensor(neg_ids, dtype=torch.long),
            'labels_neg': torch.tensor(neg_lab, dtype=torch.long),
            'attention_mask_neg': torch.ones(len(neg_ids), dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)
