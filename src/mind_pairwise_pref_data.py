"""
Pairwise (clicked > unclicked) dataset for contrastive fine-tuning (DPO / RLCD)
on top of the pointwise Yes/No SFT pipeline.

Builds ground-truth preference pairs from MIND behaviors.tsv: within each
impression, every clicked candidate (preferred) is paired with a sampled
unclicked candidate (dispreferred). Both members are rendered in the EXACT
pointwise Yes/No prompt format used by sft_mind_pointwise_ds.py training
(same system prompt, history rendering incl. recency/timestamp flags, chat
template), so a contrastive fine-tune starts from a model whose SFT prompt
distribution is unchanged.

Each __getitem__ returns:
    input_ids_pos / labels_pos / attention_mask_pos   (answer " Yes")
    input_ids_neg / labels_neg / attention_mask_neg   (answer " No")
The pair shares the same history; only the candidate differs.

The trainer reads the answer-boundary Yes/No logits from both halves and
applies a contrastive objective (DPO logistic or RLCD hinge).
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset

from data import MINDPointwiseSFTDataset


class MINDPairwisePrefDataset(Dataset):
    """Ground-truth (clicked, unclicked) candidate pairs in pointwise Yes/No format.

    Reuses MINDPointwiseSFTDataset's news loading, prompt building (including
    use_recency / use_impression_timestamp / use_profile_summary / use_entity
    flags) and hard-negative sampling, then groups positives with
    same-impression negatives into preference pairs.
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
        use_recency: bool = False,
        use_profile_summary: bool = False,
        use_impression_timestamp: bool = False,
        use_entity: bool = False,
        history_selection: str = "recent",
        history_format: str = "flat",
        pairs_per_positive: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seed = seed

        base = MINDPointwiseSFTDataset(
            behaviors_path=behaviors_path,
            news_path=news_path,
            tokenizer=tokenizer,
            max_len=max_len,
            sample=-1,  # limit applied to pairs below, not to flattened samples
            seed=seed,
            max_history=max_history,
            neg_ratio=neg_ratio,
            hard_neg_ratio=hard_neg_ratio,
            use_abstract=use_abstract,
            use_subcategory=use_subcategory,
            use_chat_template=use_chat_template,
            use_recency=use_recency,
            use_profile_summary=use_profile_summary,
            use_impression_timestamp=use_impression_timestamp,
            use_entity=use_entity,
            history_selection=history_selection,
            history_format=history_format,
        )
        self.base = base
        self.samples = self._group_pairs(base, pairs_per_positive)
        # Sampling the base dataset's shuffled pos/neg samples would orphan
        # most positives from their same-impression negative; cap pairs instead.
        if sample > 0:
            self.samples = self.samples[:sample]

        print(f"Loaded {len(self.samples)} preference pairs (pos/neg candidate)")
        print(f"  Max len: {max_len}, history cap: {max_history}")

    def _group_pairs(self, base, pairs_per_positive):
        """Group positives and negatives by impression into (pos, neg) pairs."""
        by_imp = {}
        for s in base.samples:
            by_imp.setdefault(s['impression_id'], []).append(s)

        rng = random.Random(self.seed + 1)
        pairs = []
        for imp_id, samples in by_imp.items():
            poss = [s for s in samples if s['label'] == 1]
            negs = [s for s in samples if s['label'] == 0]
            if not poss or not negs:
                continue
            for pos in poss:
                # Prefer a same-category (hard) negative when available.
                hard = [n for n in negs if n['candidate'].get('category', '')
                        == pos['candidate'].get('category', '')]
                pool = hard if hard else negs
                for _ in range(pairs_per_positive):
                    neg = rng.choice(pool)
                    pairs.append({
                        'impression_id': imp_id,
                        'history': pos['history'],
                        'impression_timestamp': pos.get('impression_timestamp', ''),
                        'pos_candidate': pos['candidate'],
                        'neg_candidate': neg['candidate'],
                    })
        rng.shuffle(pairs)
        return pairs

    def _encode(self, sample, candidate, label):
        """Render one candidate in the pointwise Yes/No format, delegating to
        MINDPointwiseSFTDataset.encode_example so the format (system prompt,
        history rendering, over-length handling) cannot drift from SFT."""
        input_ids, labels, _ = self.base.encode_example(
            sample['history'], candidate, label,
            impression_timestamp=sample.get('impression_timestamp', ''),
        )
        return input_ids, labels

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_ids, pos_lab = self._encode(sample, sample['pos_candidate'], 1)
        neg_ids, neg_lab = self._encode(sample, sample['neg_candidate'], 0)
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


class PairwisePrefCollator:
    """Collate pos/neg halves independently, then stack into a single 2B batch.

    Output keys: input_ids, labels, attention_mask (first B rows = preferred,
    last B rows = dispreferred) — the trainer splits on batch size.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _collate_side(self, features, key):
        import transformers
        feats = [
            {
                'input_ids': f[f'input_ids_{key}'],
                'labels': f[f'labels_{key}'],
                'attention_mask': f[f'attention_mask_{key}'],
            }
            for f in features
        ]
        collated = transformers.DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            padding=True,
        )(feats)
        return collated

    def __call__(self, features):
        pos = self._collate_side(features, 'pos')
        neg = self._collate_side(features, 'neg')
        # Pad the shorter side to match so both halves can be stacked into a
        # single 2B batch (DataCollatorForSeq2Seq pads each side independently
        # to its own max length).
        max_len = max(pos['input_ids'].shape[1], neg['input_ids'].shape[1])
        pad_id = self.tokenizer.pad_token_id

        def _pad(t, fill):
            # Left padding (padding_side=left): pads go at the front. Labels
            # pad with -100 (ignored), inputs with the pad token, attention
            # mask with 0 — anything else corrupts real_len/predict_pos in
            # the trainer's answer-boundary indexing.
            cur = t.shape[1]
            if cur == max_len:
                return t
            pad_amt = max_len - cur
            pad = torch.full((t.shape[0], pad_amt), fill, dtype=t.dtype)
            return torch.cat([pad, t], dim=1)

        return {
            'input_ids': torch.cat([_pad(pos['input_ids'], pad_id),
                                    _pad(neg['input_ids'], pad_id)], dim=0),
            'labels': torch.cat([_pad(pos['labels'], -100),
                                 _pad(neg['labels'], -100)], dim=0),
            'attention_mask': torch.cat([_pad(pos['attention_mask'], 0),
                                         _pad(neg['attention_mask'], 0)], dim=0),
        }