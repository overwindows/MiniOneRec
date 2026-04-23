import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from collections import Counter
import json
import random
from tqdm import tqdm
import os
import copy
import logging
import torch.nn.functional as F


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class SFTData(Dataset):
    def __init__(self, train_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        # self.K = K
        self.dedup = dedup
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title'])
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has played the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""

        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,

            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            pass

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
            # print(inputs[-1])

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class MINDPointwiseSFTDataset:
    """
    MIND dataset for point-wise SFT training.

    Each sample is a (history, single_candidate, label) tuple.
    Model learns to predict Yes/No for relevance.
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 0,  # 0 = no limit
        neg_ratio: float = 1.0,  # Ratio of negatives to positives per impression
        use_abstract: bool = False,
        use_chat_template: bool = False,  # Use chat template for instruct models
        use_subcategory: bool = False,  # Include subcategory in [cat/subcat] format
        use_recency: bool = False,  # Mark 5 most recent history items with "(recent)"
        use_profile_summary: bool = False,  # Prepend top-3 category interest summary
        use_impression_timestamp: bool = False,  # Add day/time-of-day context from impression timestamp
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history if max_history > 0 else None
        self.neg_ratio = neg_ratio
        self.use_abstract = use_abstract
        self.use_chat_template = use_chat_template
        self.use_subcategory = use_subcategory
        self.use_recency = use_recency
        self.use_profile_summary = use_profile_summary
        self.use_impression_timestamp = use_impression_timestamp
        self.seed = seed

        # Load news articles
        self.news = self._load_news(news_path)

        # Load and expand behaviors into point-wise samples
        self.samples = self._load_and_expand_behaviors(behaviors_path, sample)

        print(f"Loaded {len(self.samples)} point-wise samples")
        pos_count = sum(1 for s in self.samples if s['label'] == 1)
        neg_count = len(self.samples) - pos_count
        print(f"  Positives: {pos_count}, Negatives: {neg_count}, Ratio: {neg_count/max(pos_count,1):.2f}")

    def _load_news(self, news_path: str):
        """Load news articles from news.tsv"""
        news = {}
        with open(news_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue
                news_id = parts[0]
                category = parts[1] if len(parts) > 1 else ""
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""

                if self.use_abstract and abstract:
                    text = f"{title} {abstract}"
                else:
                    text = title

                news[news_id] = {
                    'text': text,
                    'category': category,
                    'subcategory': parts[2] if len(parts) > 2 else "",
                }
        return news

    def _load_and_expand_behaviors(self, behaviors_path: str, sample_limit: int):
        """Load behaviors and expand into point-wise samples."""
        all_samples = []
        rng = random.Random(self.seed)

        with open(behaviors_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue

                impression_id = parts[0]
                impression_timestamp = parts[2] if len(parts) > 2 else ""
                history_ids = parts[3].split()
                impressions = parts[4].split()

                # Limit history
                if self.max_history is not None:
                    history_ids = history_ids[-self.max_history:]

                # Build history objects
                history = [self.news[nid] for nid in history_ids if nid in self.news]

                # Parse candidates
                positives = []
                negatives = []
                for imp in impressions:
                    if '-' not in imp:
                        continue
                    news_id, label = imp.rsplit('-', 1)
                    if news_id not in self.news:
                        continue
                    if int(label) == 1:
                        positives.append(news_id)
                    else:
                        negatives.append(news_id)

                # Skip if no positives
                if not positives:
                    continue

                # Add all positive samples
                for pos_id in positives:
                    all_samples.append({
                        'impression_id': impression_id,
                        'impression_timestamp': impression_timestamp,
                        'history': history,
                        'candidate_id': pos_id,
                        'candidate': self.news[pos_id],
                        'label': 1
                    })

                # IMPROVED: Hard negative sampling (50% same category, 50% different)
                num_neg_to_sample = int(len(positives) * self.neg_ratio)
                if num_neg_to_sample > 0 and negatives:
                    # Get categories of positive items
                    pos_categories = set([self.news[pid].get('category', '') for pid in positives])

                    # Split negatives by category match
                    hard_negs = []  # Same category
                    easy_negs = []  # Different category
                    for neg_id in negatives:
                        neg_cat = self.news[neg_id].get('category', '')
                        if neg_cat in pos_categories:
                            hard_negs.append(neg_id)
                        else:
                            easy_negs.append(neg_id)

                    # Sample 50/50 mix of hard and easy negatives
                    num_hard = num_neg_to_sample // 2
                    num_easy = num_neg_to_sample - num_hard

                    sampled_negs = []
                    if hard_negs:
                        sampled_negs.extend(rng.sample(hard_negs, min(num_hard, len(hard_negs))))
                    if easy_negs and len(sampled_negs) < num_neg_to_sample:
                        remaining = num_neg_to_sample - len(sampled_negs)
                        sampled_negs.extend(rng.sample(easy_negs, min(remaining, len(easy_negs))))

                    for neg_id in sampled_negs:
                        all_samples.append({
                            'impression_id': impression_id,
                            'impression_timestamp': impression_timestamp,
                            'history': history,
                            'candidate_id': neg_id,
                            'candidate': self.news[neg_id],
                            'label': 0
                        })

        # Shuffle all samples
        rng.shuffle(all_samples)

        # Sample if requested
        if sample_limit > 0 and sample_limit < len(all_samples):
            all_samples = all_samples[:sample_limit]

        return all_samples

    def _build_pointwise_prompt(self, history, candidate, label, impression_timestamp=""):
        """
        Build OPTIMIZED point-wise prompt for binary classification.

        Based on Prompt4NR research (arXiv:2304.05263):
        - Concise format saves ~20 tokens (removes verbose "Role/Task")
        - Category in [brackets] at start for better visibility
        - Natural language question
        - Limit to last 30 history items for focus

        Optional enhancements (all off by default):
        - use_recency: marks 5 most recent history items with "(recent)"
        - use_profile_summary: prepends top-3 category interest summary
        - use_impression_timestamp: adds day/time-of-day context
        """
        prompt = ""

        # Optional: time-of-day / day-of-week context
        if self.use_impression_timestamp and impression_timestamp:
            day, period = self._parse_timestamp(impression_timestamp)
            if day and period:
                prompt += f"Reading time: {day} {period}\n"

        # Optional: user interest profile summary
        if self.use_profile_summary and history:
            cats = [h.get('category', '') for h in history if h.get('category', '')]
            if cats:
                top = Counter(cats).most_common(3)
                prompt += "User interests: " + ", ".join(f"{c} ({n})" for c, n in top) + "\n"

        prompt += "A user read these news articles:\n"

        # User history (already truncated by max_history in data loader)
        if history:
            recency_cutoff = max(0, len(history) - 5) if self.use_recency else len(history)
            for i, h in enumerate(history, 1):
                cat = h.get('category', 'General')
                tag = " (recent)" if self.use_recency and (i - 1) >= recency_cutoff else ""
                prompt += f"{i}. [{cat}] {h['text']}{tag}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\n"

        # Candidate article - category first in brackets
        prompt += "Candidate article:\n"
        cat = candidate.get('category', 'General')
        prompt += f"[{cat}] {candidate['text']}\n"

        prompt += "\n"
        # Simple, natural question
        prompt += "Will this user read this article? Answer:"

        # Target
        target = " Yes" if label == 1 else " No"

        return prompt, target

    @staticmethod
    def _parse_timestamp(ts: str):
        """Parse MIND timestamp string into (day_of_week, time_period) or (None, None)."""
        try:
            from datetime import datetime
            dt = datetime.strptime(ts.strip(), "%m/%d/%Y %I:%M:%S %p")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day = days[dt.weekday()]
            h = dt.hour
            if 6 <= h < 12:
                period = "morning"
            elif 12 <= h < 18:
                period = "afternoon"
            elif 18 <= h < 24:
                period = "evening"
            else:
                period = "night"
            return day, period
        except Exception:
            return None, None

    def _build_pointwise_prompt_subcategory(self, history, candidate, label):
        """
        Variant of _build_pointwise_prompt that uses [category/subcategory] format.
        """
        def _fmt(item):
            cat = item.get('category', 'General')
            subcat = item.get('subcategory', '')
            return f"{cat}/{subcat}" if subcat else cat

        prompt = "A user read these news articles:\n"
        if history:
            for i, h in enumerate(history, 1):
                prompt += f"{i}. [{_fmt(h)}] {h['text']}\n"
        else:
            prompt += "(No reading history)\n"

        prompt += "\n"
        prompt += "Candidate article:\n"
        prompt += f"[{_fmt(candidate)}] {candidate['text']}\n"
        prompt += "\n"
        prompt += "Will this user read this article? Answer:"

        target = " Yes" if label == 1 else " No"
        return prompt, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Build prompt
        if self.use_subcategory:
            prompt, target = self._build_pointwise_prompt_subcategory(
                sample['history'], sample['candidate'], sample['label']
            )
        else:
            prompt, target = self._build_pointwise_prompt(
                sample['history'], sample['candidate'], sample['label'],
                impression_timestamp=sample.get('impression_timestamp', ''),
            )

        if self.use_chat_template:
            # Use chat template for instruct models with system prompt
            system_prompt = (
                "You are a news recommendation assistant. "
                "Based on a user's reading history, predict whether they will read a given article. "
                "Each article includes its category and title. "
                "Answer with Yes or No."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            # Apply chat template to get the formatted prompt with generation prompt
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Full text includes the assistant response
            full_text = formatted_prompt + target

            # Tokenize full text
            input_ids = self.tokenizer.encode(
                full_text,
                max_length=self.max_len,
                truncation=True,
                add_special_tokens=False  # Chat template already added them
            )

            # Get prompt length for masking
            prompt_ids = self.tokenizer.encode(
                formatted_prompt,
                max_length=self.max_len,
                truncation=True,
                add_special_tokens=False
            )

            # Mask prompt, only train on target
            train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        else:
            # Original raw text format
            full_text = prompt + target
            input_ids = self.tokenizer.encode(
                full_text,
                max_length=self.max_len,
                truncation=True,
                add_special_tokens=True
            )

            # Create training labels (mask prompt, only train on Yes/No)
            prompt_ids = self.tokenizer.encode(
                prompt,
                max_length=self.max_len,
                truncation=True,
                add_special_tokens=True
            )

            train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        input_ids = input_ids[:self.max_len]
        train_labels = train_labels[:self.max_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(train_labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
        }


class DOCAPointwiseSFTDataset:
    """
    DOCA feed dataset for point-wise SFT training.

    Each sample is a (user_context, single_candidate, is_clicked) tuple.
    User context includes: interests (with sources/intent/rationale),
    negative_interests, conversation (with inline curation), shown_10d.
    Model learns to predict Yes/No for click likelihood.

    Input: JSONL file produced by src/prepare_doca.py
    """

    SYSTEM_PROMPT = (
        "You are a content recommendation assistant. "
        "Based on a user's interest profile, conversation history, and previously shown articles, "
        "predict whether they will click on a given article.\n\n"
        "Output format: Answer ONLY \"Yes\" or \"No\". Do not explain.\n\n"
        "Ranking guidelines (highest to lowest priority):\n"
        "1. Source signal priority: Inline Curation (user explicitly selected, strongest signal) "
        "> User Interaction (clicks/likes) > Chat History (inferred from messages).\n"
        "2. Interest strength: High (0.9-1.0) > Medium (0.8-0.9) > Exploratory (<0.8).\n"
        "3. Long-term interest relevance: How well does the article align with established interests?\n"
        "4. Short-term task relevance: How relevant is it to the user's recent activities and needs?\n"
        "5. Freshness: Prefer up-to-date content; consider if information might be outdated.\n"
        "6. Importance: How significant is this content for the user?\n"
        "7. Novelty: Prefer content the user hasn't seen recently (check shown articles).\n\n"
        "Also consider:\n"
        "- Articles matching disliked interests should NOT be clicked.\n"
        "- [CURATED] messages in conversations indicate the strongest user intent.\n"
        "- Interest rationale explains WHY something is an interest — use it to judge relevance."
    )

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_len: int = 4096,
        sample: int = -1,
        seed: int = 42,
        neg_ratio: float = 1.0,
        max_interests: int = 0,
        max_conversation_msgs: int = 15,
        max_shown: int = 10,
        use_chat_template: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neg_ratio = neg_ratio
        self.max_interests = max_interests
        self.max_conversation_msgs = max_conversation_msgs
        self.max_shown = max_shown
        self.use_chat_template = use_chat_template
        self.seed = seed

        self.samples = self._load_and_expand(jsonl_path, sample)

        pos_count = sum(1 for s in self.samples if s['label'] == 1)
        neg_count = len(self.samples) - pos_count
        print(f"DOCAPointwiseSFTDataset: {len(self.samples)} samples "
              f"(pos={pos_count}, neg={neg_count}, ratio={neg_count / max(pos_count, 1):.2f})")

    def _load_and_expand(self, jsonl_path: str, sample_limit: int):
        """Load JSONL feeds and expand into pointwise samples."""
        rng = random.Random(self.seed)
        all_samples = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                feed = json.loads(line)

                user_context = {
                    'interests': feed.get('interests', [])[:self.max_interests] if self.max_interests > 0 else feed.get('interests', []),
                    'negative_interests': feed.get('negative_interests', []),
                    'conversation': feed.get('conversation', [])[:self.max_conversation_msgs],
                    'shown_10d': feed.get('shown_10d', [])[:self.max_shown],
                }

                candidates = feed.get('candidates', [])
                positives = [c for c in candidates if c.get('is_clicked')]
                negatives = [c for c in candidates if not c.get('is_clicked')]

                # Add all positive samples
                for pos in positives:
                    all_samples.append({
                        'user_context': user_context,
                        'candidate': pos,
                        'label': 1,
                    })

                # Sample negatives per impression
                if positives:
                    num_neg = int(len(positives) * self.neg_ratio)
                else:
                    # For feeds with no clicks, sample 1 negative as pure negative example
                    num_neg = 1

                if negatives and num_neg > 0:
                    sampled_negs = rng.sample(negatives, min(num_neg, len(negatives)))
                    for neg in sampled_negs:
                        all_samples.append({
                            'user_context': user_context,
                            'candidate': neg,
                            'label': 0,
                        })

        rng.shuffle(all_samples)

        if sample_limit > 0 and sample_limit < len(all_samples):
            all_samples = all_samples[:sample_limit]

        return all_samples

    def _build_prompt(self, user_context, candidate, label):
        """
        Build pointwise prompt with all user signals.
        Mirrors the evidence structure from the DOCA ranking liquid template.
        """
        parts = []

        # 1. User interests (with sources, intent, classification, status, rationale)
        interests = user_context.get('interests', [])
        if interests:
            parts.append("User interests:")
            for i, intr in enumerate(interests, 1):
                name = intr.get('name', '')
                strength = intr.get('strength', 0)
                domain = intr.get('domain', '')
                sources = ', '.join(intr.get('sources', []))
                intent = intr.get('intent', '')
                classification = intr.get('classification', '')
                status = intr.get('status', '')
                keywords = ', '.join(intr.get('keywords', [])[:5])
                # Core line: name, strength, domain, source signal
                line = f"{i}. {name} (strength={strength:.2f}"
                if domain:
                    line += f", {domain}"
                if sources:
                    line += f", source: {sources}"
                line += ")"
                # Additional metadata
                meta = []
                if classification:
                    meta.append(classification)
                if intent:
                    meta.append(f"intent: {intent}")
                if status:
                    meta.append(status)
                if meta:
                    line += f" [{', '.join(meta)}]"
                line += f" - {keywords}"
                parts.append(line)
                # Rationale (key evidence for why this is an interest)
                rationale = intr.get('rationale', '')
                if rationale:
                    if len(rationale) > 200:
                        rationale = rationale[:200] + "..."
                    parts.append(f"   Reason: {rationale}")

        # 2. Negative interests (with sources and rationale)
        neg_interests = user_context.get('negative_interests', [])
        if neg_interests:
            parts.append("\nDislikes:")
            for i, intr in enumerate(neg_interests, 1):
                name = intr.get('name', '')
                keywords = ', '.join(intr.get('keywords', [])[:5])
                sources = ', '.join(intr.get('sources', []))
                line = f"{i}. {name}"
                if sources:
                    line += f" (source: {sources})"
                line += f" - {keywords}"
                parts.append(line)
                rationale = intr.get('rationale', '')
                if rationale:
                    if len(rationale) > 200:
                        rationale = rationale[:200] + "..."
                    parts.append(f"   Reason: {rationale}")

        # 3. Conversation history (human messages, inline curation marked)
        conversation = user_context.get('conversation', [])
        if conversation:
            parts.append("\nRecent conversations:")
            for msg in conversation:
                text = msg.get('text', '').strip()
                if text:
                    if len(text) > 150:
                        text = text[:150] + "..."
                    if msg.get('is_inline_curation'):
                        parts.append(f'- [CURATED] "{text}"')
                    else:
                        parts.append(f'- "{text}"')

        # 4. Shown 10d
        shown = user_context.get('shown_10d', [])
        if shown:
            parts.append("\nRecently shown articles:")
            for item in shown:
                if isinstance(item, dict):
                    title = item.get('title', '')
                    date = item.get('event_time', '')[:10]
                    parts.append(f'- "{title}" ({date})')
                else:
                    parts.append(f'- "{item}"')

        # 5. Candidate
        parts.append("\nCandidate article:")
        parts.append(f"Title: {candidate.get('title', '')}")
        summary = candidate.get('summary', '')
        if summary:
            parts.append(f"Summary: {summary}")

        parts.append("\nWill this user click on this article? Answer:")

        prompt = '\n'.join(parts)
        target = " Yes" if label == 1 else " No"
        return prompt, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt, target = self._build_prompt(
            sample['user_context'], sample['candidate'], sample['label']
        )

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full_text = formatted_prompt + target

            input_ids = self.tokenizer.encode(
                full_text, max_length=self.max_len,
                truncation=True, add_special_tokens=False
            )
            prompt_ids = self.tokenizer.encode(
                formatted_prompt, max_length=self.max_len,
                truncation=True, add_special_tokens=False
            )
            train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        else:
            full_text = prompt + target
            input_ids = self.tokenizer.encode(
                full_text, max_length=self.max_len,
                truncation=True, add_special_tokens=True
            )
            prompt_ids = self.tokenizer.encode(
                prompt, max_length=self.max_len,
                truncation=True, add_special_tokens=True
            )
            train_labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

        input_ids = input_ids[:self.max_len]
        train_labels = train_labels[:self.max_len]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(train_labels, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
        }


class InstructionJSONLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=4096, sample=-1, seed=0):
        random.seed(seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if sample > 0:
            self.data = random.sample(self.data, min(sample, len(self.data)))
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        instruction = data_point.get("instruction", "")
        input_text = data_point.get("input", "")
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:\n"""
        return f"""### Instruction:
{instruction}

### Response:\n"""

    def prepare_sample(self, idx):
        data_point = self.data[idx]
        output_text = data_point.get("output", "")
        prompt = self.generate_prompt(data_point)

        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        attention_mask = [1] * len(tokens)

        golden_tokens = self.tokenizer.encode(output_text, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class ItemTokenSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        item_emb_path,
        item_token="<item>",
        max_len=4096,
        sample=-1,
        seed=0,
        category="",
        inject_target=False,
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.category = category
        self.item_token = item_token
        self.inject_target = inject_target
        self.item_embs = np.load(item_emb_path)
        self.item_token_id = tokenizer.convert_tokens_to_ids(item_token)
        if self.item_token_id == tokenizer.unk_token_id:
            raise ValueError(f"Item token {item_token} is not in tokenizer vocab.")

        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def _history_placeholders(self, history_ids):
        placeholders = []
        for _ in history_ids:
            placeholders.append(self.item_token)
        return ", ".join(placeholders)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n"""

    def get_history(self, row):
        history_ids = eval(row["history_item_id"])
        history_str = self._history_placeholders(history_ids)
        target_item_id = int(row["item_id"])
        return {
            "input": f"The user has played the following {self.category}s before: {history_str}",
            "history_ids": history_ids,
            "target_item_id": target_item_id,
        }

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        history_ids = history["history_ids"]
        target_item_id = history["target_item_id"]

        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        input_prompt_len = len(tokens)

        golden_tokens = self.tokenizer.encode(self.item_token, bos=False, eos=True)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)

        labels = [-100] * input_prompt_len + golden_tokens

        item_positions = [i for i, t in enumerate(tokens) if t == self.item_token_id]
        expected_items = len(history_ids) + 1
        if len(item_positions) != expected_items:
            raise ValueError(
                f"Expected {expected_items} item tokens, found {len(item_positions)} for row {idx}."
            )

        item_ids = list(history_ids) + [target_item_id]
        item_embeds = [self.item_embs[item_id] for item_id in item_ids]

        target_pos = item_positions[-1]
        target_embed = self.item_embs[target_item_id]

        inject_positions = item_positions[:-1]
        inject_embeds = item_embeds[:-1]
        if self.inject_target:
            inject_positions = item_positions
            inject_embeds = item_embeds

        start = max(0, len(tokens) - self.max_len)
        tokens = tokens[start:]
        attention_mask = attention_mask[start:]
        labels = labels[start:]

        def _shift_positions(positions, embeds):
            shifted_positions = []
            shifted_embeds = []
            for pos, emb in zip(positions, embeds):
                if pos >= start:
                    shifted_positions.append(pos - start)
                    shifted_embeds.append(emb)
            return shifted_positions, shifted_embeds

        inject_positions, inject_embeds = _shift_positions(inject_positions, inject_embeds)
        target_positions, target_embeds = _shift_positions([target_pos], [target_embed])

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "inject_positions": inject_positions,
            "inject_embeds": inject_embeds,
            "target_positions": target_positions,
            "target_embeds": target_embeds,
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class TextMetaSFTDataset(Dataset):
    def __init__(
        self,
        train_file,
        tokenizer,
        item_meta_path,
        max_len=4096,
        sample=-1,
        seed=0,
        category="",
    ):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.category = category

        with open(item_meta_path, "r") as f:
            self.item_meta = json.load(f)

        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def _format_item(self, item_id):
        meta = self.item_meta.get(str(item_id), {})
        title = meta.get("title", "")
        brand = meta.get("brand", "")
        description = meta.get("description", "")
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if brand:
            parts.append(f"Brand: {brand}")
        if description:
            parts.append(f"Description: {description}")
        return " | ".join(parts)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_ids = eval(row["history_item_id"])
        history = []
        for item_id in history_ids:
            history.append(self._format_item(item_id))
        history_str = ",\t".join([f"\"{h}\"" for h in history])
        target_item = str(row["item_title"])
        target_item = f"\"{target_item}\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = history_ids[-1]
        return {
            "input": f"The user has played the following {self.category}s before: {history_str}",
            "output": target_item,
            "dedup": target_item_id == last_history_item_id,
        }

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history["output"]
        history["output"] = ""

        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)

        attention_mask = [1] * len(tokens)
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class MINDTextSFTDataset(Dataset):
    def __init__(
        self,
        behaviors_path,
        news_path,
        tokenizer,
        max_len=4096,
        sample=-1,
        seed=0,
        max_history=50,
        use_abstract=False,
    ):
        random.seed(seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.max_history = max_history
        self.use_abstract = use_abstract

        self.news = {}
        with open(news_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                news_id = parts[0]
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""
                if use_abstract and abstract:
                    self.news[news_id] = f"{title} {abstract}"
                else:
                    self.news[news_id] = title

        self.data = []
        with open(behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                history = parts[3].split()
                impressions = parts[4].split()
                positives = []
                for imp in impressions:
                    if "-" not in imp:
                        continue
                    nid, label = imp.rsplit("-", 1)
                    if label == "1":
                        positives.append(nid)
                if not positives:
                    continue
                history = history[-self.max_history :]
                for pos in positives:
                    self.data.append(
                        {
                            "history": history,
                            "target": pos,
                        }
                    )

        if sample > 0:
            self.data = random.sample(self.data, min(sample, len(self.data)))
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def _history_titles(self, history_ids):
        titles = [self.news.get(nid, "") for nid in history_ids]
        titles = [t for t in titles if t]
        return ", ".join([f"\"{t}\"" for t in titles])

    def generate_prompt(self, history_text):
        return f"""### User Input: 
The user has read the following news before: {history_text}

### Response:\n"""

    def prepare_sample(self, idx):
        row = self.data[idx]
        history_text = self._history_titles(row["history"])
        target_title = self.news.get(row["target"], "")
        target_title = f"\"{target_title}\"\n"

        prompt = self.generate_prompt(history_text)
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        attention_mask = [1] * len(tokens)

        golden_tokens = self.tokenizer.encode(target_title, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class EvalMINDTextDataset(Dataset):
    def __init__(
        self,
        behaviors_path,
        news_path,
        tokenizer,
        max_len=4096,
        sample=-1,
        seed=0,
        max_history=50,
        use_abstract=False,
        test=False,
    ):
        random.seed(seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.max_history = max_history
        self.use_abstract = use_abstract
        self.test = test

        self.news = {}
        with open(news_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue
                news_id = parts[0]
                title = parts[3]
                abstract = parts[4] if len(parts) > 4 else ""
                if use_abstract and abstract:
                    self.news[news_id] = f"{title} {abstract}"
                else:
                    self.news[news_id] = title

        self.data = []
        with open(behaviors_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                history = parts[3].split()
                impressions = parts[4].split()
                positives = []
                for imp in impressions:
                    if "-" not in imp:
                        continue
                    nid, label = imp.rsplit("-", 1)
                    if label == "1":
                        positives.append(nid)
                if not positives:
                    continue
                history = history[-self.max_history :]
                for pos in positives:
                    self.data.append(
                        {
                            "history": history,
                            "target": pos,
                        }
                    )

        if sample > 0:
            self.data = random.sample(self.data, min(sample, len(self.data)))
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def _history_titles(self, history_ids):
        titles = [self.news.get(nid, "") for nid in history_ids]
        titles = [t for t in titles if t]
        return ", ".join([f"\"{t}\"" for t in titles])

    def generate_prompt(self, history_text):
        return f"""### User Input: 
The user has read the following news before: {history_text}

### Response:\n"""

    def get_history(self, row):
        history_text = self._history_titles(row["history"])
        target_title = self.news.get(row["target"], "")
        target_title = f"\"{target_title}\""
        return {
            "input": f"The user has read the following news before: {history_text}",
            "output": target_title + "\n",
        }

    def prepare_sample(self, idx):
        row = self.data[idx]
        history_text = self._history_titles(row["history"])
        prompt = self.generate_prompt(history_text)
        tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        target_title = self.news.get(row["target"], "")
        target_title = f"\"{target_title}\"\n"
        golden_tokens = self.tokenizer.encode(target_title, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data[i]))
        return temp

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class D3Dataset(Dataset):
    def __init__(self, train_file, max_len=4096, sample=-1, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title'])
        history = ""
        history_str = "::".join(row["history_item_title"])
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\"\n"
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item,
                "history_str": history_str,
                "dedup": target_item_id == last_history_item_id}

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n 
"""
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''

        prompt = self.generate_prompt(history)
        self.prompt2history[instruction + prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item

        return {
            "prompt": instruction + prompt,
            "completion": target_item,
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class EvalD3Dataset(Dataset):

    def __init__(self, train_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title'])
        history = ""
        for i in range(L):
            if i == 0:
                history += "\"" + row['history_item_title'][i] + "\""
            else:
                history += ",\t\"" + row['history_item_title'][i] + "\""
        target_item = str(row['item_title'])
        target_item = "\"" + target_item + "\""
        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""

        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,

                # "select_index": select_index,
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            pass

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp


class EvalTextMetaDataset(Dataset):
    def __init__(self, train_file, tokenizer, item_meta_path, max_len=4096, sample=-1, test=False, seed=0, category=""):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category

        with open(item_meta_path, "r") as f:
            self.item_meta = json.load(f)

        self.instructs = [
            f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
            f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
            f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
            f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
            f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
            f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
            f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
            f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
            f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
            f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
            f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def _format_item(self, item_id):
        meta = self.item_meta.get(str(item_id), {})
        title = meta.get("title", "")
        brand = meta.get("brand", "")
        description = meta.get("description", "")
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if brand:
            parts.append(f"Brand: {brand}")
        if description:
            parts.append(f"Description: {description}")
        return " | ".join(parts)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_ids = eval(row["history_item_id"])
        history = []
        for item_id in history_ids:
            history.append(self._format_item(item_id))
        history_str = ",\t".join([f"\"{h}\"" for h in history])
        target_item = str(row["item_title"])
        target_item = f"\"{target_item}\""
        target_item_id = row["item_id"]
        last_history_item_id = history_ids[-1]
        return {
            "input": f"The user has palyed the following {self.category}s before: {history_str}",
            "output": target_item + "\n",
            "dedup": target_item_id == last_history_item_id,
        }

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[random.randint(0, len(self.instructs)-1)]}\n
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        history["output"] = ""

        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)

        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        target_item = self.get_history(self.data.iloc[idx])["output"]
        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class SidDataset(Dataset):
    def __init__(self, train_file, max_len=4096, sample=-1, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        L = len(row['history_item_sid'])
        history = ""
        history_str = "::".join(history_item_sid)
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                # Analyze user preferences and then predict the semantic ID of the next item.
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}

    def prepare_sample(self, idx):
        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''

        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item

        return {
            "prompt": prompt,
            "completion": target_item,

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class SidSFTDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # Lazy loading: removed self.get_inputs() - now processes on-demand in __getitem__

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        L = len(row['history_item_sid'])
        history = ""
        history_str = ", ".join(history_item_sid)
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        # print("**********************")
        # print("history: ", history)
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)
        # print("prompt: ", prompt)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        # print("tokens: ", tokens)
        # print("**********************")
        history["input"] = ""

        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            import warnings; warnings.warn(f"Token length {len(tokens)} exceeds max_len {self.max_len}", stacklevel=2)

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    # Deprecated: Pre-loading all samples uses too much memory
    # Use lazy loading via __getitem__ instead for DataLoader compatibility
    # def get_inputs(self):
    #     inputs = []
    #     for i in tqdm(range(len(self.data))):
    #         inputs.append(self.prepare_sample(i))
    #     self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def __getitem__(self, idx):
        # Lazy loading: prepare sample on-demand (enables DataLoader multi-worker + DDP)
        return self.prepare_sample(idx)


class SidSFTDataset_GPR(Dataset):
    def __init__(self, train_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        # Try to load features from standard location
        try:
            with open(f'data/{category}/{category}.user.json', 'r') as f:
                self.user_features = json.load(f)
        except FileNotFoundError:
            try:
                dataset_dir = os.path.dirname(train_file)
                # Assuming structure data/Amazon/train/Sports... -> data/Sports/Sports.user.json
                with open(f'data/{category}/{category}.user.json', 'r') as f:
                    self.user_features = json.load(f)
            except:
                self.user_features = {}

        try:
            with open(f'data/{category}/{category}.item.json', 'r') as f:
                self.item_features = json.load(f)
        except FileNotFoundError:
            self.item_features = {}

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        L = len(row['history_item_sid'])
        history = ""
        history_str = ", ".join(history_item_sid)
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input": f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                "output": target_item + "\n",
                "history_str": history_str,
                "dedup": target_item_sid == last_history_item_sid}

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        row = self.data.iloc[idx]

        # Heterogeneous Prompt Construction
        user_id = str(row.get('user_id_original_str', ''))
        u_token = self.user_features.get(user_id, '[USER_UNKNOWN]')
        e_token = row.get('e_token', '[CTX_HOMEPAGE]')

        try:
            history_item_ids = eval(str(row['history_item_id']))
            history_sids = eval(str(row['history_item_sid']))
        except:
            history_item_ids = []
            history_sids = []

        history_str = ""
        for i, item_id in enumerate(history_item_ids):
            item_type = self.item_features.get(
                str(item_id), {}).get('item_type', 'O')
            token_prefix = '[O_TOKEN]' if item_type == 'O' else '[I_TOKEN]'

            if i > 0:
                history_str += ", "
            sid = history_sids[i] if i < len(history_sids) else ""
            history_str += f"{token_prefix}{sid}"

        target_item_id = str(row['item_id'])
        target_sid = str(row['item_sid'])
        target_type = self.item_features.get(
            target_item_id, {}).get('item_type', 'O')
        target_prefix = '[O_TOKEN]' if target_type == 'O' else '[I_TOKEN]'

        target_item_with_prefix = f"{target_prefix}{target_sid}\n"

        # Prompt
        input_text = f"{u_token} {e_token} The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"

        history = {
            "input": input_text,
            "output": target_item_with_prefix
        }

        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""

        attention_mask = [1] * len(tokens)

        # Final Value for VAFT
        final_value = self.item_features.get(
            target_item_id, {}).get('final_value', 0.0)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
                "final_value": final_value
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            import warnings; warnings.warn(f"Token length {len(tokens)} exceeds max_len {self.max_len}", stacklevel=2)

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
            "final_value": final_value
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class EvalSidDataset(Dataset):

    def __init__(self, train_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response:\n{data_point["output"]}
"""

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        L = len(row['history_item_sid'])
        history = ""
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]
        target_item = str(row['item_sid'])
        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None
        return {"input":  # f"The user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?",
                f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}",
                "output": target_item + '\n',
                "dedup": target_item_sid == last_history_item_sid}

    def prepare_sample(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)

        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""

        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,

            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            import warnings; warnings.warn(f"Token length {len(tokens)} exceeds max_len {self.max_len}", stacklevel=2)

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))

        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class SidItemFeatDataset(Dataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=4096, sample=-1, test=False, seed=0, category=""):
        """
        Dataset for sid2title and title2sid tasks.

        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
        """
        random.seed(seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(
            tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.category = category

        # Build sid2title and title2sid mappings
        self.sid2title = {}
        self.title2sid = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid

        # Create data samples
        self.data = []

        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })

        # Create title2sid samples
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        if self.tokenizer is not None:
            pass  # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # sid2title
            prompt = f'What is the title of item "{data_point["input"]}"?'
            response = data_point['output']

        return f"""### User Input: 
{prompt}

### Response:\n"""

    def prepare_sample(self, idx):
        if self.tokenizer is None:
            return self.data[idx]

        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        data_point = self.data[idx]

        prompt = self.generate_prompt(data_point)
        # print("sidfeature prompt: ", prompt)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        target = data_point['output'] + '\n'

        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(
                f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else [self.prepare_sample(i) for i in range(len(self))]

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        return self.prepare_sample(idx)


class RLTitle2SidDataset(Dataset):
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for title2sid and description2sid tasks.
        Returns prompt-completion pairs for RL training.

        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            max_len: Maximum sequence length (not used for RL format)
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}

        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.title2sid = {}
        self.sid2description = {}
        self.description2sid = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']

                # Handle description format
                if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
                    try:
                        desc_list = eval(description)
                        description = desc_list[0] if desc_list else description
                    except:
                        pass

                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.title2sid[title] = combined_sid
                    self.sid2description[combined_sid] = description
                    self.description2sid[description] = combined_sid

        # Create data samples
        self.data = []

        # Create title2sid samples
        for title, sid in self.title2sid.items():
            self.data.append({
                'task': 'title2sid',
                'input': title,
                'output': sid
            })

        # Create description2sid samples
        for description, sid in self.description2sid.items():
            self.data.append({
                'task': 'description2sid',
                'input': description,
                'output': sid
            })

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        if data_point['task'] == 'title2sid':
            prompt = f"Which item has the title: {data_point['input']}?"
            response = data_point['output']
        else:  # description2sid
            prompt = f"An item can be described as follows: \"{data_point['input']}\". Which item is it describing?"
            response = data_point['output']

        return f"""### User Input: 
{prompt}

### Response:\n"""

    def prepare_sample(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"

        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item

        return {
            "prompt": prompt,
            "completion": target_item,

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.data[i])
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class RLSeqTitle2SidDataset(Dataset):
    def __init__(self, train_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sequential recommendation using title sequences.
        Uses user interaction history with item titles to recommend next item's semantic ID.

        Args:
            train_file: Path to CSV file with sequence data (must have history_item_title and item_sid columns)
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, inter_titles):
        return f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"

    def get_history(self, row):
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])

        # Format title sequence for prompt
        inter_titles = ", ".join(
            [f'"{title}"' for title in history_item_title])

        target_sid = row['item_sid']

        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                target_item_id = row.get('item_id', None)
                last_history_item_id = history_item_id[-1] if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False

        return {
            "inter_titles": inter_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate,
            "history_str": "::".join(history_item_title)
        }

    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def prepare_sample(self, idx):
        history_data = self.get_history(self.data.iloc[idx])

        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None

        # Generate prompt using title sequence
        prompt = self.generate_prompt(history_data['inter_titles'])
        target = history_data['target_sid'] + '\n'

        formatted_prompt = self.generate_formatted_prompt(prompt, "")

        self.prompt2history[formatted_prompt] = history_data['history_str']
        self.history2target[history_data['history_str']] = target

        return {
            "prompt": formatted_prompt,
            "completion": target,

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        result = self.prepare_sample(idx)
        return result if result is not None else {"prompt": "", "completion": ""}


class RLSid2TitleDataset(Dataset):
    def __init__(self, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sid2title tasks.
        Returns prompt-completion pairs for RL training where input is semantic ID and output is item title.

        Args:
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices  
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}

        # Build sid2title mapping
        self.sid2title = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title

        # Create data samples
        self.data = []

        # Create sid2title samples
        for sid, title in self.sid2title.items():
            self.data.append({
                'task': 'sid2title',
                'input': sid,
                'output': title
            })

        if sample > 0 and sample < len(self.data):
            self.data = random.sample(self.data, sample)

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        prompt = f'What is the title of item "{data_point["input"]}"?'
        response = data_point['output']

        return f"""### User Input: 
{prompt}

### Response:\n"""

    def prepare_sample(self, idx):
        data_point = self.data[idx]
        prompt = self.generate_prompt(data_point)
        target_item = data_point['output'] + "\n"

        self.prompt2history[prompt] = data_point['input']
        self.history2target[data_point['input']] = target_item

        return {
            "prompt": prompt,
            "completion": target_item,

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            inputs.append(self.prepare_sample(i))
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.data[i])
        return temp

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.prepare_sample(idx)  # Lazy loading


class RLSidhis2TitleDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, sample=-1, seed=0, category="", dedup=False):
        """
        RL-specific dataset for sequential recommendation using semantic IDs in history and outputting item titles.
        Uses user interaction history with item semantic IDs to recommend next item's title.

        Args:
            train_file: Path to CSV file with sequence data (must have history_item_sid and item_id columns)
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            sample: Number of samples to use (-1 for all)
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.category = category
        self.dedup = dedup
        self.prompt2history = {}
        self.history2target = {}

        # Build item_id to title mapping
        self.id2title = {}
        for item_id, features in self.item_feat.items():
            self.id2title[item_id] = features['title']

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        L = len(row['history_item_sid'])
        history = ""
        history_str = "::".join(history_item_sid)
        for i in range(L):
            if i == 0:
                history += row['history_item_sid'][i]
            else:
                history += ", " + row['history_item_sid'][i]

        # Get target item title from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2title:
            target_title = self.id2title[target_item_id]
        else:
            target_title = f"Unknown item {target_item_id}"

        target_item_sid = row["item_sid"]
        last_history_item_sid = row['history_item_sid'][-1] if row['history_item_sid'] else None

        return {
            "input": f"The user has interacted with items {history} in chronological order. Can you predict the title of the next item that the user may expect? Analyze user preferences and then predict the title of the next item.",
            "output": target_title + "\n",
            "history_str": history_str,
            "dedup": target_item_sid == last_history_item_sid
        }

    def prepare_sample(self, idx):
        history = self.get_history(self.data.iloc[idx])

        # Skip if duplicate and dedup is enabled
        if self.dedup and history['dedup']:
            return None

        target_item = history['output']
        history['output'] = ''

        prompt = self.generate_prompt(history)
        self.prompt2history[prompt] = history["history_str"]
        self.history2target[history["history_str"]] = target_item

        return {
            "prompt": prompt,
            "completion": target_item,

        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        result = self.prepare_sample(idx)
        return result if result is not None else {"prompt": "", "completion": ""}


class FusionSeqRecDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        Fusion dataset combining sequence recommendation with item features.
        Uses semantic IDs for user history, outputs item titles or descriptions.

        Args:
            train_file: Path to CSV file with sequence data
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        # Build sid2title and sid2description mappings
        self.sid2title = {}
        self.sid2description = {}

        for item_id, sids in self.indices.items():
            if item_id in self.item_feat:
                title = self.item_feat[item_id]['title']
                description = self.item_feat[item_id]['description']

                # Process description according to requirements:
                # 1. If description is empty, use title
                # 2. If description is a list, select the longest one
                # 3. If the longest in list is also empty, use title
                processed_description = self._process_description(
                    description, title)

                # Concatenate all three semantic IDs as the key
                if len(sids) >= 3:
                    combined_sid = sids[0] + sids[1] + sids[2]
                    self.sid2title[combined_sid] = title
                    self.sid2description[combined_sid] = processed_description
        # print("self.sid2title: ", self.sid2title)
        # print("self.sid2description: ", self.sid2description)
        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def _process_description(self, description, title):
        """
        Process description according to the requirements:
        1. If description is empty, use title
        2. If description is a list, select the longest one
        3. If the longest in list is also empty, use title

        Args:
            description: The description field from item_feat
            title: The title field from item_feat

        Returns:
            str: Processed description
        """
        # Check if description is empty or None
        if not description or description == '':
            return title

        # Check if description is a list (either actual list or string representation)
        if isinstance(description, list):
            # It's already a list
            desc_list = description
        elif isinstance(description, str) and description.startswith('[') and description.endswith(']'):
            try:
                # Try to parse string representation of list
                desc_list = eval(description)
            except:
                # If parsing fails, treat as regular string
                return description if description.strip() else title
        else:
            # Regular string description
            return description if description.strip() else title

        # If we have a list, find the longest non-empty item
        if desc_list:
            # Filter out empty strings and find the longest
            non_empty_descriptions = [
                desc for desc in desc_list if desc and desc.strip()]
            if non_empty_descriptions:
                # Return the longest description
                longest_desc = max(non_empty_descriptions, key=len)
                return longest_desc
            else:
                # All descriptions in list are empty, use title
                return title
        else:
            # Empty list, use title
            return title

    def __len__(self):
        return len(self.data)

    def generate_prompt_title(self, history):
        return f"The user has sequentially interacted with items {history}. Can you recommend the next item for him? Tell me the title of the item"

    def generate_prompt_description(self, history):
        return f"Please review the user's historical interactions: {history}, and describe what kind of item he still needs."

    def get_history(self, row):
        history_item_sid = eval(row['history_item_sid'])
        history_str = ", ".join(history_item_sid)

        target_sid = row['item_sid']

        # Use the new sid2title and sid2description mappings
        if target_sid in self.sid2title:
            target_title = self.sid2title[target_sid]
        else:
            target_title = target_sid

        if target_sid in self.sid2description:
            target_description = self.sid2description[target_sid]
            # Clean description if it's a string representation of a list
            if isinstance(target_description, str) and target_description.startswith("['") and target_description.endswith("']"):
                try:
                    desc_list = eval(target_description)
                    target_description = desc_list[0] if desc_list else target_description
                except:
                    pass  # Keep original if eval fails
        else:
            target_description = f"An item with semantic ID {target_sid}"

        # Check for deduplication
        last_history_sid = history_item_sid[-1] if history_item_sid else None
        is_duplicate = target_sid == last_history_sid

        return {
            "history_str": history_str,
            "target_title": target_title,
            "target_description": target_description,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }

    def generate_formatted_prompt(self, prompt, response):
        return f"""### User Input: 
{prompt}

### Response:\n"""

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you recommend the next item for the user based on their interaction history?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history_data = self.get_history(self.data.iloc[idx])

        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None

        # Randomly choose between title and description tasks
        """if random.random() < 0.5:
            # Title task
            prompt = self.generate_prompt_title(history_data['history_str'])
            target = history_data['target_title'] + '\n'
        else:
            # Description task
            prompt = self.generate_prompt_description(history_data['history_str'])
            target = history_data['target_description'] + '\n'
        """
        prompt = self.generate_prompt_title(history_data['history_str'])
        target = history_data['target_title'] + '\n'
        # print("fusion prompt: ", prompt)

        formatted_prompt = self.generate_formatted_prompt(prompt, "")
        tokens = tokens + \
            self.tokenizer.encode(formatted_prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        golden_tokens = self.tokenizer.encode(target, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(
                f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        return self.prepare_sample(idx)


class TitleHistory2SidSFTDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses item titles in user history to predict next item's semantic ID.

        Args:
            train_file: Path to CSV file with sequence data (must have history_item_title and item_id columns)
            item_file: Path to .item.json file with item features
            index_file: Path to .index.json file with item indices
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load sequence data
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)

        # Load item features and indices
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        # Build item_id to semantic ID mapping
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3:
                combined_sid = sids[0] + sids[1] + sids[2]
                self.id2sid[item_id] = combined_sid

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def __len__(self):
        return len(self.data)

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def get_history(self, row):
        """Extract user history from title sequence and target semantic ID"""
        # Parse history_item_title field
        history_item_title = eval(row['history_item_title'])

        # Format title sequence for prompt
        history_titles = ", ".join(
            [f'"{title}"' for title in history_item_title])

        # Get target item's semantic ID from item_id
        target_item_id = str(row['item_id'])
        if target_item_id in self.id2sid:
            target_sid = self.id2sid[target_item_id]
        else:
            target_sid = target_item_id  # Fallback to item_id if no semantic ID found

        # Check for deduplication if needed
        is_duplicate = False
        if self.dedup and 'history_item_id' in row:
            try:
                history_item_id = eval(row['history_item_id'])
                last_history_item_id = str(
                    history_item_id[-1]) if history_item_id else None
                is_duplicate = target_item_id == last_history_item_id
            except:
                is_duplicate = False

        return {
            "input": f"The user has interacted with the following {self.category} items in chronological order: {history_titles}. Can you predict the next item the user may expect?",
            "output": target_sid + "\n",
            "history_titles": history_titles,
            "target_sid": target_sid,
            "dedup": is_duplicate
        }

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history_data = self.get_history(self.data.iloc[idx])

        # Skip if duplicate and dedup is enabled
        if self.dedup and history_data['dedup']:
            return None

        target_output = history_data['output']
        history_data['output'] = ''

        prompt = self.generate_prompt(history_data)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        golden_tokens = self.tokenizer.encode(
            target_output, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(
                f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from deduplication
                inputs.append(result)
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        result = self.prepare_sample(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}


class PreferenceSFTDataset(Dataset):
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses user interaction history and preferences from preference file.

        Args:
            user_preference_file: Path to JSON file with user preferences (format: {"user": "user_id", "user_preference": ["pref text", ...]})
            index_file: Path to .index.json file mapping item_id to semantic IDs
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))

        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []

        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')

                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + \
                    ([target_item] if target_item is not None else [])

                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })

        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        # Find users with preferences and prepare data
        self.matched_data = self._prepare_preference_data()

        if sample > 0 and sample < len(self.matched_data):
            self.matched_data = random.sample(self.matched_data, sample)

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def _prepare_preference_data(self):
        """Prepare data directly from training samples"""
        matched_data = []

        for sample in self.training_samples:
            interaction_history = sample['interaction_history']
            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue

            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target

            # Create data point from each training sample
            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }

            matched_data.append(row_dict)

        return matched_data

    def __len__(self):
        return len(self.matched_data)

    def _convert_to_semantic_ids(self, item_ids):
        """Convert item IDs to semantic ID format using index.json"""
        semantic_ids = []

        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)

        return semantic_ids

    def get_history_and_preference(self, row_data):
        """Extract and format user history and preference"""
        # Get input history item IDs (all but last item) and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)

        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)

        # Get user preference
        user_preference = row_data['user_preference']

        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(
            target_item_id)

        result = {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences and predict the next item?",
            "output": f"### Reasoning:\n{user_preference}\n### Response:\n{target_sid}",
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }
        # print(result)
        return result

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Analyze the user's interaction history, provide insights about their preferences, and predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        row_data = self.matched_data[idx]
        history_and_pref = self.get_history_and_preference(row_data)

        # Skip empty histories or missing targets
        if not history_and_pref['history_semantic_ids'] or not history_and_pref['target_sid']:
            return None

        target_output = history_and_pref['output']
        history_and_pref['output'] = ''

        prompt = self.generate_prompt(history_and_pref)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        golden_tokens = self.tokenizer.encode(
            target_output, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(
                f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.matched_data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from empty histories
                inputs.append(result)
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.matched_data)):
            temp.append(self.get_history_and_preference(self.matched_data[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        result = self.prepare_sample(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}


class UserPreference2sidSFTDataset(Dataset):
    def __init__(self, user_preference_file, index_file, tokenizer, max_len=4096, sample=-1, test=False, seed=0, category="", dedup=False):
        """
        SFT dataset that uses user interaction history with preferences to predict next item's semantic ID.
        Uses interaction history from preference file, predicts the last item in the sequence.

        Args:
            user_preference_file: Path to JSON file with user preferences
            index_file: Path to .index.json file mapping item_id to semantic IDs
            tokenizer: Tokenizer for encoding text
            max_len: Maximum sequence length
            sample: Number of samples to use (-1 for all)
            test: Whether this is test mode
            seed: Random seed
            category: Category name for prompts
            dedup: Whether to filter duplicate items
        """
        random.seed(seed)

        # Load user preferences - handle both JSON and JSONL formats
        with open(user_preference_file, 'r') as f:
            try:
                preference_data = json.load(f)
            except json.JSONDecodeError:
                # Try JSONL format (multiple JSON objects, one per line)
                f.seek(0)
                preference_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        preference_data.append(json.loads(line))

        # Handle new flat structure: each item is a separate training sample
        self.training_samples = []

        for item in preference_data:
            if item.get('split') == 'train':  # Only process train data
                user_id = item['user']
                preference_text = item.get('user_preference', '')
                context = item.get('context', {})
                history_items = context.get('history_items', [])
                target_item = context.get('target_item')

                # Create interaction history by combining history_items and target_item
                interaction_history = history_items + \
                    ([target_item] if target_item is not None else [])

                # Each item becomes a separate training sample
                self.training_samples.append({
                    'user_id': user_id,
                    'preference_text': preference_text,
                    'interaction_history': interaction_history
                })

        # Load index mapping
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup

        # Prepare training data from preference file interaction histories
        self.matched_data = self._prepare_sequence_data()

        if sample > 0 and sample < len(self.matched_data):
            self.matched_data = random.sample(self.matched_data, sample)

        # self.get_inputs()  # DISABLED: Using lazy loading instead

    def _prepare_sequence_data(self):
        """Prepare sequence prediction data from training samples"""
        matched_data = []

        for sample in self.training_samples:
            interaction_history = sample['interaction_history']

            # Skip samples without sufficient interaction history (need at least 2 items)
            if not interaction_history or len(interaction_history) < 2:
                continue

            # Use all items except the last one as input history
            # Use the last item as the target to predict
            input_history = interaction_history[:-1]  # All but last item
            target_item = interaction_history[-1]     # Last item as target

            row_dict = {
                'user_id': sample['user_id'],
                'user_preference': sample['preference_text'],
                'input_history': input_history,
                'target_item_id': target_item
            }

            matched_data.append(row_dict)

        return matched_data

    def __len__(self):
        return len(self.matched_data)

    def _convert_to_semantic_ids(self, item_ids):
        """Convert item IDs to semantic ID format using index.json"""
        semantic_ids = []

        for item_id in item_ids:
            item_id_str = str(item_id)
            if item_id_str in self.indices:
                sids = self.indices[item_id_str]
                if len(sids) >= 3:
                    # Combine the three semantic IDs
                    combined_sid = sids[0] + sids[1] + sids[2]
                    semantic_ids.append(combined_sid)
                else:
                    semantic_ids.append(item_id_str)
            else:
                semantic_ids.append(item_id_str)

        return semantic_ids

    def get_input_and_target(self, row_data):
        """Extract and format user history, preference, and target item"""
        # Get input history item IDs and convert to semantic IDs
        input_history_ids = row_data['input_history']
        history_semantic_ids = self._convert_to_semantic_ids(input_history_ids)

        # Format semantic IDs as a comma-separated string
        history_str = ", ".join(history_semantic_ids)

        # Get user preference
        user_preference = row_data['user_preference']

        # Get target item semantic ID
        target_item_id = row_data['target_item_id']
        target_semantic_ids = self._convert_to_semantic_ids([target_item_id])
        target_sid = target_semantic_ids[0] if target_semantic_ids else str(
            target_item_id)

        return {
            "input": f"The user has interacted with items {history_str} in chronological order. Can you analyze the user's preferences?\n### Reasoning:\n{user_preference}\nCan you predict the next possible item that the user may expect?",
            "output": target_sid,
            "history_semantic_ids": history_semantic_ids,
            "user_preference": user_preference,
            "target_sid": target_sid
        }

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response:\n{data_point["output"]}"""

    def prepare_sample(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Based on the user interaction history and preference analysis, predict the next item's semantic ID.

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        row_data = self.matched_data[idx]
        input_and_target = self.get_input_and_target(row_data)

        # Skip empty histories or missing targets
        if not input_and_target['history_semantic_ids'] or not input_and_target['target_sid']:
            return None

        target_output = input_and_target['output']
        input_and_target['output'] = ''

        prompt = self.generate_prompt(input_and_target)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        attention_mask = [1] * len(tokens)

        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,
            }

        golden_tokens = self.tokenizer.encode(
            target_output + '\n', bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(
                f"Sequence length {len(tokens)} exceeds max_len {self.max_len}")

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels[-self.max_len:],
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.matched_data))):
            result = self.prepare_sample(i)
            if result is not None:  # Skip None results from empty histories or missing targets
                inputs.append(result)
        self.inputs = inputs

    def get_all(self):
        temp = []
        for i in range(len(self.matched_data)):
            temp.append(self.get_input_and_target(self.matched_data[i]))
        return temp

    def get_inputs_list(self):
        return self.inputs if hasattr(self, 'inputs') else []

    def __getitem__(self, idx):
        if hasattr(self, 'inputs'):
            return self.prepare_sample(idx)  # Lazy loading
        result = self.prepare_sample(idx)
        return result if result is not None else {"input_ids": [], "attention_mask": [], "labels": []}
