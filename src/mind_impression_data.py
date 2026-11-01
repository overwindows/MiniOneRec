"""Strict offset-indexed MIND impressions without candidate or negative sampling."""

from array import array
import hashlib
from pathlib import Path

from data import MINDPointwiseSFTDataset
from full_impression_contract import read_news


def fingerprint(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def ids_digest(identifiers):
    digest = hashlib.sha256()
    for identifier in sorted(identifiers):
        digest.update(identifier.encode("utf-8") + b"\n")
    return digest.hexdigest()


class MINDImpressionSFTDataset(MINDPointwiseSFTDataset):
    def __init__(self, behaviors_path, news_path, tokenizer, *, max_len=2048, **prompt_options):
        if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 2:
            raise ValueError("Complete impressions require room for a prompt and answer")
        self._configure_encoder(
            tokenizer, max_len=max_len, return_meta=True, strict_template=True,
            strict_prompt_budget=True, **prompt_options)
        tokens = [tokenizer.encode(answer, add_special_tokens=False) for answer in (" Yes", " No")]
        if (any(len(ids) != 1 for ids in tokens) or tokens[0] == tokens[1]
                or any(ids[0] in tokenizer.all_special_ids for ids in tokens)):
            raise ValueError("Complete impressions require distinct single Yes/No tokens")
        self.behaviors_path = Path(behaviors_path).resolve(strict=True)
        self.news_path = Path(news_path).resolve(strict=True)
        self._behavior_fingerprint = fingerprint(self.behaviors_path)
        self._news_fingerprint = fingerprint(self.news_path)
        records = read_news(self.news_path)
        self.news = {
            identifier: {
                "text": (f"{record['title']} {record['abstract']}"
                         if self.use_abstract and record["abstract"] else record["title"]),
                "category": record["category"], "subcategory": record["subcategory"],
            }
            for identifier, record in records.items()
        }
        del records
        if not self.news:
            raise ValueError("Complete impressions require a nonempty news source")
        with self.news_path.open("rb") as handle:
            news_digest = hashlib.file_digest(handle, "sha256").hexdigest()
        self.offsets = array("Q")
        source_digest = hashlib.sha256()
        seen = set()
        exposures = positives = max_candidates = 0
        with self.behaviors_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                source_digest.update(raw)
                fields, candidates = self._parse(raw, len(self.offsets) + 1)
                if fields[0] in seen:
                    raise ValueError(f"Duplicate source impression: {fields[0]}")
                seen.add(fields[0])
                self.offsets.append(offset)
                exposures += len(candidates)
                positives += sum(label for _, label in candidates)
                max_candidates = max(max_candidates, len(candidates))
        if not self.offsets:
            raise ValueError("Complete impression source is empty")
        self._check_sources()
        self.source_manifest = {
            "behaviors_sha256": source_digest.hexdigest(), "news_sha256": news_digest,
            "impression_ids_sha256": ids_digest(seen),
            "impressions": len(self.offsets), "exposures": exposures, "positives": positives,
            "max_candidates": max_candidates, "candidate_sampling": "none",
        }

    def _check_sources(self):
        if (fingerprint(self.behaviors_path) != self._behavior_fingerprint
                or fingerprint(self.news_path) != self._news_fingerprint):
            raise ValueError("Indexed MIND source changed; rebuild and reverify the source manifest")

    def _parse(self, raw, number):
        fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
        if (len(fields) != 5 or not fields[0] or fields[0].strip() != fields[0]
                or not fields[4].split()):
            raise ValueError(f"Malformed impression at {self.behaviors_path}:{number}")
        for news_id in fields[3].split():
            if news_id not in self.news:
                raise ValueError(f"Unknown history article {news_id} at impression {fields[0]}")
        candidates = []
        for token in fields[4].split():
            if not token.endswith(("-0", "-1")):
                raise ValueError(f"Unlabeled/invalid candidate at impression {fields[0]}: {token}")
            news_id, label = token.rsplit("-", 1)
            if news_id not in self.news:
                raise ValueError(f"Unknown candidate article {news_id} at impression {fields[0]}")
            candidates.append((news_id, int(label)))
        return fields, candidates

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        self._check_sources()
        with self.behaviors_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            fields, candidates = self._parse(handle.readline(), index + 1)
        history_ids = fields[3].split()
        if self.max_history is not None:
            history_ids = history_ids[-self.max_history:]
        history = [self.news[news_id] for news_id in history_ids]
        features = [
            self._encode_pointwise_sample({
                "impression_id": fields[0], "impression_timestamp": fields[2],
                "history": history, "candidate": self.news[news_id], "label": label,
            })
            for news_id, label in candidates
        ]
        return {"candidates": features, "expected_candidates": len(candidates)}
