"""Replay the complete raw official-dev source against every encoded identity and target."""

import json

import numpy as np

from full_impression_contract import check_deadline, read_news
from full_impression_evaluation import small_signatures, source_signature
from jev_content_ranker import digest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_source(dev, small_dev, train_contract, encoded, deadline):
    check_deadline(deadline)
    report = json.loads((encoded / "manifest.json").read_bytes())
    n, m = report["impressions"], report["exposures"]
    require(0 < n < m, "Complete official-dev dimensions are required")
    specs = {
        "candidate_news.npy": (np.dtype("int32"), (m,)),
        "candidate_labels.npy": (np.dtype("uint8"), (m,)),
        "candidate_offsets.npy": (np.dtype("uint64"), (n + 1,)),
        "histories.npy": (np.dtype("int32"), (n, 30)),
        "small_dev_signature_overlap.npy": (np.dtype("bool"), (n,)),
    }
    inventory = set(specs) | {"news.ids.json", "impression_ids.txt",
                              "new_articles/news.ids.json", "new_articles/news.metadata.jsonl"}
    require(set(report["sha256"]) == inventory, "Complete official-dev contract inventory required")
    for name in inventory:
        check_deadline(deadline)
        require(digest(encoded / name) == report["sha256"][name], "Official-dev contract bytes changed")
    sources = {"dev_news": dev / "news.tsv", "dev_behaviors": dev / "behaviors.tsv",
               "small_dev_behaviors": small_dev / "behaviors.tsv"}
    require(set(report["source_sha256"]) == set(sources), "Exact raw-source inventory required")
    for name, path in sources.items():
        require(digest(path) == report["source_sha256"][name], "Raw official-dev source bytes changed")
    train_ids = json.loads((train_contract / "news.ids.json").read_bytes())
    train_records = [json.loads(line) for line in
                     (train_contract / "news.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    require(train_ids == [row["news_id"] for row in train_records] and len(set(train_ids)) == len(train_ids),
            "Exact training prefix identities are required")
    news = read_news(dev / "news.tsv")
    train_news = dict(zip(train_ids, train_records))
    require(all(news[name] == train_news[name] for name in set(news) & set(train_news)),
            "Reused article metadata differs from raw source")
    new_ids = sorted(set(news) - set(train_ids))
    require(json.loads((encoded / "news.ids.json").read_bytes()) == train_ids + new_ids
            and json.loads((encoded / "new_articles" / "news.ids.json").read_bytes()) == new_ids,
            "Union identities must be the exact training prefix and sorted new articles")
    new_records = [json.loads(line) for line in
                   (encoded / "new_articles" / "news.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    require(new_records == [news[name] for name in new_ids], "New article metadata differs from raw source")
    require(report["training_articles_reused"] == len(train_ids)
            and report["new_dev_articles"] == len(new_ids) and report["dev_news"] == len(news)
            and report["union_articles"] == len(train_ids) + len(new_ids)
            and report["history_limit"] == 30 and report["history_padding"] == "left_zero"
            and report["candidate_sampling"] == "none", "Complete union and no-sampling semantics required")
    arrays = {name: np.load(encoded / name, mmap_mode="r", allow_pickle=False) for name in specs}
    require(all(arrays[name].dtype == dtype and arrays[name].shape == shape
                for name, (dtype, shape) in specs.items()), "Lossless exact-sized source arrays required")
    require(arrays["candidate_offsets.npy"][0] == 0, "Candidate offsets must start at zero")
    lookup = {name: index + 1 for index, name in enumerate(train_ids + new_ids)}
    signatures, small_rows = small_signatures(small_dev / "behaviors.tsv", deadline)
    seen = set()
    position = positives = overlaps = rows = 0
    with (dev / "behaviors.tsv").open(encoding="utf-8") as raw, (
            encoded / "impression_ids.txt").open(encoding="utf-8") as identities:
        for index, (line, identity) in enumerate(zip(raw, identities, strict=True)):
            if index % 4096 == 0:
                check_deadline(deadline)
            fields = line.rstrip("\r\n").split("\t")
            require(len(fields) == 5 and bool(fields[0]) and fields[0] not in seen and index < n,
                    "Every unique raw impression must occur once")
            seen.add(fields[0])
            require(identity.rstrip("\r\n") == fields[0], "Impression identity/order differs from raw source")
            tokens = [token.rsplit("-", 1) for token in fields[4].split()]
            require(bool(tokens) and all(len(token) == 2 and token[1] in ("0", "1") for token in tokens),
                    "Every source candidate needs its binary target")
            names = [name for name, _ in tokens]
            labels = np.asarray([int(label) for _, label in tokens], dtype=np.uint8)
            require(0 < int(labels.sum()) < len(labels), "Every raw impression must enter every metric")
            history = fields[3].split()[-30:]
            require(all(name in lookup for name in history + names), "Every article must resolve")
            end = position + len(names)
            require(end <= m and arrays["candidate_offsets.npy"][index] == position
                    and arrays["candidate_offsets.npy"][index + 1] == end,
                    "Candidate offsets differ from raw source")
            require(np.array_equal(arrays["candidate_news.npy"][position:end], [lookup[name] for name in names])
                    and np.array_equal(arrays["candidate_labels.npy"][position:end], labels),
                    "Candidate identity/order/targets differ from raw source")
            expected_history = np.zeros(30, dtype=np.int32)
            if history:
                expected_history[-len(history):] = [lookup[name] for name in history]
            require(np.array_equal(arrays["histories.npy"][index], expected_history),
                    "History differs from raw source")
            overlap = source_signature(history, names) in signatures
            require(bool(arrays["small_dev_signature_overlap.npy"][index]) == overlap,
                    "Overlap differs from raw source")
            positives += int(labels.sum())
            overlaps += overlap
            position, rows = end, index + 1
    require(rows == n and position == m and positives == report["positives"]
            and m - positives == report["negatives"]
            and overlaps == report["small_dev_signature_overlap_impressions"]
            and small_rows == report["small_dev_source_impressions"], "Every source row and count must match")
    for name, path in sources.items():
        require(digest(path) == report["source_sha256"][name], "Raw source changed during independent replay")
    return {"impressions": rows, "exposures": position, "positives": positives,
            "small_dev_signature_overlap_impressions": overlaps,
            "source_sha256": report["source_sha256"], "all_source_identities_targets_histories_verified": True}
