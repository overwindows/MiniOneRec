"""Pinned public contextual title vectors; never consumes click labels."""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = {
    "config.json": (612, "72b987fd805cfa2b58c4c8c952b274a11bfd5a00"),
    "special_tokens_map.json": (112, "e7b0375001f109a6b8873d756ad4f7bbb15fbaa5"),
    "tokenizer.json": (466247, "cb202bfe2e3c98645018a6d12f182a434c9d3e02"),
    "tokenizer_config.json": (350, "c79f2b6a0cea6f4b564fed1938984bace9d30ff0"),
    "vocab.txt": (231508, "fb140275c155a9c7c5a3b3e0e77a9e839594a938"),
    "model.safetensors": (90868376, "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db")}
NEWS_SHA = "8d16f3e4df6ac8d8943c5f29e9155d0daf73047f1e41d37133013fe0e19953e9"
CORPUS_SHA = "1a678cbb88976656001fe868c85edb58786c3db1a299d34520cc860b04262467"
RSS_LIMIT = 3758096384


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def limits(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Contextual preparation reached its unchanged cutoff")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RSS_LIMIT:
        raise MemoryError("Contextual preparation exceeded the unchanged RSS ceiling")


def masked_pool(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * weights).sum(1) / weights.sum(1).clamp_min(1), dim=1)


class PublicEncoder:
    def __init__(self, directory):
        torch.set_num_threads(4)
        directory = Path(directory)
        for name, (size, expected) in FILES.items():
            path = directory / name
            if path.stat().st_size != size:
                raise ValueError("Pinned public contextual file length changed: " + name)
            content = path.read_bytes()
            actual = (hashlib.sha256(content).hexdigest() if name.endswith(".safetensors")
                      else hashlib.sha1(f"blob {size}\0".encode() + content).hexdigest())
            if actual != expected:
                raise ValueError("Pinned public contextual file changed: " + name)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.model, info = AutoModel.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, attn_implementation="eager", output_loading_info=True)
        if (info.get("mismatched_keys") or info.get("error_msgs")
                or any(not key.startswith("pooler.") for key in info.get("missing_keys", []))
                or any(not key.startswith("pooler.") for key in info.get("unexpected_keys", []))):
            raise ValueError("Public contextual encoder did not restore every used parameter")
        self.model.eval().requires_grad_(False)
        if any(parameter.device.type != "cpu" for parameter in self.model.parameters()):
            raise ValueError("Local CPU-only public encoding required")

    def encode(self, texts):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        encoded = self.tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**encoded).last_hidden_state
            result = masked_pool(hidden, encoded["attention_mask"]).numpy()
        if result.shape != (len(texts), 384) or not np.isfinite(result).all():
            raise ValueError("Complete finite public contextual vectors required")
        return result


def prepare(directory, news_path, corpus_path, output, deadline):
    if (digest(news_path) != NEWS_SHA or digest(corpus_path) != CORPUS_SHA
            or output.exists() or output.with_suffix(".json").exists()):
        raise ValueError("Pinned metadata inputs and fresh public-vector output required")
    document = json.loads(corpus_path.read_bytes())
    names = document["names"]
    titles = {}
    with news_path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 8 or fields[0] in titles:
                raise ValueError("Malformed/duplicate raw training article")
            titles[fields[0]] = fields[3]
    if set(titles) != set(names):
        raise ValueError("Frozen contextual article identities changed")
    encoder = PublicEncoder(directory)
    vectors = np.zeros((len(names) + 1, 384), dtype=np.float32)
    started = time.monotonic()
    for start in range(0, len(names), 128):
        limits(deadline)
        selected = names[start:start + 128]
        vectors[start + 1:start + 1 + len(selected)] = encoder.encode([titles[name] for name in selected])
        if start % 4096 == 0:
            print(json.dumps({"stage": "contextual_titles", "completed": start + len(selected),
                              "total": len(names), "elapsed": time.monotonic() - started}), flush=True)
    if digest(news_path) != NEWS_SHA or digest(corpus_path) != CORPUS_SHA:
        raise ValueError("Pinned metadata changed while encoding")
    limits(deadline)
    with output.open("xb") as handle:
        np.savez(handle, vectors=vectors, names=np.asarray(names))
    with output.with_suffix(".json").open("x") as handle:
        json.dump({"model": "sentence-transformers/all-MiniLM-L6-v2", "revision": REVISION,
                   "vectors_sha256": digest(output), "source_sha256": digest(Path(__file__)),
                   "news_sha256": NEWS_SHA, "corpus_sha256": CORPUS_SHA,
                   "rows": len(vectors), "dimensions": 384, "maximum_tokens": 64,
                   "text": "title_only", "pooling": "masked_mean_then_l2_normalize",
                   "dtype": "float32", "public_encoder_fitted": False,
                   "click_labels_read": False, "dev_or_test_contents_read": False,
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-directory", "news", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        raise ValueError("Positive bounded preparation budget required")
    prepare(args.public_directory, args.news, args.corpus, args.output, time.monotonic() + args.seconds)


if __name__ == "__main__":
    main()
