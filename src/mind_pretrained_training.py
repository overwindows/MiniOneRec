"""Complete-impression fitting for a new frozen-GloVe ordered-title hypothesis."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from full_impression_contract import read_news
from full_impression_training import MANIFEST_SHA, PROOF_SHA, epoch, load_contract
from mind_cpu_ranker import Corpus, Priors, Source, digest, write_json
from mind_lexical_glm import ENCODER_HASHES
from mind_pretrained_news import PretrainedNewsRanker
from mind_title_ranker import evaluate, encode_behaviors


EPOCHS = 2
RSS_LIMIT = 3758096384
DEV_NEWS_SHA = "03ba5f02160672500bf5f1af6dd18e8f99830a845b08ca9fd9fbbc9bdd72baff"
DEV_BEHAVIORS_SHA = "7031ced7b8de5183c1bf2c4a303018df8a49d858714597421d9489e571d87b45"


def load_public_encoder(path, corpus):
    with np.load(path, allow_pickle=False) as arrays:
        if arrays["news_names"].tolist() != corpus.names:
            raise ValueError("Frozen public encoder article identities differ")
        return arrays["tokens"], arrays["embeddings"], arrays["vocabulary"].tolist()


def fit(source, arrays, corpus, public_encoder, groups, directory, deadline):
    torch.manual_seed(42)
    tokens, embeddings, vocabulary = load_public_encoder(public_encoder, corpus)
    priors = Priors(source, corpus, groups, deadline, RSS_LIMIT)
    model = PretrainedNewsRanker(corpus, tokens, embeddings, vocabulary, priors)
    print(json.dumps({"stage": "pretrained_model_initialized", "fitting": source.counts(groups)}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0001)
    coverage, expected = [], source.counts(groups)
    mask = np.zeros(len(source.days), dtype=bool)
    mask[groups] = True
    bitmap = hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    for index in range(EPOCHS):
        parts = []
        for fold in (index % 2, 1 - index % 2):
            model.prior_fold = fold
            selected = groups[source.folds[groups] == fold]
            if not len(selected):
                raise ValueError("Both nonempty source folds are required for leakage-free priors")
            print(json.dumps({"stage": "pretrained_fold_started", "epoch": index + 1,
                              "fold": fold, "impressions": len(selected)}), flush=True)
            parts.append(epoch(model, selected, arrays, optimizer, 42 + index, deadline))
        record = {key: sum(part[key] for part in parts)
                  for key in ("impressions", "exposures", "positives", "optimizer_steps")}
        record["mean_loss"] = sum(part["mean_loss"] * part["impressions"] for part in parts) / len(groups)
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Complete pretrained epoch omitted impressions/candidates/clicks")
        coverage.append(dict(record, epoch=index + 1, consumed_source_positions_sha256=bitmap))
        print(json.dumps({"stage": "pretrained_complete_epoch", **coverage[-1]}), flush=True)
    directory.mkdir()
    torch.save(model.state_dict(), directory / "model.pt")
    priors.save(directory)
    write_json(directory / "epoch_coverage.json", coverage)
    write_json(directory / "model_config.json", {
        "hypothesis": "frozen_public_glove_ordered_title_cnn_history_self_attention_v1",
        "public_encoder_sha256": digest(public_encoder), "epochs": EPOCHS,
        "seed": 42, "title_length": 30, "history_length": 30,
        "public_word_dimensions": 300, "learned_word_projection": 32, "news_dimensions": 64,
        "cnn_kernel": 3, "user_attention_heads": 4, "dropout": .2,
        "word_dropout_after_projection": True,
        "optimizer": "AdamW", "learning_rate": .0005, "weight_decay": .0001,
        "loss": "complete_impression_mean_positive_log_softmax", "candidate_sampling": "none",
        "click_prior_training": "opposite_blake2b_impression_fold_only",
        "source_coverage": expected})
    return model


def restore(directory, corpus, public_encoder):
    config = json.loads((directory / "model_config.json").read_bytes())
    if (config["public_encoder_sha256"] != digest(public_encoder) or config["epochs"] != EPOCHS
            or config["candidate_sampling"] != "none"):
        raise ValueError("Immutable public representation or fixed recipe differs")
    model = PretrainedNewsRanker(corpus, *load_public_encoder(public_encoder, corpus), Priors.load(directory))
    model.load_state_dict(torch.load(directory / "model.pt", weights_only=True, map_location="cpu"), strict=True)
    model.eval()
    model.prepare_inference()
    return model


def main(fitter=fit):
    parser = argparse.ArgumentParser(description=inspect.getdoc(inspect.getmodule(fitter)))
    for name in ("contract", "encoder", "public-encoder", "train-news", "dev-news", "dev-behaviors", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.seconds < 1:
        raise ValueError("Fresh bounded pretrained-title attempt required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    deadline = time.monotonic() + args.seconds
    for name, expected in ENCODER_HASHES.items():
        if digest(args.encoder / name) != expected:
            raise ValueError("Frozen label-free metadata corpus changed")
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    print(json.dumps({"stage": "pretrained_complete_contract_authenticated"}), flush=True)
    if (manifest["implementation_sha256"] != digest(Path(__file__).with_name("full_impression_contract.py"))
            or digest(args.train_news) != manifest["input_sha256"]["news.tsv"]):
        raise ValueError("Complete-source implementation/raw news changed")
    corpus = Corpus.load(args.encoder)
    news = read_news(args.train_news)
    if sorted(news) != corpus.names:
        raise ValueError("Complete article metadata identities differ")
    for name in corpus.names:
        if corpus._metadata_hash(news[name]) != corpus.metadata_sha256[name]:
            raise ValueError("Complete article metadata changed")
    public_sha = digest(args.public_encoder)
    source = object.__new__(Source)
    for field, name in (("offsets", "candidate_offsets.npy"), ("candidates", "candidate_news.npy"),
                        ("labels", "candidate_labels.npy"), ("histories", "histories.npy"),
                        ("days", "training_days.npy")):
        setattr(source, field, arrays[name])
    with (args.contract / "impression_ids.txt").open("rb") as handle:
        source.folds = np.fromiter(
            (hashlib.blake2b(line.rstrip(b"\r\n"), digest_size=1).digest()[0] & 1 for line in handle),
            dtype=np.uint8, count=len(source.days))
        if handle.read(1):
            raise ValueError("Extra source impression identity")
    earlier = np.flatnonzero(source.days < source.days.max())
    heldout = np.flatnonzero(source.days == source.days.max())
    if (source.counts(earlier) != {"impressions": 1801231, "exposures": 66107268, "positives": 2712612}
            or source.counts(heldout) != {"impressions": 431517, "exposures": 17400106, "positives": 671044}):
        raise ValueError("Pinned fitting/screening populations changed")
    args.output.mkdir()
    write_json(args.output / "registration.json", {
        "source_sha256": digest(Path(__file__)), "public_encoder_sha256": public_sha,
        "fitter_source_sha256": digest(Path(inspect.getsourcefile(fitter))),
        "manifest_sha256": MANIFEST_SHA, "epochs": EPOCHS, "gate_auc": .69,
        "seconds": args.seconds, "previous_ranker_weights_or_priors_read": False,
        "dev_or_test_contents_read": False, "new_gpu_or_cloud_allocations": 0})
    model = fitter(source, arrays, corpus, args.public_encoder, earlier, args.output / "internal_selection", deadline)
    internal = evaluate(model, source, heldout, args.output / "internal_scores.jsonl", deadline)
    write_json(args.output / "internal_validation.json", internal)
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if digest(args.public_encoder) != public_sha:
        raise ValueError("Frozen public features changed while fitting")
    if internal["metrics"]["auc"] < .69:
        write_json(args.output / "result.json", {
            "stage": "rejected_internal_gate", "internal_validation": internal,
            "final_all_training_fit": False, "official_dev_contents_read": False,
            "competition_test_goal_achieved": False, "model_fits": 1})
        return
    del model
    all_groups = np.arange(len(source.days))
    if source.counts(all_groups) != {"impressions": 2232748, "exposures": 83507374, "positives": 3383656}:
        raise ValueError("Final fitting requires ALL source impressions/candidates")
    model = fitter(source, arrays, corpus, args.public_encoder, all_groups, args.output / "all_training_fit", deadline)
    if digest(args.dev_news) != DEV_NEWS_SHA or digest(args.dev_behaviors) != DEV_BEHAVIORS_SHA:
        raise ValueError("Pinned full official-dev bytes changed")
    model.eval()
    model.extend(news | read_news(args.dev_news))
    dev = encode_behaviors(args.dev_behaviors, corpus, 376471, 14085557)
    if int(dev.labels.sum()) != 574845:
        raise ValueError("Complete official-dev click population changed")
    official = evaluate(model, dev, np.arange(376471), args.output / "official_dev_scores.jsonl", deadline)
    if (digest(args.dev_news) != DEV_NEWS_SHA or digest(args.dev_behaviors) != DEV_BEHAVIORS_SHA
            or digest(args.public_encoder) != public_sha):
        raise ValueError("Immutable representation/dev sources changed during evaluation")
    load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    write_json(args.output / "result.json", {
        "stage": "complete_all_training_fit_and_official_dev_scored",
        "internal_validation": internal, "official_dev": official,
        "final_all_training_fit": True, "official_dev_contents_read": True,
        "competition_test_contents_read": False, "competition_test_goal_achieved": False, "model_fits": 2})


if __name__ == "__main__":
    main()
