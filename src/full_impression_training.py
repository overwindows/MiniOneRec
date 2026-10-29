"""Training-only frozen-JEV article features, fixed history recipes and full refit."""

import argparse
from datetime import datetime
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from full_impression_contract import ARRAYS, check_deadline
from full_impression_ranker import FrozenArticleRanker, full_impression_loss
from jev_content_ranker import digest, write_json
from mind_utils import auc_score, mrr_score, ndcg_score


MANIFEST_SHA = "4daa3377b4601aca4ae5b72e45ed2931becb8dd648dd23998f17fb3e1342a5f4"
PROOF_SHA = "de60c130dcd7ea853ce8d2b7fbe81b02b83d6b604fbbba99b96bbfa1ec841b25"
SOURCE_SHA = "542b4d659fb4ab84b7bfbdb7e4d15af4650797737aa334520396df95e3ec0d29"
MODES = ("mean", "candidate_attention")
EPOCHS = 3


def load_contract(directory, manifest_sha, proof_sha, deadline):
    if digest(directory / "manifest.json") != manifest_sha or digest(directory / "verification.json") != proof_sha:
        raise ValueError("The completed independently replayed training contract changed")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    proof = json.loads((directory / "verification.json").read_bytes())
    if (proof["manifest_sha256"] != manifest_sha or proof["source_result_sha256"] != manifest["source_result_sha256"]
            or not proof["all_histories_days_identities_and_article_metadata_verified"]
            or manifest["stage"] != "all_training_candidates_encoded"
            or manifest["history_limit"] != 30 or manifest["padding_news_index"] != 0
            or manifest["sampling"] != "none;every_source_candidate_in_original_order"
            or not manifest["temporary_split_only"] or not manifest["final_training_requires_all_impressions"]
            or any(proof[key] for key in ("gpu_allocations", "model_fits", "dev_or_test_contents_read"))):
        raise ValueError("Full source-origin evidence and no-sampling/final-all-impressions semantics are required")
    expected = set(ARRAYS) | {"news.ids.json", "news.metadata.jsonl", "impression_ids.txt"}
    if set(manifest["sha256"]) != expected:
        raise ValueError("Unexpected complete training-contract inventory")
    for name, expected_sha in manifest["sha256"].items():
        check_deadline(deadline)
        if digest(directory / name) != expected_sha:
            raise ValueError("A complete training input changed: " + name)
    if any(proof[key] != manifest[name] for key, name in (
            ("impressions_replayed", "impressions"), ("candidate_exposures_replayed", "exposures"),
            ("positives_replayed", "positives"), ("negatives_replayed", "negatives"))):
        raise ValueError("Complete replay counts differ")
    arrays = {name: np.load(directory / name, mmap_mode="r", allow_pickle=False) for name in ARRAYS}
    return manifest, arrays


def batches(indices, arrays, candidate_limit=8192, impression_limit=256):
    if candidate_limit < 1 or impression_limit < 1:
        raise ValueError("Batch capacities must be positive")
    offsets = arrays["candidate_offsets.npy"]
    selected, size = [], 0
    for index in indices:
        count = int(offsets[index + 1] - offsets[index])
        if not 0 < count <= candidate_limit:
            raise ValueError("Never drop or truncate an oversized/empty impression")
        if selected and (size + count > candidate_limit or len(selected) == impression_limit):
            yield np.asarray(selected, dtype=np.int64)
            selected, size = [], 0
        selected.append(int(index))
        size += count
    if selected:
        yield np.asarray(selected, dtype=np.int64)


def batch_inputs(indices, arrays, device):
    offsets = arrays["candidate_offsets.npy"]
    slices = [(int(offsets[index]), int(offsets[index + 1])) for index in indices]
    lengths = np.asarray([end - start for start, end in slices], dtype=np.int64)
    candidates = np.concatenate([arrays["candidate_news.npy"][start:end] for start, end in slices]).astype(np.int64)
    labels = np.concatenate([arrays["candidate_labels.npy"][start:end] for start, end in slices]).astype(np.float32)
    history = np.array(arrays["histories.npy"][indices], dtype=np.int64)
    groups = np.repeat(np.arange(len(indices), dtype=np.int64), lengths)
    return tuple(torch.from_numpy(value).to(device) for value in (history, candidates, labels, groups)), lengths


def epoch(model, indices, arrays, optimizer, seed, deadline):
    order = np.random.default_rng(seed).permutation(indices)
    model.train()
    seen = np.zeros(len(arrays["histories.npy"]), dtype=np.bool_)
    impressions = exposures = positives = steps = 0
    total_loss = 0.
    for batch in batches(order, arrays):
        check_deadline(deadline)
        if seen[batch].any():
            raise ValueError("Training impressions must occur exactly once per epoch")
        seen[batch] = True
        (history, candidates, labels, groups), _ = batch_inputs(batch, arrays, model.features.device)
        optimizer.zero_grad(set_to_none=True)
        loss = full_impression_loss(model(history, candidates, groups), labels, groups, len(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
        impressions += len(batch)
        exposures += len(candidates)
        positives += int(labels.sum())
        steps += 1
        if steps % 200 == 0:
            print(json.dumps({"stage": "full_impression_training", "mode": model.mode,
                              "epoch_seed": seed, "impressions": impressions, "exposures": exposures,
                              "positive_targets": positives, "mean_loss": total_loss / impressions}), flush=True)
    expected_exposures = int(np.sum(arrays["candidate_offsets.npy"][indices + 1]
                                    - arrays["candidate_offsets.npy"][indices], dtype=np.uint64))
    if impressions != len(indices) or int(seen.sum()) != len(indices) or exposures != expected_exposures:
        raise ValueError("Incomplete all-impression/all-candidate training epoch")
    return {"impressions": impressions, "exposures": exposures, "positives": positives,
            "mean_loss": total_loss / impressions, "optimizer_steps": steps}


@torch.inference_mode()
def evaluate(indices, arrays, predictor, output, deadline):
    values = np.empty((len(indices), 4), dtype=np.float64)
    total_candidates = 0
    row = 0
    with output.open("xb") as handle:
        for batch in batches(indices, arrays):
            check_deadline(deadline)
            (history, candidates, labels, groups), lengths = batch_inputs(batch, arrays, predictor.device)
            scores = predictor(history, candidates, groups).detach().cpu().numpy().astype(np.float64)
            targets = labels.cpu().numpy()
            if scores.shape != targets.shape or not np.isfinite(scores).all():
                raise ValueError("Every held-out candidate requires one finite prediction")
            handle.write(scores.tobytes())
            total_candidates += len(scores)
            offset = 0
            for count in lengths:
                y, s = targets[offset:offset + count], scores[offset:offset + count]
                if not 0 < y.sum() < count:
                    raise ValueError("Never drop a held-out impression from any ranking metric")
                values[row] = (auc_score(y, s), mrr_score(y, s), ndcg_score(y, s, 5), ndcg_score(y, s, 10))
                offset += count
                row += 1
    if row != len(indices):
        raise ValueError("Incomplete chronological validation coverage")
    np.save(output.with_suffix(".metrics.npy"), values)
    return {"metrics": dict(zip(("auc", "mrr", "ndcg5", "ndcg10"), values.mean(axis=0).tolist())),
            "impressions": row, "exposures": total_candidates, "scores_sha256": digest(output),
            "per_impression_metrics_sha256": digest(output.with_suffix(".metrics.npy"))}


class ModelPredictor:
    def __init__(self, model):
        self.model = model.eval()
        self.device = model.features.device

    def __call__(self, history, candidates, groups):
        return self.model(history, candidates, groups)


class ControlPredictor:
    def __init__(self, features, kind, margins=None, ctr=None):
        self.features, self.kind = features, kind
        self.device = features.device
        self.margins, self.ctr = margins, ctr

    def __call__(self, history, candidates, groups):
        if self.kind == "popularity":
            return self.ctr[candidates]
        if self.kind == "frozen_article_margin":
            return self.margins[candidates]
        if self.kind != "semantic_mean":
            raise ValueError("Unknown frozen selection control")
        mask = history != 0
        pooled = (F.embedding(history, self.features) * mask.unsqueeze(-1)).sum(dim=1)
        pooled = F.normalize(pooled, dim=-1)[groups]
        return (F.embedding(candidates, self.features) * pooled).sum(dim=-1)


def popularity(indices, arrays, news, deadline):
    exposures, clicks = np.zeros(news + 1, dtype=np.int64), np.zeros(news + 1, dtype=np.int64)
    for batch in batches(indices, arrays):
        check_deadline(deadline)
        (_, candidates, labels, _), _ = batch_inputs(batch, arrays, "cpu")
        ids, y = candidates.numpy(), labels.numpy()
        exposures += np.bincount(ids, minlength=news + 1)
        clicks += np.bincount(ids[y == 1], minlength=news + 1)
    prior = float(clicks.sum() / exposures.sum())
    ctr = (clicks + 20 * prior) / (exposures + 20)
    return np.log(ctr / (1 - ctr)).astype(np.float32)


def clipped_article_state(scorer, article, maximum=512):
    from evaluate_mind_jev import build_jev_state
    text = article["title"] + (" " + article["abstract"] if article["abstract"] else "")

    def state(length):
        return build_jev_state([], {"text": text[:length], "category": article["category"]})

    full = state(len(text))
    if len(scorer._encode(full)) <= maximum:
        return full, False
    if len(scorer._encode(state(0))) > maximum:
        raise ValueError("Even an empty article cannot retain the full fixed decision prompt")
    low, high = 0, len(text)
    while low + 1 < high:
        mid = (low + high) // 2
        if len(scorer._encode(state(mid))) <= maximum:
            low = mid
        else:
            high = mid
    result = state(low)
    if len(scorer._encode(result)) > maximum:
        raise ValueError("Article clipping must preserve the complete decision answer boundary")
    return result, True


def article_features(contract, model_path, small_data, parity_cache, parity_sha, output, deadline):
    from evaluate_mind_jev import JevScorer
    from full_jev_readout import DATA_SHA256
    from mind_utils import load_news
    from probe_jev_readout import extract
    from uniform_jev_features import check_parity, load_parity, training_states
    model_root = Path(model_path)
    model_files = sorted(model_root.glob("*.safetensors")) + [
        model_root / name for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                                      "tokenizer_config.json", "chat_template.jinja")
        if (model_root / name).is_file()
    ]
    if not any(path.suffix == ".safetensors" for path in model_files):
        raise ValueError("The complete local frozen model weights are required")
    fingerprints = {}
    for path in model_files:
        check_deadline(deadline)
        fingerprints[path.name] = digest(path)
    for name, expected in DATA_SHA256.items():
        if digest(small_data / name) != expected:
            raise ValueError("Original frozen-parity source changed")
    rows, expected_x, expected_base = load_parity(parity_cache, parity_sha)
    news = load_news(str(small_data / "train" / "news.tsv"), use_abstract=True)
    scorer = JevScorer(model_path, prompt_contract="legacy", max_prompt_tokens=4095)
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(False)
    x, base = extract(scorer, training_states(rows, small_data / "train" / "behaviors.tsv", news),
                      None, "original_exact16_parity", report_progress=False)
    check_parity(x, base, expected_x, expected_base)
    scorer.max_prompt_tokens = 0
    records = [json.loads(line) for line in (contract / "news.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    names = json.loads((contract / "news.ids.json").read_bytes())
    if [row["news_id"] for row in records] != names:
        raise ValueError("Every article representation must retain the verified source identity/order")
    features = np.lib.format.open_memmap(output / "article.features.npy", mode="w+",
                                        dtype=np.float32, shape=(len(records) + 1, 2560))
    margins = np.zeros(len(records) + 1, dtype=np.float32)
    features[0] = 0
    clipped = 0
    for start in range(0, len(records), 16):
        check_deadline(deadline)
        states = []
        for row in records[start:start + 16]:
            state, shortened = clipped_article_state(scorer, row)
            clipped += int(shortened)
            states.append({"state": state})
        x, base = extract(scorer, states, None, "frozen_article_features", report_progress=False)
        if (x.shape != (len(states), 2560) or base.shape != (len(states),)
                or np.any(np.linalg.norm(x, axis=1) == 0)):
            raise ValueError("Every training article requires one complete frozen representation")
        features[start + 1:start + 1 + len(states)] = x
        margins[start + 1:start + 1 + len(states)] = base
        if start % 256 == 0:
            print(json.dumps({"stage": "frozen_article_features", "articles": start + len(states),
                              "total": len(records), "clipped_text_prefixes": clipped}), flush=True)
    features.flush()
    del features, scorer
    gc.collect()
    torch.cuda.empty_cache()
    for path in model_files:
        check_deadline(deadline)
        if digest(path) != fingerprints[path.name]:
            raise ValueError("Frozen model bytes changed during article extraction")
    write_json(output / "model.first-observed-sha256.json", fingerprints)
    np.save(output / "article.margins.npy", margins)
    result = {"articles": len(records), "dimension": 2560, "clipped_text_prefixes": clipped,
              "maximum_complete_prompt_tokens": 512, "exact_original16_parity": True,
              "backbone_fitting": False, "features_sha256": digest(output / "article.features.npy"),
              "margins_sha256": digest(output / "article.margins.npy"),
              "model_fingerprints_sha256": digest(output / "model.first-observed-sha256.json"),
              "complete_model_bytes_rehashed_before_after": True,
              "external_canonical_model_provenance_newly_verified": False,
              "runtime": {name: version(name) for name in
                          ("torch", "transformers", "tokenizers", "flash-linear-attention", "fla-core", "triton")},
              "gpu": torch.cuda.get_device_name(0)}
    write_json(output / "article.manifest.json", result)
    return result


def fit(features, indices, arrays, mode, output, deadline):
    torch.manual_seed(42)
    model = FrozenArticleRanker(features, mode).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = [epoch(model, indices, arrays, optimizer, 42 + number, deadline) for number in range(EPOCHS)]
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError("Trained history parameters are nonfinite")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, output)
    return model, {"epochs": epochs, "head_sha256": digest(output), "mode": mode}


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve existing semantic-training output; no retries/overwrites")
    deadline = time.monotonic() + args.max_seconds
    manifest, arrays = load_contract(args.contract, MANIFEST_SHA, PROOF_SHA, deadline)
    if (manifest["source_result_sha256"] != SOURCE_SHA or manifest["impressions"] != 2232748
            or manifest["exposures"] != 83507374 or manifest["positives"] != 3383656
            or manifest["negatives"] != 80123718 or manifest["news"] != 101527):
        raise ValueError("Only the preregistered complete MIND-large training population is allowed")
    if not torch.cuda.is_available():
        raise RuntimeError("The registered semantic experiment requires its single bounded GPU")
    args.output.mkdir(parents=True)
    feature = article_features(args.contract, args.model_path, args.small_data, args.parity_cache,
                               args.parity_sha256, args.output, min(deadline, time.monotonic() + 7200))
    features = torch.from_numpy(np.load(args.output / "article.features.npy")).to("cuda:0")
    features = F.normalize(features, dim=-1)
    margins = torch.from_numpy(np.load(args.output / "article.margins.npy")).to(features.device)
    last = datetime.fromisoformat(manifest["temporary_validation_day"]).date().toordinal()
    fitting = np.flatnonzero(arrays["training_days.npy"] < last)
    validation = np.flatnonzero(arrays["training_days.npy"] == last)
    if len(fitting) != 1801231 or len(validation) != 431517:
        raise ValueError("The complete prespecified chronological cohorts are required")
    ctr = torch.from_numpy(popularity(fitting, arrays, manifest["news"], deadline)).to(features.device)
    controls, recipes = {}, {}
    for kind in ("popularity", "frozen_article_margin", "semantic_mean"):
        controls[kind] = evaluate(validation, arrays, ControlPredictor(features, kind, margins, ctr),
                                  args.output / f"validation.{kind}.scores.f64", deadline)
    for mode in MODES:
        model, training = fit(features, fitting, arrays, mode, args.output / f"{mode}.selection.pt", deadline)
        recipes[mode] = {**training, **evaluate(validation, arrays, ModelPredictor(model),
                                               args.output / f"validation.{mode}.scores.f64", deadline)}
        del model
    selected = max(MODES, key=lambda mode: recipes[mode]["metrics"]["auc"])
    control = max(controls, key=lambda kind: controls[kind]["metrics"]["auc"])
    gain = recipes[selected]["metrics"]["auc"] - controls[control]["metrics"]["auc"]
    opens = gain >= .005 and all(recipes[selected]["metrics"][name] >= controls[control]["metrics"][name]
                                for name in ("mrr", "ndcg5", "ndcg10"))
    result = {
        "stage": "training_only_gate_evaluated", "source_revision": args.source_revision,
        "contract_manifest_sha256": MANIFEST_SHA, "contract_proof_sha256": PROOF_SHA,
        "source_result_sha256": SOURCE_SHA, "feature_extraction": feature,
        "controls": controls, "recipes": recipes, "selected_recipe": selected, "strongest_control": control,
        "training_only_auc_gain": gain, "opens_dev_gate": opens,
        "metric_scope": "last_official_training_day_only;not_official_dev_or_test",
        "recipe_contract": {"epochs_per_selection_and_final_fit": EPOCHS, "seed": 42, "dimension": 128,
                            "learning_rate": .001, "weight_decay": .0001, "gradient_norm_cap": 1.,
                            "maximum_batch_candidates": 8192, "maximum_batch_impressions": 256,
                            "article_feature_normalization": "l2_each_article;zero_padding",
                            "objective": "all_candidates_softmax;all_positives_normalized;equal_impression_weight",
                            "epoch_endpoint": "fixed_three_epochs;no_convergence_claim"},
        "temporary_fitting_impressions": len(fitting), "temporary_validation_impressions": len(validation),
        "gpu_allocations": 1, "dev_or_test_contents_read": False, "competition_test_goal_achieved": False,
    }
    write_json(args.output / "selection.json", result)
    if opens:
        all_indices = np.arange(manifest["impressions"], dtype=np.int64)
        model, training = fit(features, all_indices, arrays, selected, args.output / "selected-full.pt", deadline)
        result.update(stage="full_training_fitted", final_refit=training, final_refit_holdout_impressions=0,
                      training_impressions=manifest["impressions"], training_exposures=manifest["exposures"])
        del model
    check_deadline(deadline)
    write_json(args.output / "result.pending.json", result)
    (args.output / "result.pending.json").rename(args.output / "result.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--small-data", type=Path, required=True)
    parser.add_argument("--parity-cache", type=Path, required=True)
    parser.add_argument("--parity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-seconds", type=int, default=21600)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 21600:
        parser.error("The whole worker has a strict six-hour maximum")
    torch.set_num_threads(4)
    run(args)
