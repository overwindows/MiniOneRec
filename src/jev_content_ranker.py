"""Train-only selection of a frozen-JEV, content/history and popularity ranker."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from behavior_jev_readout import build_counts, behavior_features, save_counts
from mind_utils import auc_score, mrr_score, ndcg_score


TRAIN_NEWS_SHA = "cd49d7da275e9c405f627d75a0de9d4ec6ad43ff8a8db5d23a447591676e09d6"
TRAIN_BEHAVIORS_SHA = "a424547c8fa17c9ea4879c2110221c2b2f2709f4f7615ede0b0d8e4765158658"
ROWS_SHA = "b4dba4f73cdf5878ba1d67582d8b81899240e13c3670f38b8b26ee04605af254"
MARGINS_SHA = "892be41a4f09a72ed98dab45fbd9b714ba5225840cb52d16070ca1fdeaf5546f"
FEATURE_NAMES = (
    "frozen_jev_margin", "category", "subcategory", "history_length",
    "candidate_text_length", "missing_metadata", "missing_history_fraction",
    "category_fraction", "subcategory_fraction", "recent_category_fraction",
    "recent_subcategory_fraction", "last_category_match", "last_subcategory_match",
    "semantic_mean", "semantic_recent", "semantic_last", "semantic_max",
    "semantic_top3", "semantic_min", "semantic_positive_fraction",
    "history_seen_candidate", "log_exposure_rate", "log_click_rate",
    "smoothed_ctr_log_odds", "known_item",
)
BASE_COLUMNS = (0, 21, 22, 23, 24)
RECIPES = (
    {"name": "frozen_popularity", "columns": BASE_COLUMNS, "leaves": 15, "iterations": 120},
    {"name": "content_15", "columns": tuple(range(len(FEATURE_NAMES))), "leaves": 15, "iterations": 120},
    {"name": "content_31", "columns": tuple(range(len(FEATURE_NAMES))), "leaves": 31, "iterations": 160},
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_news(path):
    result = {}
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 8 or not fields[0] or fields[0] in result:
                raise ValueError(f"Malformed or duplicate news at line {number}")
            result[fields[0]] = {
                "category": fields[1], "subcategory": fields[2],
                "text": fields[3] + " " + fields[4],
            }
    if not result:
        raise ValueError("News cannot be empty")
    return result


def behavior_records(path):
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5 or not fields[0] or fields[0] in seen or not fields[4]:
                raise ValueError(f"Malformed or duplicate impression at line {number}")
            seen.add(fields[0])
            items = []
            for token in fields[4].split():
                news, label = token.rsplit("-", 1)
                if not news or label not in ("0", "1"):
                    raise ValueError(f"Invalid candidate at line {number}")
                items.append((news, int(label)))
            if len(dict(items)) != len(items):
                raise ValueError(f"Duplicate candidate at line {number}")
            yield {
                "impression_id": fields[0],
                "day": datetime.strptime(fields[2], "%m/%d/%Y %I:%M:%S %p").date().isoformat(),
                "history": tuple(fields[3].split()[-30:]), "items": items,
            }


def category_codes(news, field):
    counts = Counter(article[field] for article in news.values() if article[field])
    return {name: index + 1 for index, (name, _) in enumerate(counts.most_common(254))}


def align_records(records, rows):
    indexed = {}
    for record in records:
        if record["impression_id"] in indexed:
            raise ValueError("Duplicate content impression")
        indexed[record["impression_id"]] = record
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["impression_id"]].append((row["candidate_id"], row["label"]))
    if set(indexed) != set(grouped):
        raise ValueError("Content and cached impression coverage differ")
    aligned = []
    for impression, items in grouped.items():
        actual = dict(indexed[impression]["items"])
        if len(dict(items)) != len(items) or any(actual.get(name) != label for name, label in items):
            raise ValueError("Content candidate identities or labels differ from cached rows")
        aligned.append({**indexed[impression], "items": items})
    return aligned


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class Content:
    def __init__(self, vectorizer, svd, categories, subcategories):
        self.vectorizer, self.svd = vectorizer, svd
        self.categories, self.subcategories = categories, subcategories

    @classmethod
    def fit(cls, news, components=64):
        vectorizer = TfidfVectorizer(
            max_features=32768, min_df=2, sublinear_tf=True, dtype=np.float32,
            strip_accents="unicode", stop_words="english", ngram_range=(1, 2),
        )
        terms = vectorizer.fit_transform([value["text"] for value in news.values()])
        dimension = min(components, terms.shape[0] - 1, terms.shape[1] - 1)
        if dimension < 2:
            raise ValueError("Content encoding requires at least three news articles and terms")
        svd = TruncatedSVD(n_components=dimension, n_iter=5, random_state=42)
        svd.fit(terms)
        return cls(vectorizer, svd, category_codes(news, "category"),
                   category_codes(news, "subcategory"))

    def table(self, news):
        names = list(news)
        vectors = normalize(self.svd.transform(
            self.vectorizer.transform([news[name]["text"] for name in names]),
        )).astype(np.float32)
        vectors = np.concatenate((np.zeros((1, vectors.shape[1]), dtype=np.float32), vectors))
        index = {name: row + 1 for row, name in enumerate(names)}
        categories = np.asarray([0] + [self.categories.get(news[name]["category"], 0) for name in names])
        subcategories = np.asarray([0] + [self.subcategories.get(news[name]["subcategory"], 0) for name in names])
        lengths = np.asarray([0] + [len(news[name]["text"].split()) for name in names])
        return index, vectors, categories, subcategories, lengths


def content_features(records, table):
    index, vectors, categories, subcategories, lengths = table
    blocks, rows = [], []
    for record in records:
        ids = np.asarray([index.get(name, 0) for name, _ in record["items"]])
        history = np.asarray([index.get(name, 0) for name in record["history"]], dtype=int)
        count = len(ids)
        block = np.zeros((count, 20), dtype=np.float32)
        block[:, :5] = np.column_stack((
            categories[ids], subcategories[ids], np.full(count, np.log1p(len(history))),
            np.log1p(lengths[ids]), ids == 0,
        ))
        if len(history):
            hcat, hsub = categories[history], subcategories[history]
            cat, sub = categories[ids], subcategories[ids]
            similarities = vectors[ids] @ vectors[history].T
            block[:, 5] = np.mean(history == 0)
            block[:, 6] = ((cat[:, None] == hcat) & (cat[:, None] != 0)).mean(axis=1)
            block[:, 7] = ((sub[:, None] == hsub) & (sub[:, None] != 0)).mean(axis=1)
            block[:, 8] = ((cat[:, None] == hcat[-5:]) & (cat[:, None] != 0)).mean(axis=1)
            block[:, 9] = ((sub[:, None] == hsub[-5:]) & (sub[:, None] != 0)).mean(axis=1)
            block[:, 10] = (cat == hcat[-1]) & (cat != 0)
            block[:, 11] = (sub == hsub[-1]) & (sub != 0)
            block[:, 12] = vectors[ids] @ normalize(vectors[history].mean(axis=0))
            block[:, 13] = vectors[ids] @ normalize(vectors[history[-5:]].mean(axis=0))
            block[:, 14] = similarities[:, -1]
            block[:, 15] = similarities.max(axis=1)
            block[:, 16] = np.sort(similarities, axis=1)[:, -3:].mean(axis=1)
            block[:, 17] = similarities.min(axis=1)
            block[:, 18] = (similarities > 0.3).mean(axis=1)
            block[:, 19] = np.isin(ids, history[history != 0])
        blocks.append(block)
        rows.extend({"impression_id": record["impression_id"], "candidate_id": name, "label": label}
                    for name, label in record["items"])
    if not blocks:
        raise ValueError("Content feature rows cannot be empty")
    features = np.concatenate(blocks)
    if not np.isfinite(features).all():
        raise FloatingPointError("Content features are nonfinite")
    return rows, features


def assembled_features(rows, content, margins, counts, totals, cross_fitted):
    margins = np.asarray(margins, dtype=np.float32)
    if content.shape != (len(rows), 20) or margins.shape != (len(rows),):
        raise ValueError("Content, margins and row coverage differ")
    population = behavior_features(rows, counts, totals, cross_fitted).astype(np.float32)
    features = np.column_stack((margins, content, population))
    if not np.isfinite(features).all():
        raise FloatingPointError("Ranker features are nonfinite")
    return features


def ranking_metrics(rows, scores):
    scores = np.asarray(scores)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite and aligned")
    buckets = defaultdict(list)
    for position, row in enumerate(rows):
        buckets[row["impression_id"]].append(position)
    values, skipped = [], 0
    for positions in buckets.values():
        labels = np.asarray([rows[position]["label"] for position in positions])
        if not 0 < labels.sum() < len(labels):
            skipped += 1
            continue
        selected = scores[positions]
        values.append([auc_score(labels, selected), mrr_score(labels, selected),
                       ndcg_score(labels, selected, 5), ndcg_score(labels, selected, 10)])
    if not values:
        raise ValueError("No mixed-label impressions for ranking metrics")
    return {
        "metrics": dict(zip(("auc", "mrr", "ndcg5", "ndcg10"),
                            np.mean(values, axis=0).tolist())),
        "scored_impressions": len(values), "single_class_impressions": skipped,
    }


def fit_model(recipe, features, rows):
    columns = recipe["columns"]
    categorical = [position for position, column in enumerate(columns) if column in (1, 2)]
    sizes = Counter(row["impression_id"] for row in rows)
    weights = np.asarray([1. / sizes[row["impression_id"]] for row in rows])
    model = HistGradientBoostingClassifier(
        learning_rate=.06, max_iter=recipe["iterations"], max_leaf_nodes=recipe["leaves"],
        min_samples_leaf=100, l2_regularization=10., categorical_features=categorical or None,
        early_stopping=False, random_state=42,
    )
    model.fit(features[:, columns], [row["label"] for row in rows], sample_weight=weights)
    return model


def train(train_news, train_behaviors, sampled_rows, train_margins, output, source_revision):
    if output.exists():
        raise FileExistsError("Content-ranker output must not already exist")
    expected = ((train_news, TRAIN_NEWS_SHA), (train_behaviors, TRAIN_BEHAVIORS_SHA),
                (sampled_rows, ROWS_SHA), (train_margins, MARGINS_SHA))
    for path, checksum in expected:
        if digest(path) != checksum:
            raise ValueError("Training input checksum differs: " + path.name)
    rows = json.loads(sampled_rows.read_bytes())
    with np.load(train_margins, allow_pickle=False) as stored:
        margins = stored["frozen"].copy()
    if margins.shape != (len(rows),) or not np.isfinite(margins).all():
        raise ValueError("Frozen margins do not cover every training row")
    sampled = defaultdict(list)
    for row in rows:
        sampled[row["impression_id"]].append((row["candidate_id"], row["label"]))
    news = read_news(train_news)
    records, days = [], []
    for record in behavior_records(train_behaviors):
        impression = record["impression_id"]
        actual = dict(record["items"])
        if impression not in sampled or any(actual.get(name) != label
                                           for name, label in sampled[impression]):
            raise ValueError("Sampled candidate/label differs from source training impression")
        record["items"] = sampled.pop(impression)
        records.append(record)
        days.append(record["day"])
    if sampled or len(rows) != 469264 or len(records) != 156965:
        raise ValueError("Training must cover all 156965 official impressions and 469264 sampled rows")
    records = align_records(records, rows)
    output.mkdir(parents=True)
    validation_day = max(days)
    excluded = {record["impression_id"] for record in records if record["day"] == validation_day}
    fit_mask = np.asarray([row["impression_id"] not in excluded for row in rows])
    if not fit_mask.any() or fit_mask.all():
        raise ValueError("Temporal training selection requires at least two training dates")
    write_json(output / "protocol.json", {
        "source_revision": source_revision, "training_only": True, "selection_day": validation_day,
        "selection_rule": "Highest last-training-day impression AUC; fixed recipe order breaks ties",
        "compute_gate": "Selected content model exceeds frozen-popularity baseline by at least0.005 AUC",
        "feature_names": FEATURE_NAMES, "recipes": RECIPES,
        "validation_labels_excluded_from_all_training_features": True,
        "training_own_impression_fold_excluded": True, "training_impressions": len(records),
        "training_rows": len(rows), "final_refit_holdout_impressions": 0,
        "official_dev_evaluations": 0,
        "limitation": "Sampled training validation is a compute gate,not benchmark quality evidence",
        "inputs": {path.name: checksum for path, checksum in expected},
    })
    print(json.dumps({"stage": "fitting_train_only_content", "validation_day": validation_day}), flush=True)
    content = Content.fit(news)
    actual_rows, content_x = content_features(records, content.table(news))
    if actual_rows != rows:
        raise ValueError("Content features changed cached training row identities or order")
    del actual_rows, records
    fit_rows = [row for row, keep in zip(rows, fit_mask) if keep]
    validation_rows = [row for row, keep in zip(rows, fit_mask) if not keep]
    counts, totals = build_counts(train_behaviors, excluded, rows)
    x = assembled_features(fit_rows, content_x[fit_mask], margins[fit_mask], counts, totals, True)
    vx = assembled_features(validation_rows, content_x[~fit_mask], margins[~fit_mask], counts, totals, False)
    candidates = []
    for recipe in RECIPES:
        model = fit_model(recipe, x, fit_rows)
        scores = model.decision_function(vx[:, recipe["columns"]])
        values = ranking_metrics(validation_rows, scores)
        candidates.append({"name": recipe["name"], **values})
        print(json.dumps({"stage": "train_only_candidate", **candidates[-1]}), flush=True)
    best = max(candidates[1:], key=lambda value: value["metrics"]["auc"])
    baseline = candidates[0]
    opens = best["metrics"]["auc"] - baseline["metrics"]["auc"] >= .005
    selection = {
        "selected": best["name"], "candidates": candidates, "opens_dev_gate": opens,
        "training_only": True, "official_dev_evaluations": 0,
        "selection_day": validation_day, "selection_impressions": len(excluded),
    }
    write_json(output / "train-selection.json", selection)
    if not opens:
        write_json(output / "result.json", {
            **selection, "stage": "train_only_gate_closed", "quality_result": None,
        })
        return selection
    del x, vx
    counts, totals = build_counts(train_behaviors, set(), rows)
    x = assembled_features(rows, content_x, margins, counts, totals, True)
    recipe = next(recipe for recipe in RECIPES if recipe["name"] == best["name"])
    print(json.dumps({"stage": "refitting_all_official_training_impressions", "rows": len(rows)}), flush=True)
    model = fit_model(recipe, x, rows)
    save_counts(output / "population.npz", counts, totals)
    joblib.dump({
        "model": model, "content": {
            "vectorizer": content.vectorizer, "svd": content.svd,
            "categories": content.categories, "subcategories": content.subcategories,
        }, "news": news, "recipe": recipe,
    }, output / "model.joblib", compress=3)
    result = {
        **selection, "stage": "full_training_fitted", "training_rows": len(rows),
        "training_impressions": len(set(row["impression_id"] for row in rows)),
        "final_refit_holdout_impressions": 0, "source_revision": source_revision,
        "missing_metadata_rows": int(content_x[:, 4].sum()),
        "rows_with_missing_history_metadata": int((content_x[:, 5] > 0).sum()),
        "model_sha256": digest(output / "model.joblib"),
        "population_sha256": digest(output / "population.npz"),
        "train_selection_sha256": digest(output / "train-selection.json"),
        "quality_result": None,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-news", type=Path, required=True)
    parser.add_argument("--train-behaviors", type=Path, required=True)
    parser.add_argument("--sampled-rows", type=Path, required=True)
    parser.add_argument("--train-margins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    train(args.train_news, args.train_behaviors, args.sampled_rows,
          args.train_margins, args.output, args.source_revision)


if __name__ == "__main__":
    main()
