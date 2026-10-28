"""Registered train-only selection and conditional all-impression corrected-data refit."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from behavior_jev_readout import behavior_features, build_counts, save_counts
from evaluate_jev_readout import load_head
from jev_content_ranker import (
    behavior_records, digest, ranking_metrics, write_json, TRAIN_BEHAVIORS_SHA, TRAIN_NEWS_SHA,
)
from streamed_jev_readout import (
    check_deadline, fit_streamed, predict_streamed, select_joint, validation_impressions,
)


FEATURE_REVISION = "15c1c56bb519acd998d737a6b57697f36c84f1fc"
ROWS_SHA = "fd9f50c2668cd50ae03f918636010f84e43563c73c711997ef721ea30a9a3a84"
REFERENCE_SHA = "279ef6c47f09e7fe49b803810159cac6fba7ee23d612ea33318af6ec37f86457"
REFERENCE_ROWS_SHA = "9e19ff15e63d93d8add5dd510acb410d1ea9da77b471526c809915e6208eb565"
REFERENCE_SELECTION_SHA = "c92a5c914f5dc7b99928c34bacf18fe5f15243f68f4127618776be0a6e55fe15"
POPULATION_SHA = "993852a23416f436ed1dd0d7f6a8fde75dfb884b69da9c6b3242a1d00e67d06f"
RECIPES = (
    {"name": "uniform_binary", "balanced": False, "pair_fraction": 0., "l2": .01},
    {"name": "uniform_balanced", "balanced": True, "pair_fraction": 0., "l2": .01},
    {"name": "uniform_mixed", "balanced": True, "pair_fraction": .5, "l2": .01},
)
CONTROL = {"name": "retained_text_fresh_popularity", "balanced": True, "pair_fraction": 1., "l2": .01}


def train(features_root, campaign_path, data_root, reference_cache, reference_path, output, revision):
    created = datetime.now(timezone.utc)
    campaign = json.loads(campaign_path.read_bytes())
    remaining = (datetime.fromisoformat(campaign["campaign_deadline"]) - created).total_seconds()
    if remaining <= 60:
        raise TimeoutError("Original uniform campaign deadline is exhausted")
    deadline = time.monotonic() + min(7140, remaining - 60)
    if output.exists():
        raise FileExistsError("Uniform training output exists; never reset its budget or overwrite it")
    output.mkdir()
    state = {
        "stage": "preflight", "created_at": created.isoformat(), "source_revision": revision,
        "feature_source_revision": FEATURE_REVISION, "training_only": True, "gpu_submissions": 0,
        "official_dev_evaluations": 0, "wall_budget_seconds": 7200,
        "campaign_deadline": campaign["campaign_deadline"],
        "implementation_sha256": digest(Path(__file__)),
        "optimizer_sha256": digest(Path(__file__).with_name("streamed_jev_readout.py")),
        "evaluator_sha256": digest(Path(__file__).with_name("evaluate_uniform_jev.py")),
        "parity_manifest_sha256": campaign.get("parity_manifest_sha256"),
        "opens_dev_gate": False, "quality_result": None,
    }
    write_json(output / "state.json", state)
    try:
        proof = json.loads((features_root / "result.json").read_bytes())
        if (campaign["stage"] != "uniform_features_complete"
                or campaign["source_revision"] != FEATURE_REVISION
                or digest(features_root / "result.json") != campaign["result_sha256"]
                or proof["stage"] != "uniform_features_complete" or proof["rows"] != 472626
                or proof["impressions"] != 156965 or proof["reused_rows"] != 266352
                or proof["new_rows"] != 206274 or proof["rows_sha256"] != ROWS_SHA
                or proof["source_revision"] != FEATURE_REVISION
                or not proof["exactly_once_identity_label_coverage"] or not proof["all_values_finite"]):
            raise ValueError("Corrected feature campaign is incomplete or its provenance differs")
        paths = {
            "features": (features_root / "uniform.features.npy", proof["features_sha256"]),
            "margins": (features_root / "uniform.margins.npy", proof["margins_sha256"]),
            "rows": (features_root / "train.rows.json", ROWS_SHA),
            "behaviors": (data_root / "train" / "behaviors.tsv", TRAIN_BEHAVIORS_SHA),
            "news": (data_root / "train" / "news.tsv", TRAIN_NEWS_SHA),
            "reference_rows": (reference_cache / "train.rows.json", REFERENCE_ROWS_SHA),
            "reference_selection": (reference_cache / "train_holdout.rows.json", REFERENCE_SELECTION_SHA),
            "reference_head": (reference_path, REFERENCE_SHA),
        }
        for name, (path, sha) in paths.items():
            check_deadline(deadline)
            if digest(path) != sha:
                raise ValueError("Pinned uniform training input differs: " + name)
        rows = json.loads(paths["rows"][0].read_bytes())
        records = list(behavior_records(paths["behaviors"][0]))
        day, excluded = validation_impressions(
            records, json.loads(paths["reference_rows"][0].read_bytes()),
            json.loads(paths["reference_selection"][0].read_bytes()),
        )
        classes = defaultdict(set)
        for row in rows:
            classes[row["impression_id"]].add(row["label"])
        if (len(rows) != 472626 or len(classes) != 156965 or len(excluded) != 23988
                or sum(row["label"] for row in rows) != 236344
                or any(values != {0, 1} for values in classes.values())
                or set(classes) != {record["impression_id"] for record in records}):
            raise ValueError("Uniform candidate/impression/class/cohort coverage differs")
        groups = [row["impression_id"] for row in rows]
        labels = np.asarray([row["label"] for row in rows])
        mask = np.asarray([group not in excluded for group in groups])
        validation = [row for row, keep in zip(rows, mask, strict=True) if not keep]
        features = np.load(paths["features"][0], mmap_mode="r", allow_pickle=False)
        frozen = np.load(paths["margins"][0], mmap_mode="r", allow_pickle=False)
        if (features.shape != (472626, 2560) or frozen.shape != (472626,)
                or features.dtype != np.float32 or frozen.dtype != np.float32):
            raise ValueError("Unexpected corrected frozen-feature dimensions or dtype")
        write_json(output / "protocol.json", {
            **state, "feature_result_sha256": campaign["result_sha256"],
            "validation_day": day, "validation_impressions": len(excluded),
            "fitting_impressions": len(classes) - len(excluded), "recipes": RECIPES,
            "control": CONTROL, "optimizer_iterations": 250, "optimizer_history": 100,
            "gate": "AUC>=both controls+0.005 and all secondary metrics nondecreasing",
            "not_fully_chronological": True,
            "all_validation_ids_excluded_from_all_population_statistics": True,
            "fitting_population_features_exclude_own_entire_impression_fold": True,
            "inputs": {name: {"path": str(path), "sha256": sha}
                       for name, (path, sha) in paths.items()},
        })
        write_json(output / "validation.rows.json", validation)
        counts, totals = build_counts(paths["behaviors"][0], excluded, rows)
        extras = np.empty((len(rows), 4))
        extras[mask] = behavior_features(
            [row for row, keep in zip(rows, mask, strict=True) if keep], counts, totals, True,
        )
        extras[~mask] = behavior_features(validation, counts, totals, False)
        save_counts(output / "selection-population.npz", counts, totals)
        reference, reference_bytes = load_head(reference_path, REFERENCE_SHA)
        (output / "reference.head.npz").write_bytes(reference_bytes)
        empty = np.empty((len(rows), 0))
        text = predict_streamed(reference, features, empty, frozen)
        state["stage"] = "fitting_fresh_popularity_control"
        write_json(output / "state.json", state)
        correction, control_fit = fit_streamed(
            extras, empty, text, labels, groups, mask, CONTROL, deadline, max_iter=250,
        )
        np.savez(output / "retained_text_fresh_popularity.head.npz", **correction)
        corrected = predict_streamed(correction, extras, empty, text)
        controls, candidates = [], []
        saved = {"indices": np.flatnonzero(~mask), "labels": labels[~mask]}

        def arm(name, scores, fit=None):
            values = ranking_metrics(validation, scores)
            if values["scored_impressions"] != len(excluded) or values["single_class_impressions"]:
                raise ValueError("Uniform validation must score every held training impression")
            return {"name": name, **values, **({"fit": fit} if fit is not None else {})}

        for name, scores in (("retained_text", text[~mask]), (CONTROL["name"], corrected[~mask])):
            entry = arm(name, scores, control_fit if name == CONTROL["name"] else None)
            controls.append(entry)
            saved[name] = scores
            print(json.dumps({"stage": "uniform_train_only_control", **entry}), flush=True)
        np.savez(output / "training-validation.scores.npz", **saved)
        for recipe in RECIPES:
            check_deadline(deadline)
            state["stage"] = "fitting_" + recipe["name"]
            write_json(output / "state.json", state)
            head, fit = fit_streamed(
                features, extras, frozen, labels, groups, mask, recipe, deadline,
                max_iter=250, initial_head=reference,
            )
            np.savez(output / (recipe["name"] + ".head.npz"), **head)
            scores = predict_streamed(head, features, extras, frozen)[~mask]
            entry = arm(recipe["name"], scores, fit)
            candidates.append(entry)
            saved[recipe["name"]] = scores
            np.savez(output / "training-validation.scores.npz", **saved)
            print(json.dumps({"stage": "uniform_train_only_candidate", **entry}), flush=True)
        best, opens = select_joint(candidates, controls)
        selection = {
            "selected": best["name"], "opens_dev_gate": opens, "controls": controls,
            "candidates": candidates, "training_only": True, "official_dev_evaluations": 0,
            "validation_day": day, "validation_impressions": len(excluded), "quality_result": None,
        }
        write_json(output / "train-selection.json", selection)
        state.update(stage="train_only_gate_closed", selection=selection, opens_dev_gate=opens)
        if opens:
            check_deadline(deadline)
            state["stage"] = "refitting_all_official_training_impressions"
            write_json(output / "state.json", state)
            counts, totals = build_counts(paths["behaviors"][0], set(), rows)
            if totals[0, 0] != 5843444:
                raise ValueError("Final population omits raw official training exposures")
            extras = behavior_features(rows, counts, totals, True)
            recipe = next(recipe for recipe in RECIPES if recipe["name"] == best["name"])
            selected_path = output / (best["name"] + ".head.npz")
            with np.load(selected_path, allow_pickle=False) as stored:
                selected = {name: stored[name] for name in stored.files}
            head, fit = fit_streamed(
                features, extras, frozen, labels, groups, np.ones(len(rows), dtype=bool),
                recipe, deadline, max_iter=250, initial_head=selected,
            )
            np.savez(output / "selected-full.head.npz", **head)
            save_counts(output / "population.npz", counts, totals)
            if digest(output / "population.npz") != POPULATION_SHA:
                raise ValueError("Final all-exposure population differs from the verified original")
            state.update(
                stage="full_training_fitted", final_refit_holdout_impressions=0,
                training_impressions=len(classes), training_rows=len(rows), final_fit=fit,
                head_sha256=digest(output / "selected-full.head.npz"), population_sha256=POPULATION_SHA,
            )
        check_deadline(deadline)
        state.update(
            checked_at=datetime.now(timezone.utc).isoformat(),
            selection_sha256=digest(output / "train-selection.json"),
            validation_scores_sha256=digest(output / "training-validation.scores.npz"),
        )
        write_json(output / "result.json", state)
        write_json(output / "state.json", state)
        print(json.dumps(state), flush=True)
        return state
    except Exception as error:
        state.update(stage="failed", error_type=type(error).__name__, error=str(error),
                     checked_at=datetime.now(timezone.utc).isoformat())
        write_json(output / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("features-root", "campaign", "data-root", "reference-cache", "reference-head", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    train(args.features_root, args.campaign, args.data_root, args.reference_cache,
          args.reference_head, args.output, args.source_revision)
