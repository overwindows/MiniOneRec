"""Disk-backed all-pairs tree fitting without dropping training candidates."""

import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import xgboost as xgb

import mind_tree_ranker as core


PARAMETERS = {
    "objective": "rank:pairwise", "tree_method": "hist", "device": "cpu",
    "eta": .1, "max_depth": 6, "max_leaves": 31, "grow_policy": "lossguide",
    "min_child_weight": 20, "max_bin": 15, "nthread": 4, "seed": 42,
    "subsample": 1., "colsample_bytree": 1., "colsample_bylevel": 1.,
    "colsample_bynode": 1., "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 8192, "disable_default_eval_metric": 1,
    "max_cached_hist_node": 31}
ROUNDS = 64
FEATURE_TYPES = ["c" if name in ("category", "subcategory") else "q" for name in core.FEATURES]


class Model:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, features, num_threads=4):
        if num_threads != 4:
            raise ValueError("Fixed four-thread predictor required")
        return self.booster.inplace_predict(features).astype(np.float64)

    @classmethod
    def load(cls, path):
        return cls(xgb.Booster(params={"nthread": 4}, model_file=path))


class CachedIterator(xgb.DataIter):
    def __init__(self, features, labels, sizes, cache_prefix, deadline):
        self.features, self.labels, self.sizes = features, labels, sizes
        self.offsets = np.r_[np.int64(0), np.cumsum(sizes, dtype=np.int64)]
        if (features.shape != (int(self.offsets[-1]), len(core.FEATURES))
                or labels.shape != (len(features),) or np.any(sizes < 2)
                or np.any(sizes > 8192)):
            raise ValueError("Complete aligned candidate/query caches required")
        self.cursor, self.deadline, self.passes = 0, deadline, []
        self.label_digest, self.positives = hashlib.sha256(), 0
        super().__init__(cache_prefix=str(cache_prefix), release_data=True)

    def reset(self):
        if self.cursor not in (0, len(self.sizes)):
            raise ValueError("Incomplete external-memory population pass")
        self.cursor, self.positives, self.label_digest = 0, 0, hashlib.sha256()

    def next(self, input_data):
        core.limits(self.deadline)
        if self.cursor == len(self.sizes):
            return False
        end = self.cursor
        while (end < len(self.sizes) and end - self.cursor < 4096
               and self.offsets[end + 1] - self.offsets[self.cursor] <= 131072):
            end += 1
        start_row, end_row = map(int, self.offsets[[self.cursor, end]])
        data = np.asarray(self.features[start_row:end_row])
        target = np.asarray(self.labels[start_row:end_row], dtype=np.float32)
        input_data(data=data, label=target, group=self.sizes[self.cursor:end],
                   feature_names=list(core.FEATURES), feature_types=FEATURE_TYPES)
        self.label_digest.update(target.astype(np.uint8).tobytes())
        self.positives += int(target.sum(dtype=np.float64))
        self.cursor = end
        if end == len(self.sizes):
            self.passes.append({"impressions": len(self.sizes), "exposures": end_row,
                                "positives": self.positives, "labels_sha256": self.label_digest.hexdigest()})
        core.evict_pages([self.features, self.labels])
        return True


def build_cache(source, metadata, statistics, groups, directory, deadline):
    sequence = core.FeatureSequence(source, metadata, statistics, groups, True, deadline)
    features = np.memmap(Path(directory) / "features.float32", mode="w+", dtype=np.float32,
                         shape=(len(sequence), len(core.FEATURES)))
    labels = np.memmap(Path(directory) / "labels.uint8", mode="w+", dtype=np.uint8, shape=(len(sequence),))
    cursor, positives, label_digest = 0, 0, hashlib.sha256()
    start, batches = 0, 0
    while start < len(groups):
        stop, count = start, 0
        while (stop < len(groups) and stop - start < 256
               and count + sequence.sizes[stop] <= 8192):
            count += int(sequence.sizes[stop])
            stop += 1
        rows, queries = source.rows(groups[start:stop])
        end = cursor + len(rows)
        features[cursor:end] = sequence.features(queries, source.candidates[rows])
        target = source.labels[rows]
        labels[cursor:end] = target
        label_digest.update(target.tobytes())
        positives += int(target.sum(dtype=np.uint64))
        cursor = end
        if batches % 32 == 0:
            features.flush()
            labels.flush()
            core.evict_pages([*source.arrays, features, labels])
            if batches % 512 == 0:
                print(json.dumps({"stage": "caching_all_tree_features", "impressions": start,
                                  "exposures": cursor}), flush=True)
        start, batches = stop, batches + 1
    features.flush()
    labels.flush()
    core.evict_pages([*source.arrays, features, labels])
    if cursor != len(sequence):
        raise ValueError("Feature cache omitted source candidates")
    coverage = {"impressions": len(groups), "exposures": cursor, "positives": positives,
                "labels_sha256": label_digest.hexdigest()}
    return sequence, features, labels, coverage


class CutoffCallback(xgb.callback.TrainingCallback):
    def __init__(self, deadline):
        self.deadline = deadline

    def after_iteration(self, model, epoch, evals_log):
        core.limits(self.deadline)
        if epoch % 8 == 0:
            print(json.dumps({"stage": "external_all_pairs_tree_fit", "iteration": epoch + 1}), flush=True)
        return False


def fit(source, metadata, groups, directory, deadline):
    if xgb.__version__ != "3.1.1" or Path(directory).exists():
        raise ValueError("Pinned CPU XGBoost3.1.1 and fresh destination required")
    if shutil.disk_usage(tempfile.gettempdir()).free < 20000000000:
        raise OSError("At least20GB local temporary disk required before external fitting")
    directory = Path(directory)
    directory.mkdir()
    statistics = core.Statistics.fit(source, metadata, groups, deadline)
    with tempfile.TemporaryDirectory(prefix="jev-external-tree-") as cache:
        sequence, features, labels, coverage = build_cache(
            source, metadata, statistics, groups, cache, deadline)
        feature_hash = core.digest(Path(cache) / "features.float32")
        iterator = CachedIterator(features, labels, sequence.sizes, Path(cache) / "native", deadline)
        dataset = xgb.ExtMemQuantileDMatrix(iterator, max_bin=15, nthread=4, enable_categorical=True)
        native_groups = dataset.get_uint_info("group_ptr")
        if (dataset.num_row() != coverage["exposures"] or not iterator.passes
                or any(item != coverage for item in iterator.passes)
                or not np.array_equal(native_groups, sequence.offsets)):
            raise ValueError("Native external dataset changed targets, queries or coverage")
        core.evict_pages([*source.arrays, features, labels])
        gc.collect()
        booster = xgb.train(PARAMETERS, dataset, num_boost_round=ROUNDS,
                            callbacks=[CutoffCallback(deadline)])
        if booster.num_boosted_rounds() != ROUNDS:
            raise ValueError("Incomplete fixed-round external tree recipe")
        booster.save_model(directory / "model.ubj")
        native_passes = iterator.passes
        del dataset, iterator
        gc.collect()
        features._mmap.close()
        labels._mmap.close()
        del features, labels
    np.savez_compressed(directory / "statistics.npz", pooled=statistics.pooled, first=statistics.first)
    mask = np.zeros(len(source.offsets) - 1, dtype=bool)
    mask[groups] = True
    core.write_json(directory / "fit_coverage.json", {
        **coverage, "rounds": ROUNDS, "every_candidate_streamed_once_per_pass": True,
        "source_positions_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
        "feature_cache_sha256": feature_hash, "native_passes": native_passes,
        "training_candidate_sampling": "none", "pair_sampling": "none_all_query_positions_topk8192",
        "bin_cutpoints_use_feature_values_only": True})
    core.write_json(directory / "model_config.json", {
        "xgboost": xgb.__version__, "parameters": PARAMETERS, "rounds": ROUNDS, "features": core.FEATURES,
        "feature_types": FEATURE_TYPES, "priors": "entire_opposite_impression_fold_only",
        "age_source": "fitting_exposures_only", "source_coverage": coverage,
        "native_prediction_dtype": "float32", "evaluation_ranking_dtype": "float64"})
    return Model(booster), statistics


if __name__ == "__main__":
    core.main(fitter=fit, parameters=PARAMETERS)
