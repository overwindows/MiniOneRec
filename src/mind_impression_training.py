"""Whole-impression batching and exact epoch-consumption checks."""

import transformers

from mind_impression_data import ids_digest


class CompleteImpressionCollator:
    def __init__(self, candidate_collator):
        self.candidates = candidate_collator

    def __call__(self, features):
        if len(features) != 1:
            raise ValueError("Complete-impression training requires batch_size=1 impression")
        feature = features[0]
        if len(feature["candidates"]) != feature["expected_candidates"]:
            raise ValueError("Source impression candidates were dropped before collation")
        batch = self.candidates(feature["candidates"])
        batch["expected_candidates"] = feature["expected_candidates"]
        return batch


class CompleteEpochCoverage(transformers.TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        runner = self.trainer
        if runner._active_epoch:
            raise ValueError("Previous complete-source epoch did not finish verification")
        runner._active_epoch = True
        runner._consumed_ids = set()
        runner._exposures = runner._positives = 0
        runner._start_step = state.global_step

    def on_epoch_end(self, args, state, control, **kwargs):
        runner = self.trainer
        record = {
            "epoch": len(runner.epoch_coverage) + 1,
            "impressions": len(runner._consumed_ids),
            "exposures": runner._exposures, "positives": runner._positives,
            "impression_ids_sha256": ids_digest(runner._consumed_ids),
            "optimizer_steps": state.global_step - runner._start_step,
        }
        for key in ("impressions", "exposures", "positives", "impression_ids_sha256"):
            if record[key] != runner.expected_source[key]:
                raise ValueError(f"Training epoch does not exactly cover the source: {key}")
        runner.epoch_coverage.append(record)
        runner._active_epoch = False

    def on_train_end(self, args, state, control, **kwargs):
        if (self.trainer._active_epoch
                or len(self.trainer.epoch_coverage) != int(args.num_train_epochs)):
            raise ValueError("Complete-source fixed-epoch training ended without all epoch proofs")
