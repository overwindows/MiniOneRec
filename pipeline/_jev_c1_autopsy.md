# JEV-C1 Autopsy — why every C1 run collapsed, and the fix

**Date:** 2026-10-16
**Track:** JEV-4B decision model (task #17); gateway to RLCD (#14), RL/GRPO (#15), teacher OPD (#16).

## Symptom
Every C1 SFT run "completed" but produced ~chance dev AUC:
- C1 (R-Drop KL distill): dev AUC **0.4975**
- C1fix (plain CE, `enable_kl=0`): dev AUC **0.4959** (rescored from recovered predictions)

Both are worse than the *untrained* stock JEV baseline (0.6432). The model learned nothing.

## Root cause — the destructive in-band HF eval loop (NOT the KL, NOT the loss scale)
`src/sft_mind_jev.py` configured:
```python
"eval_strategy": "steps", "metric_for_best_model": "eval_loss",
"load_best_model_at_end": True,  + EarlyStoppingCallback
```
Two compounding misconceptions:

1. **"Diverging" train loss was a red herring.** Full-vocab next-token CE over a 30-history prompt is
   legitimately O(50-150) (JEV vocab ~150k+, loss accumulates over the whole generated suffix, not a
   2-class 0.7). Reading 12-150 as instability was wrong — the weights stay finite.

2. **The in-band eval emits `eval_loss=NaN` from the very first eval** (epoch 0.034, observed in every
   C1 log). Because `metric_for_best_model=eval_loss` + `EarlyStoppingCallback`, HF fires EarlyStopping
   on the NaN at ~0.14ep and kills the run before it learns.

## The fix (implemented)
`src/sft_mind_jev.py`:
- `eval_strategy="no"`, `load_best_model_at_end=False`, removed `EarlyStoppingCallback` callback and
  `eval_dataset`. Dev AUC is produced ONLY by the external scorer (`src/evaluate_mind_jev.py` via
  `run_jev_eval.py`) after training — the same design as every other MiniOneRec pipeline.
- Added a NaN `FloatingPointError` guard in `JEVWithinModelDistillTrainer.compute_loss` so a *real*
  divergence halts loudly instead of feeding NaN through optimization.

**IMPORTANT:** the AML component does `git clone github.com/overwindows/MiniOneRec.git && git checkout dev`
→ the fix MUST be pushed to the `dev` branch before resubmitting, or the node runs the old buggy code.

## The good news
C1fix's checkpoint WAS externally scored — `jev_eval_results/dev_jev_predictions.txt` (5000 MIND_small
dev impressions) was recovered from the datalake and rescored offline (see `score_jev_c1fix.py`) => 0.4959.
This both *confirms* the eval-path diagnosis and gives us a reusable offline AUC harness for validation.

## Next steps
1. Push the `sft_mind_jev.py` fix to `dev`.
2. Resubmit C1-clean (plain CE, `enable_kl=0`, no in-band eval) on MIND_small.
3. External-eval the resulting checkpoint for a real dev AUC.
4. Once a clean CE base exceeds stock JEV (~0.6432), revisit distillation / RLCD with a healthy base.
