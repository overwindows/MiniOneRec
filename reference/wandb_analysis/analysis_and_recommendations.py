"""
MIND Experiment Results — Full Analysis & Next-Step Recommendations
===================================================================
Generated: 2026-03-30
Current Best: 70.49% AUC (L1.3: Abstract + MINDlarge + Qwen3-1.7B)
Target: 72.72% AUC (MIND Leaderboard SOTA)
Gap: ~2.23%

Data sources: wandb_mind_runs_wuchen.json, EXPERIMENT_PLAN.md
"""

# =============================================================================
# SECTION 1: WHAT WE LEARNED FROM ALL EXPERIMENTS
# =============================================================================
#
# 1.1 Learning Rate
#   - lr=2e-5 >> lr=3e-4 across ALL tasks (pointwise, ranking)
#   - lr=3e-4 runs showed clear overfitting; ranking runs at 3e-4 never converged well
#   - KEEP lr=2e-5 going forward
#
# 1.2 Epochs
#   - Pointwise: ep3-5 is sweet spot; ep7 shows eval loss rising (overfitting)
#   - Exception: with large data + abstract, ep7 may still help (L1.2 got 0.6932 AUC)
#   - Ranking: ep3 sufficient at lr=2e-5; ep8 at lr=3e-4 just memorizes
#
# 1.3 Data Size
#   - MINDlarge consistently beats MINDsmall (tight eval loss, less overfit)
#   - L1.3 (large+abstract) = 70.49% vs P1.B (small) = 68.61% → +1.88%
#
# 1.4 Abstract Feature
#   - HUGE win on large data: L1.3 (abstract) = 70.49% vs L1.7 (no abstract) = 69.46% → +1.03%
#   - On small data: P1.3 was worse (66.62%) but cutoff was too small (2048→4096 re-run pending)
#
# 1.5 Model Size
#   - Qwen3-1.7B and 1.7B-Base are best. 8B consistently underperforms + crashes
#   - 4B models (Instruct, Base, Reranker) all worse than 1.7B
#   - Hypothesis: smaller models generalize better on this task size
#   - 1.7B-Base slightly edges 1.7B (69.46% vs 69.32% on large)
#
# 1.6 Negative Ratio
#   - neg_ratio=2.0 is best; neg_ratio=3.0 hurts on both small and large
#
# 1.7 History Length
#   - max_history=30 is optimal; hist=50 slightly hurts (more noise)
#
# 1.8 Training Stability
#   - gradient norm ~1-2 for lr=2e-5 runs (healthy)
#   - 1.7B-Base on large has grad_norm ~8-10 (still converges well)
#   - Most crashes are Azure VM preemption, not code bugs
#
# 1.9 RL
#   - Small improvement: 69.39% → 69.69% (+0.30%) on MINDsmall
#   - RL acc@1 improved from 0.199 → 0.250
#   - Not yet tried on the new best model (L1.3)
#
# 1.10 Multi-task
#   - M4.0 (pw_ratio=0.7): terrible 62.13% AUC — multi-task hurts badly
#   - Ranking SFT gives much better MRR/nDCG but lower AUC
#
# =============================================================================
# SECTION 2: EXPERIMENTS THAT NEED RE-RUN
# =============================================================================
#
# Priority  | Exp ID | Why Re-run                              | Expected Impact
# ----------+--------+-----------------------------------------+----------------
# HIGH      | L1.8   | Abstract + EP=7 on large (just launched)| Beat L1.3 (70.49%)
# HIGH      | L1.9   | Abstract + Base model on large          | Combines two winners
# HIGH      | R2.L   | RL on top of L1.3 (best SFT)           | +0.3-0.5% from RL
# MEDIUM    | E3.A   | Ensemble L1.3 + L1.7                    | +0.5-1.0% from diversity
# MEDIUM    | P1.3r  | Abstract on small with cutoff=4096      | Verify abstract helps small
# LOW       | L1.6   | 8B on large (needs ZeRO-3 or LoRA)     | Probably won't help
#
# =============================================================================
# SECTION 3: TOP PRIORITY NEXT STEPS (ordered by expected ROI)
# =============================================================================
#
# STEP 1: Finish L1.8 + L1.9 (already in plan)
#   - L1.8: Abstract + EP=7 on large → already launched
#   - L1.9: Abstract + 1.7B-Base on large → submit next
#   - Expected: 70.5-71.0% AUC
#
# STEP 2: RL on best SFT checkpoint
#   - Take winner of L1.8/L1.9
#   - Run RL_mind with mind_auc reward on MINDlarge
#   - Use lower KL (0.05 instead of 0.5) for more aggressive policy shift
#   - Expected: +0.3-0.5% → 71.0-71.5%
#
# STEP 3: Ensemble (cheapest way to close the remaining gap)
#   - Ensemble top 2-3 diverse models:
#     * L1.3 (1.7B + abstract)
#     * L1.7 (1.7B-Base, no abstract)
#     * L1.8/L1.9 (new best)
#   - Weighted averaging of pointwise scores
#   - Expected: +0.5-1.0% → 71.5-72.0%
#
# STEP 4: Ranking + Pointwise cascade
#   - Train ranking model on MINDlarge with lr=2e-5, ep3, neg8, hist50
#   - Cascade: pointwise filters top-K → ranking re-ranks
#   - Expected: +0.3-0.5% on top of ensemble
#
# =============================================================================
# SECTION 4: DO YOU NEED TO CHANGE MODEL ARCHITECTURE?
# =============================================================================
#
# SHORT ANSWER: NO — not now.
#
# REASONING:
# 1. Your LLM-based approach (Qwen3-1.7B as scorer) is already competitive:
#    - 70.49% AUC vs 72.72% SOTA = 2.23% gap
#    - NRMS baseline = 67.76%, you're already +2.73% above it
#
# 2. The biggest gains so far came from DATA and TRAINING, not architecture:
#    - Abstract feature: +1.03%
#    - Large data: +1.88%
#    - These are "free" gains with zero architecture change
#
# 3. Architecture changes to consider ONLY IF plateau after Step 4:
#
#    a) PROMPT ENGINEERING (zero arch change, high ROI):
#       - Add impression position info ("this article appeared in position X")
#       - Add timestamp/recency feature
#       - Structured user profile summary before history
#
#    b) TRAINING STRATEGY (zero arch change, medium ROI):
#       - Curriculum learning: easy negatives → hard negatives
#       - Contrastive loss instead of binary cross-entropy
#       - Temperature scaling on logits
#
#    c) LIGHTWEIGHT ARCH CHANGES (low risk, medium ROI):
#       - LoRA adapters on different layers for pointwise vs ranking
#       - Cross-attention between user history and candidate (instead of concat)
#       - Learnable [USER] / [CANDIDATE] special tokens
#
#    d) HEAVY ARCH CHANGES (high risk, uncertain ROI):
#       - Replace LLM with dedicated news encoder + user encoder (like NRMS)
#       - This would be a completely different system
#       - NOT recommended given your current trajectory
#
# =============================================================================
# SECTION 5: CONCRETE ACTION PLAN
# =============================================================================
#
# Week 1 (now):
#   [x] L1.8 launched (abstract + ep7 + large)
#   [ ] Submit L1.9 (abstract + 1.7B-Base + large)
#   [ ] After L1.8 finishes: evaluate, update EXPERIMENT_PLAN.md
#
# Week 2:
#   [ ] RL on best L1.x checkpoint (mind_auc reward, KL=0.05)
#   [ ] Start ensemble experiments E3.A, E3.B, E3.C
#   [ ] Train ranking model on MINDlarge (lr=2e-5, ep3, neg8, hist50)
#
# Week 3:
#   [ ] Cascade evaluation (pointwise top-15 → ranking re-rank)
#   [ ] Full ensemble: multi-PW + PW-RK cascade
#   [ ] If still <72%: try prompt engineering (position, timestamp, profile)
#
# Expected trajectory:
#   Current:  70.49% (L1.3)
#   After L1.8/L1.9 + RL:  ~71.0-71.5%
#   After ensemble:  ~71.5-72.0%
#   After cascade:  ~72.0-72.5%  → SOTA territory
