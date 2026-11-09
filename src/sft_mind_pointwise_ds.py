"""
Train a model on MIND dataset using Point-wise SFT with DeepSpeed support.

This script uses a point-wise approach where each (history, candidate) pair
is scored independently with a Yes/No classification.

Key differences from list-wise (sft_mind_ranking.py):
- Each candidate is evaluated independently
- Binary classification: "Is this article relevant? Yes/No"
- More training signal (every candidate gets a label)
- Shorter context per sample
- No position bias

Training format:
    Prompt: "User History: ... Candidate: [article] Is this relevant? Answer:"
    Target: " Yes" or " No"

Usage:
    deepspeed --hostfile=hostfile src/sft_mind_pointwise_ds.py \
        --base_model Qwen/Qwen3-1.7B \
        --train_behaviors_path ../data/MIND/train/behaviors.tsv \
        --train_news_path ../data/MIND/train/news.tsv \
        --eval_behaviors_path ../data/MIND/dev/behaviors.tsv \
        --eval_news_path ../data/MIND/dev/news.tsv \
        --output_dir output_dir/sft_mind_pointwise \
        --deepspeed_config ds_config_zero2.json
"""

import os
import sys
import random
import collections
import datetime
import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
import fire

# Pre-initialize process group with a 2-hour timeout before DeepSpeed does it.
# Default PyTorch pg_timeout is 30 min — too short for large checkpoint writes
# to slow network-mounted storage (e.g. AML datastore).
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(hours=2),
    )

# Secondary process groups (e.g. DeepSpeed ZeRO's parameter-parallel groups)
# are created via dist.new_group() WITHOUT a timeout argument, so they inherit
# torch's 10-minute NCCL default — NOT the 2h timeout set above. A transient
# rank stall (NFS hiccup, slow dataloader worker) longer than 10 min then trips
# the watchdog on that group and kills the whole job (exit 122; seen on histfull
# @ step 3958: PG 1 ALLREDUCE Timeout(ms)=600000 while PG 0 was healthy).
# Wrap new_group so every group gets the same 2h default.
if "LOCAL_RANK" in os.environ:
    _orig_new_group = dist.new_group

    def _new_group_default_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(hours=2))
        return _orig_new_group(*args, **kwargs)

    dist.new_group = _new_group_default_timeout

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import MINDPointwiseSFTDataset

def resolve_report_to(wandb_project):
    """Decide the HF `report_to` target for this run.

    WandB CANNOT be reached from AzureML/Singularity compute nodes: the node is
    network-isolated (`setup_multi_node.sh` logs "WandB login failed") and even
    with a valid key HF's WandbCallback -> wandb.init() crashes rank 0 with
    "TypeError: the JSON object must be str, bytes or bytearray, not NoneType"
    from wandb's query_with_timeout (server.py:39) reading a None flags field.
    Under DeepSpeed that kills rank 0, hangs the barrier, and the other ranks
    dump a misleading NCCL DistBackendError as a secondary symptom.

    So remote wandb is disabled by default (report_to="none"); checkpoints and
    eval still write to disk, so training results are unaffected. Set
    MINIONEREC_ENABLE_WANDB=1 to force real logging (e.g. on a node with
    outbound access).
    """
    if not wandb_project:
        return "none"
    if os.environ.get("MINIONEREC_ENABLE_WANDB", "") == "1":
        return "wandb"
    return "none"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def answer_token_len(tokenizer) -> int:
    """Number of tokens the Yes/No target occupies (1 for every backbone here).

    The dataset appends the target to the prompt and sets the labels to
    [-100]*len(prompt) + target_ids, so the answer always occupies the LAST
    positions of the (left-padded) sequence.
    """
    return max(
        len(tokenizer.encode(t, add_special_tokens=False))
        for t in (" Yes", " No")
    )


def yes_no_token_ids(tokenizer):
    """(yes_id, no_id) exactly as src/evaluate_mind_pointwise.py derives them.

    The eval scores a candidate as logP(" Yes") - logP(" No") at the last
    PROMPT position. The pairwise (Bradley-Terry) term must use the SAME two
    token ids and the SAME position, otherwise it optimises a different
    quantity than the metric is computed on.
    """
    yes = tokenizer.encode(" Yes", add_special_tokens=False)
    no = tokenizer.encode(" No", add_special_tokens=False)
    yes_id = yes[0] if len(yes) == 1 else yes[-1]
    no_id = no[0] if len(no) == 1 else no[-1]
    return yes_id, no_id


class MetaCollator:
    """Sequence collator that also carries the pairwise-loss metadata.

    The stock `DataCollatorForSeq2Seq` looks for a "label" key (singular) and
    would try to pad our metadata as if it were the training labels, so the
    metadata is stripped BEFORE the base collator runs and re-attached
    afterwards under keys the model forward never sees.

    The keys are "bt_impression" / "bt_label" rather than "impression_id" /
    "label" so that they cannot collide with anything the tokenizer or the
    Trainer inspects.
    """

    META_KEYS = ("impression_id", "label", "answer_len")

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        meta = None
        if features and "impression_id" in features[0]:
            meta = (
                [f.get("impression_id") for f in features],
                [int(f.get("label", 0)) for f in features],
            )
            features = [
                {k: v for k, v in f.items() if k not in self.META_KEYS}
                for f in features
            ]
        batch = self.base(features)
        if meta is not None:
            batch["bt_impression"], batch["bt_label"] = meta
        return batch


def _estimate_sample_len(sample) -> int:
    """chars/4 token proxy (the same estimate the length-grouped sampler uses)."""
    cand = sample.get("candidate", {}) or {}
    hist = sample.get("history", []) or []
    txt = cand.get("text", "") + " ".join(h.get("text", "") for h in hist)
    return len(txt) // 4


def pack_impressions(samples, batch_size, seed=42):
    """Group the samples of WHOLE impressions into fixed-size micro-batches.

    The Bradley-Terry term can only form a pair when a positive and a negative
    of the SAME impression land in the SAME micro-batch. Under the shuffled
    production sampler that happened in 0.25% of micro-batches (measured on the
    real MIND_small train split), i.e. the "within the same impression" pairing
    was really a random GLOBAL pos/neg pairing across unrelated impressions.

    Two properties are preserved so that the packed sampler is a clean A/B
    against the production length-grouped sampler:

    * every micro-batch holds EXACTLY `batch_size` samples (except the last),
      so the effective batch size and the number of optimizer steps are
      unchanged;
    * the buckets stay length-homogeneous: the impressions are processed in
      length order, so consecutive impressions have similar lengths and the
      padding waste does not change.

    An impression that does not fit into the free room of the current bucket is
    SPLIT, but its samples are ordered so that every piece of length >= 2
    still contains at least one positive AND one negative -- a split piece
    therefore still contributes a pair.

    Returns a list of buckets; each bucket is a list of dataset indices.
    """
    by_imp = collections.OrderedDict()
    for j, s in enumerate(samples):
        by_imp.setdefault(s["impression_id"], []).append(j)

    # --- order each impression's samples so that every prefix of length >= 2
    #     contains both a positive and a negative -----------------------------
    ordered = []
    for imp, idxs in by_imp.items():
        pos = [j for j in idxs if samples[j].get("label") == 1]
        neg = [j for j in idxs if samples[j].get("label") != 1]
        seq = []
        p_i = n_i = 0
        while p_i < len(pos) or n_i < len(neg):
            if p_i < len(pos):
                seq.append(pos[p_i]); p_i += 1
            if n_i < len(neg):
                seq.append(neg[n_i]); n_i += 1
        cmax = max((_estimate_sample_len(samples[j]) for j in seq), default=0)
        ordered.append((cmax, seq))

    # Process the impressions in length order so that neighbouring buckets are
    # length-homogeneous (the same property the length-grouped sampler buys).
    ordered.sort(key=lambda c: c[0])

    buckets = []
    cur = []
    for cmax, seq in ordered:
        i = 0
        while i < len(seq):
            room = batch_size - len(cur)
            if room <= 0:
                buckets.append(cur)
                cur = []
                room = batch_size
            take = seq[i:i + room]
            cur.extend(take)
            i += len(take)
    if cur:
        buckets.append(cur)

    random.Random(seed).shuffle(buckets)
    return buckets


class PointwiseTrainer(transformers.Trainer):
    """Trainer that scores only the answer positions.

    Every point-wise example is `prompt + " Yes"|" No"`, so the answer is a
    single token at the END of the sequence and exactly ONE logit row matters:
    the one at position L-2, which predicts the answer token at L-1.

    The default path asks the model for the full (B, L, V) logit tensor.
    `ForCausalLMLoss` then upcasts it to fp32 and flattens it, which for
    V=248320 / B=4 / L=2048 is a 4.1 GB bf16 tensor plus an 8.1 GB fp32 copy
    (and the same again for the backward pass), and the lm_head GEMM over all
    L positions is ~11% of the forward FLOPs (~25% of the step with backward).

    Passing `logits_to_keep=answer_len+1` makes the model compute only the
    last few logit rows. The loss is still computed by the MODEL's own
    `loss_function`, so the reduction (mean, or sum/num_items_in_batch) is
    exactly the one the default path used and the gradient scale is unchanged.

    The labels are re-shaped to match the sliced logits: a -100 slot is
    PREPENDED, so the model's internal label shift lines kept logit j up with
    answer label j (kept logit j predicts position L-keep-1+j+1).

    Falls back to the stock path when the backbone has no `logits_to_keep`
    parameter, when label smoothing is on, or when a custom compute_loss_func
    is installed -- in those cases the loss is computed from the full logits
    and slicing them would change the result.
    """

    def __init__(self, *args, answer_len: int = 0, pairwise_weight: float = 0.0,
                 yes_id: int = 0, no_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_len = int(answer_len)
        self._logits_to_keep = self.answer_len + 1 if self.answer_len > 0 else 0
        self._supports_logits_to_keep = self._check_logits_to_keep_support()
        self.pairwise_weight = float(pairwise_weight)
        self.yes_id = int(yes_id)
        self.no_id = int(no_id)
        self._bt_pairs_seen = 0
        self._bt_batches_seen = 0
        self._bt_batches_with_pair = 0
        # Running sums of the two loss COMPONENTS, so a run can report how
        # much of the total loss the BT term actually accounts for. Without
        # this, a mis-scaled lambda is invisible: the run looks healthy
        # (loss decreases, no NaN) while the BT term drives everything.
        #
        # Kept as detached TENSORS on the device (not Python floats) so that
        # accumulating them does not force a GPU->CPU sync on every
        # micro-batch; the sync happens once, at the final print.
        self._ce_sum = None
        self._bt_sum = None
        # Gradient diagnostics (opt-in via MINIONEREC_GRAD_DIAG=1). They exist
        # because DeepSpeed reports an overflow only as a boolean, so the log
        # cannot say WHICH gradient went non-finite. These counters answer it.
        self._grad_diag = {
            "micro_batches": 0,
            "logits_hook": 0,
            "logits_nonfinite": 0,
            "logits_max_abs": 0.0,
            "emb_hook": 0,
            "emb_nonfinite": 0,
            "emb_max_abs": 0.0,
            "loss_nonfinite": 0,
            "ce_nonfinite": 0,
            "bt_nonfinite": 0,
            "param_hooks": 0,
            "param_nonfinite": 0,
            "param_max_abs": 0.0,
            "param_max_abs_name": "-",
            "first_bad_param": "",
            "first_bad_rank": -1,
            "first_bad_count": 0,
            "first_bad_total": 0,
        }
        self._grad_diag_enabled = os.environ.get("MINIONEREC_GRAD_DIAG", "") == "1"
        self._emb_probe_handle = None
        self._emb_probe_done = False

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _install_emb_probe(self):
        """Hook the embedding OUTPUT, to see whether the gradient is finite
        DEEP in the network (as opposed to at the logits).

        A finite gradient at the logits but a non-finite one at the embedding
        means the overflow happens inside the network's own backward (a bf16
        activation overflow), not in the loss. A non-finite gradient AT the
        logits means the loss itself produced it.
        """
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            base = getattr(unwrapped, "get_input_embeddings", lambda: None)()
            if base is None:
                return
            emb = base

            def _hook(module, args, output):
                if not isinstance(output, torch.Tensor) or not output.requires_grad:
                    return
                h = output

                def _tgrad(g):
                    d = self._grad_diag
                    d["emb_hook"] += 1
                    finite = torch.isfinite(g)
                    if not bool(finite.all()):
                        d["emb_nonfinite"] += 1
                    else:
                        m = float(g.detach().abs().max().item())
                        if m > d["emb_max_abs"]:
                            d["emb_max_abs"] = m
                h.register_hook(_tgrad)
            emb.register_forward_hook(_hook)
        except Exception as e:
            print(f"  [grad-diag] could not install the embedding probe: {e}")

    def _logits_grad_hook(self, g):
        d = self._grad_diag
        d["logits_hook"] += 1
        if not bool(torch.isfinite(g).all()):
            d["logits_nonfinite"] += 1
        else:
            m = float(g.detach().abs().max().item())
            if m > d["logits_max_abs"]:
                d["logits_max_abs"] = m
        return g

    def _install_param_grad_probe(self):
        """Hook EVERY parameter's gradient accumulation to find WHICH parameter
        goes non-finite, and on WHICH rank.

        This is the decisive diagnostic for the H200 overflow. DeepSpeed
        detects the overflow AFTER the reduce-scatter, so its boolean cannot
        say whether the inf was already in a local gradient or was produced by
        the reduction itself. `register_post_accumulate_grad_hook` fires as
        soon as a parameter's gradient is accumulated during backward -- i.e.
        BEFORE DeepSpeed's reduction -- so it localises the bad parameter on
        the bad rank.
        """
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            base = getattr(unwrapped, "get_input_embeddings", lambda: None)()
            if base is not None:
                base = base.weight
            named = list(unwrapped.named_parameters())
            if not named:
                return
            import torch.distributed as _dist
            rank = _dist.get_rank() if (_dist.is_available() and _dist.is_initialized()) else 0
            self._grad_diag["rank"] = rank
            self._grad_diag["params_hooked"] = len(named)

            def _make_hook(pname):
                def _hook(param):
                    g = param.grad
                    if g is None:
                        return
                    d = self._grad_diag
                    d["param_hooks"] += 1
                    if not bool(torch.isfinite(g).all()):
                        d["param_nonfinite"] += 1
                        if not d["first_bad_param"]:
                            d["first_bad_param"] = f"{pname}"
                            d["first_bad_rank"] = rank
                            bad = int((~torch.isfinite(g)).sum().item())
                            d["first_bad_count"] = bad
                            d["first_bad_total"] = int(g.numel())
                            print(f"  [grad-diag] NON-FINITE gradient on rank {rank}: "
                                  f"param '{pname}' has {bad}/{g.numel()} non-finite entries "
                                  f"(dtype={g.dtype}, shape={tuple(g.shape)})", flush=True)
                    else:
                        m = float(g.detach().abs().max().item())
                        if m > d["param_max_abs"]:
                            d["param_max_abs"] = m
                            d["param_max_abs_name"] = pname
                return _hook

            for pname, p in named:
                if not p.requires_grad:
                    continue
                try:
                    p.register_post_accumulate_grad_hook(_make_hook(pname))
                except Exception:
                    # Older torch without the hook API: fall back to a plain
                    # tensor hook on the parameter.
                    try:
                        p.register_hook(_make_hook(pname))
                    except Exception:
                        pass
            print(f"  [grad-diag] parameter gradient probe installed on {len(named)} params "
                  f"(rank {rank})", flush=True)
        except Exception as e:
            print(f"  [grad-diag] could not install the parameter probe: {e}")

    def _report_grad_diag(self):
        d = self._grad_diag
        print(f"  [grad-diag] rank={d.get('rank', 0)} micro-batches={d['micro_batches']} "
              f"loss_nonfinite={d['loss_nonfinite']} ce_nonfinite={d['ce_nonfinite']} "
              f"bt_nonfinite={d['bt_nonfinite']}")
        print(f"  [grad-diag] LOGITS grad: hooks={d['logits_hook']} "
              f"nonfinite={d['logits_nonfinite']} max|g|={d['logits_max_abs']:.6g}")
        print(f"  [grad-diag] EMBED  grad: hooks={d['emb_hook']} "
              f"nonfinite={d['emb_nonfinite']} max|g|={d['emb_max_abs']:.6g}")
        print(f"  [grad-diag] PARAM  grad: hooks={d['param_hooks']} "
              f"nonfinite={d['param_nonfinite']} max|g|={d['param_max_abs']:.6g} "
              f"({d.get('param_max_abs_name', '-')})")
        if d.get("first_bad_param"):
            print(f"  [grad-diag] FIRST non-finite gradient: rank {d.get('first_bad_rank')} "
                  f"param '{d['first_bad_param']}' "
                  f"({d.get('first_bad_count')}/{d.get('first_bad_total')} entries)")

    def log(self, logs, start_time=None):
        # DeepSpeed SILENTLY skips an optimizer step when it detects inf/NaN in
        # the gradients, and HF then does not advance the LR scheduler
        # (transformers/trainer.py: `if not self.accelerator.optimizer_step_was_skipped`).
        # A run that skips most of its steps looks superficially healthy -- the
        # loss creeps down and no NaN appears in the log -- but the weights
        # barely move and the LR never warms up.
        #
        # This happened for real: an 8-GPU run skipped 53 of 60 steps, so the
        # scheduler advanced only 7 times, the peak LR was 14x too small, and
        # the loss stalled at 25.5 instead of reaching 1.0. The counter lives in
        # DeepSpeed's checkpoint (`skipped_steps`), but nothing printed it, so
        # the run looked like a slow learner rather than a broken one.
        #
        # Logging it puts the signal in trainer_state.json's log_history, where
        # it is visible next to the loss and the learning rate.
        if self.is_deepspeed_enabled and getattr(self, "deepspeed", None) is not None:
            try:
                logs["ds_skipped_steps"] = int(self.deepspeed.skipped_steps)
            except Exception:
                pass
        if self.lr_scheduler is not None:
            try:
                logs["sched_epoch"] = int(self.lr_scheduler.last_epoch)
            except Exception:
                pass
        if self._grad_diag_enabled:
            # Report the gradient diagnostics WITH the loss, so a run that dies
            # mid-training still leaves the evidence in the log.
            d = self._grad_diag
            logs["grad_diag_logits_nonfinite"] = d["logits_nonfinite"]
            logs["grad_diag_logits_maxabs"] = d["logits_max_abs"]
            logs["grad_diag_emb_nonfinite"] = d["emb_nonfinite"]
            logs["grad_diag_emb_maxabs"] = d["emb_max_abs"]
        super().log(logs, start_time=start_time)

    def _check_logits_to_keep_support(self) -> bool:
        if self._logits_to_keep <= 0:
            return False
        # Kill switch: MINIONEREC_NO_ANSWER_ONLY_LOGITS=1 forces the stock
        # full-logits path. The answer-only path is numerically exact (verified
        # against ForCausalLMLoss), but this keeps a way to A/B it -- and to
        # fall back instantly if a backbone ever computes the loss differently.
        if os.environ.get("MINIONEREC_NO_ANSWER_ONLY_LOGITS", "") == "1":
            return False
        try:
            import inspect
            unwrapped = self.accelerator.unwrap_model(self.model)
            # A PEFT wrapper's forward only declares **kwargs, so the parameter
            # has to be looked up on the wrapped base model.
            try:
                from transformers.trainer_utils import unwrap_peft_model
                unwrapped = unwrap_peft_model(unwrapped)
            except Exception:
                pass
            params = inspect.signature(unwrapped.forward).parameters
            return "logits_to_keep" in params
        except Exception:
            return False

    def _scale_loss_for_devices(self, loss, num_items_in_batch):
        """Reproduce the base Trainer's cross-device loss rescaling.

        `Trainer.compute_loss` multiplies the loss by the number of processes
        when `average_tokens_across_devices` is on and the model accepted the
        `num_items_in_batch` kwarg (which is the case for every causal LM
        here). The custom pairwise path must apply the SAME factor, otherwise
        the effective learning rate would silently change.
        """
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss_scale = self.accelerator.num_processes
            pc = getattr(self.accelerator, "parallelism_config", None)
            if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
                loss_scale //= pc.tp_size
            loss = loss * (loss_scale if self.args.n_gpu <= 1 else self.args.n_gpu)
        return loss

    def _pairwise_term(self, scores, labels, impressions):
        """Bradley-Terry term: -log(sigmoid(score_pos - score_neg)).

        Only WITHIN-impression pairs count. A positive and a negative of the
        same impression must both be present in this micro-batch; the
        impression-packed sampler is what makes that happen (measured: 0.25%
        of micro-batches under the plain shuffled sampler, 99.8% with the
        packed one).

        Returns (mean_bt_loss, n_pairs) or (None, 0) when the micro-batch
        holds no within-impression pair.
        """
        imp2pos, imp2neg = {}, {}
        for i, (imp, lab) in enumerate(zip(impressions, labels)):
            if lab == 1:
                imp2pos.setdefault(imp, []).append(i)
            else:
                imp2neg.setdefault(imp, []).append(i)
        pos_idx, neg_idx = [], []
        for imp, ps in imp2pos.items():
            ns = imp2neg.get(imp)
            if not ns:
                continue
            k = min(len(ps), len(ns))
            pos_idx.extend(ps[:k])
            neg_idx.extend(ns[:k])
        if not pos_idx:
            return None, 0
        dev = scores.device
        p = scores[torch.tensor(pos_idx, device=dev)]
        n = scores[torch.tensor(neg_idx, device=dev)]
        return -torch.log(torch.sigmoid(p - n) + 1e-8).mean(), len(pos_idx)

    def _compute_loss_with_pairwise(self, model, inputs, return_outputs,
                                    num_items_in_batch, bt_impression, bt_label):
        """CE + lambda * Bradley-Terry, computed from ONE forward pass.

        The score is read from the logit row that PREDICTS the answer token
        (absolute position L-2). That is the same row the cross-entropy term
        trains (the sliced logits are [L-2, L-1] and the model's internal
        label shift lines L-2 up with the answer label), and it is the same
        quantity the evaluator scores (the last PROMPT hidden state). The
        earlier fork read row L-1 -- the answer-token row -- which the CE loss
        never trains and the evaluator never reads.

        NORMALISATION. The CE term produced by the model is already normalised
        by `num_items_in_batch`, so summing the per-micro-batch CE values over
        the window yields the window mean. The BT term is a per-micro-batch
        MEAN over pairs, so it must be divided by the number of micro-batches
        in the window to be on the same scale -- otherwise lambda would
        silently mean "lambda x gradient_accumulation_steps".

        The window is the GLOBAL one: `num_items_in_batch` is gathered across
        ranks under `average_tokens_across_devices` (the HF default), and the
        window holds `world_size * gas` micro-batches. So the BT divisor is
        `gas * world_size`. Using `gas` alone leaves the BT term `world_size`
        times too large relative to CE, which makes the effective lambda
        depend on how many GPUs the run uses. Measured on a real DeepSpeed
        ZeRO-2 engine: with `gas` alone the effective BT:CE weight doubles
        from 1 rank to 2 ranks; with `gas * world_size` it is bit-identical
        (src/probe_pairwise_grad_scale.py).
        """
        labels = inputs["labels"]
        keep = self.answer_len
        shifted = torch.cat(
            [labels.new_full((labels.shape[0], 1), -100), labels[:, -keep:]], dim=1
        )
        fwd = dict(inputs)
        fwd.pop("bt_impression", None)
        fwd.pop("bt_label", None)
        fwd["labels"] = shifted
        fwd["logits_to_keep"] = self._logits_to_keep
        if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
            fwd["num_items_in_batch"] = num_items_in_batch
        outputs = model(**fwd)

        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

        if self._grad_diag_enabled:
            d = self._grad_diag
            d["micro_batches"] += 1
            if not bool(torch.isfinite(ce_loss.detach())):
                d["ce_nonfinite"] += 1
            if isinstance(logits, torch.Tensor) and logits.requires_grad:
                logits.register_hook(self._logits_grad_hook)
            if not self._emb_probe_done:
                self._emb_probe_done = True
                self._install_emb_probe()

        # Which sliced row PREDICTS the answer? With left padding the answer
        # tokens occupy the last `a` positions of a row, so the row that
        # predicts the FIRST answer token is at absolute P-a-1, i.e. index
        # (answer_len - a) of the sliced logits. `a` is read from the labels
        # so that a multi-token answer (" Yes" splitting into 2 tokens for some
        # tokenizer) is handled exactly, not assumed to be 1. For the usual
        # single-token answer this is index 0 == absolute L-2 == the last
        # PROMPT position == exactly what the evaluator scores.
        with torch.no_grad():
            n_ans = (labels != -100).sum(dim=1)              # (B,)
            row_idx = (self.answer_len - n_ans).clamp(min=0)  # (B,)
        row_logits = logits[torch.arange(logits.shape[0], device=logits.device), row_idx]
        scores = (row_logits[:, self.yes_id] - row_logits[:, self.no_id]).float()

        bt_loss, n_pairs = self._pairwise_term(scores, bt_label, bt_impression)
        self._bt_pairs_seen += n_pairs
        self._bt_batches_seen += 1
        if n_pairs > 0:
            self._bt_batches_with_pair += 1
        if self._grad_diag_enabled and bt_loss is not None:
            if not bool(torch.isfinite(bt_loss.detach())):
                self._grad_diag["bt_nonfinite"] += 1

        loss = ce_loss
        if bt_loss is not None:
            # The CE term is normalised by `num_items_in_batch` and the BT term
            # is a per-micro-batch MEAN over pairs, so the BT term must be
            # divided by the number of micro-batches in the window to be on the
            # same scale -- otherwise lambda would silently mean
            # "lambda x gradient_accumulation_steps".
            #
            # WHICH window? Under `average_tokens_across_devices` (the HF
            # default) `num_items_in_batch` is the GLOBAL token count of the
            # window -- it is gathered across ranks -- so the CE term is a
            # GLOBAL window mean, while `gas` is the PER-DEVICE micro-batch
            # count. Dividing the BT term by `gas` alone therefore leaves it
            # W (= world_size) times too large relative to CE, and the
            # effective lambda would depend on how many GPUs the run uses.
            #
            # Measured on a real DeepSpeed ZeRO-2 engine: at W=2 the ratio
            # between this formula and the corrected one is exactly 0.5 = 1/W
            # (src/probe_pairwise_grad_scale.py). Hence the divisor carries
            # the world size.
            #
            # (When the model does NOT accept `num_items_in_batch`, the CE loss
            # is a plain per-micro-batch mean and `Trainer.training_step`
            # divides the WHOLE loss by the accumulation steps instead, so the
            # BT term must NOT be pre-divided.)
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                gas = max(1, int(getattr(self, "current_gradient_accumulation_steps", 1)))
                divisor = gas
                if self.args.average_tokens_across_devices:
                    # CE is a global mean here, so make BT a global mean too.
                    divisor *= max(1, int(self.accelerator.num_processes))
                bt_contrib = self.pairwise_weight * bt_loss / divisor
            else:
                bt_contrib = self.pairwise_weight * bt_loss
            loss = loss + bt_contrib
        else:
            bt_contrib = None
        # Accumulate the two loss COMPONENTS so the run can report the BT
        # term's SHARE of the total loss. Without this, a mis-scaled lambda is
        # invisible: the run looks healthy (loss decreases, no NaN) while the
        # BT term drives everything.
        #
        # Both are accumulated PRE-`_scale_loss_for_devices`, which multiplies
        # them by the same factor, so their ratio is the ratio of their
        # gradient contributions. A micro-batch with no pair contributes 0 to
        # the BT sum but its full CE to the CE sum.
        _ce_d = ce_loss.detach()
        _bt_d = (bt_contrib.detach() if bt_contrib is not None
                 else torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype))
        self._ce_sum = _ce_d if self._ce_sum is None else self._ce_sum + _ce_d
        self._bt_sum = _bt_d if self._bt_sum is None else self._bt_sum + _bt_d
        loss = self._scale_loss_for_devices(loss, num_items_in_batch)
        if self._grad_diag_enabled:
            if not bool(torch.isfinite(loss.detach())):
                self._grad_diag["loss_nonfinite"] += 1
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # The pairing metadata is never a model forward argument, so it is
        # always stripped before the model is called -- but it is captured
        # first, because the pairwise term needs it.
        bt_impression = inputs.get("bt_impression")
        bt_label = inputs.get("bt_label")
        has_meta = bt_impression is not None
        if has_meta:
            inputs = dict(inputs)
            inputs.pop("bt_impression", None)
            inputs.pop("bt_label", None)

        # The pairwise path is selected by the PRESENCE of the pairing metadata,
        # not by model.training: under DeepSpeed the model handed to
        # compute_loss is the Engine, whose `.training` flag is not toggled by
        # Engine.eval(), so it would be True during evaluation too. The
        # validation dataset is built WITHOUT the metadata, so the evaluation
        # batches never take this path.
        if self.pairwise_weight > 0 and has_meta and self._supports_logits_to_keep:
            return self._compute_loss_with_pairwise(
                model, inputs, return_outputs, num_items_in_batch,
                bt_impression=bt_impression, bt_label=bt_label,
            )
        if (
            self._supports_logits_to_keep
            and "labels" in inputs
            and inputs["labels"] is not None
            and self.label_smoother is None
            and self.compute_loss_func is None
            # With compute_metrics set the Trainer keeps the returned logits
            # (sliced ones would be misaligned with the labels), so fall back
            # to the stock full-logits path there.
            and self.compute_metrics is None
        ):
            labels = inputs["labels"]
            keep = self.answer_len
            shifted = torch.cat(
                [labels.new_full((labels.shape[0], 1), -100), labels[:, -keep:]], dim=1
            )
            inputs = dict(inputs)
            inputs["labels"] = shifted
            inputs["logits_to_keep"] = self._logits_to_keep
        loss = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        # NOTE: the base compute_loss already applies the cross-device
        # rescaling, so it must NOT be applied a second time here.
        return loss


def train(
    local_rank: int = -1,  # injected by DeepSpeed launcher; ignored here
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = False,
    use_subcategory: bool = False,
    max_history: int = 0,  # 0 = no limit
    neg_ratio: float = 1.0,  # Negatives per positive
    hard_neg_ratio: float = 0.5,  # Fraction of negatives from same category
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 2048,  # Shorter than list-wise since single candidate
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    train_from_scratch: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_run_id: str = "",
    deepspeed_config: str = "",
    early_stopping_patience: int = 3,
    use_chat_template: bool = None,  # Auto-detect if None
    use_recency: bool = False,  # Mark 5 most recent history items with "(recent)"
    use_profile_summary: bool = False,  # Prepend top-3 category interest summary
    use_impression_timestamp: bool = False,  # Add day/time-of-day context from impression timestamp
    use_entity: bool = False,  # Append Wikidata entity labels ("Topics: ...") to article text
    history_selection: str = "recent",  # "recent" | "relevant" | "dedup_backfill" | "dedup_only" | "catcap"
    history_format: str = "flat",  # "flat" | "grouped_profile" | "two_block" | "title_only"
    use_lora: bool = False,  # LoRA fine-tuning (for backbones too large for full FT)
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",  # comma-separated; fire may pass a tuple
    optimizer: str = "adamw",  # "adamw" | "muon" (Muon on hidden 2D matrices, AdamW on the rest)
    muon_lr: float = 0.02,    # Muon lr is in spectral-norm units, NOT comparable to AdamW lr
    pairwise_loss_weight: float = 0.0,  # lambda for the Bradley-Terry pairwise term (0 = off)
):
    """Train with point-wise SFT format (Yes/No classification) using DeepSpeed"""

    set_seed(seed)

    # Auto-detect if model is instruct variant (if use_chat_template not explicitly set)
    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()
        if use_chat_template:
            print(f"Auto-detected instruct model: will use chat template")
        else:
            print(f"Using raw text format (no chat template)")

    # Resume existing WandB run if run_id is provided
    if wandb_run_id:
        os.environ['WANDB_RUN_ID'] = wandb_run_id
        os.environ['WANDB_RESUME'] = 'allow'
        print(f"Resuming WandB run: {wandb_run_id}")

    if not base_model:
        raise ValueError("Please specify --base_model")

    # Normalize empty string to None so HF Trainer doesn't try to load a nonexistent path
    if not resume_from_checkpoint:
        resume_from_checkpoint = None

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load model — treat absolute paths and existing directories as local
    from pathlib import Path as _Path
    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    _base_model_arg = _Path(base_model) if _local else base_model
    _local_kwargs = {"local_files_only": True} if _local else {}

    # Attention implementation: use sdpa (PyTorch-native) across the board.
    # flash_attention_2 is nominally faster for dense causal LMs, but it is an
    # external wheel whose ABI can mismatch the node's torch (undefined symbol
    # `_ZN3c105ErrorC2...` at import) — seen node-variant on AzureML where
    # setup_multi_node.sh may install torch==2.6.0 vs a flash_attn built for a
    # different torch. sdpa has no such external dependency, dispatches both
    # dense and the hybrid Qwen3.5 GatedDeltaNet backbone (which FA2 cannot
    # handle anyway), so it is the robust default. Qwen3.5 checkpoints declare
    # multimodal architectures but AutoModelForCausalLM loads the pure-text
    # Qwen3_5ForCausalLM, which automatically ignores the vision tower weights.
    if train_from_scratch:
        config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        model = AutoModelForCausalLM.from_config(config)
    else:
        _config = AutoConfig.from_pretrained(_base_model_arg, **_local_kwargs)
        _arch = " ".join(_config.architectures or [])
        _hybrid_backbone = "ForConditionalGeneration" in _arch or "Qwen3_5" in _arch
        # The attention backend is overridable because the answer-only-logits
        # path was found to overflow on the H200 nodes (SM90) while the same
        # config trained fine on A100 (SM80) and B200 (SM100) -- the SDPA
        # kernel dispatch differs per SM architecture, so the backend is the
        # natural suspect. MINIONEREC_ATTN_IMPL=eager isolates it.
        _attn = os.environ.get("MINIONEREC_ATTN_IMPL", "sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            _base_model_arg,
            torch_dtype=torch.bfloat16,
            attn_implementation=_attn,
            **_local_kwargs,
        )
        print(f"  Attention implementation: {_attn}")
        if _hybrid_backbone:
            print(f"  Detected hybrid backbones ({_arch}): using sdpa attention (vision tower not loaded)")

    tokenizer = AutoTokenizer.from_pretrained(_base_model_arg, trust_remote_code=True, **_local_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Number of tokens the Yes/No answer occupies. The dataset appends the
    # target to the prompt and sets labels = [-100]*len(prompt) + target_ids,
    # so with LEFT padding the answer tokens are still the last ones.
    _answer_len = answer_token_len(tokenizer)

    # LoRA: wrap the loaded model with PEFT adapters. The base weights stay
    # frozen; only adapters train. trainer.save_model() on a PeftModel writes
    # adapter weights only (small); for eval the adapter must be merged or the
    # eval script must load base + adapter — we MERGE at save time so the
    # final_checkpoint is a plain CausalLM dir the existing eval pipeline can
    # consume unchanged.
    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[m.strip() for m in
                            (lora_target_modules.split(",") if isinstance(lora_target_modules, str)
                             else lora_target_modules) if m.strip()],
        )
        model = get_peft_model(model, lora_config)
        # Gradient checkpointing + frozen base: inputs to checkpointed blocks
        # no longer require grad, which silently zeroes the adapter gradients.
        model.enable_input_require_grads()
        model.print_trainable_parameters()

    # Load point-wise datasets
    train_data = MINDPointwiseSFTDataset(
        behaviors_path=train_behaviors_path,
        news_path=train_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
        use_entity=use_entity,
        history_selection=history_selection,
        history_format=history_format,
        # The pairing metadata is only needed when the Bradley-Terry term is
        # on; the validation set never carries it, so the eval batches cannot
        # accidentally take the pairwise path.
        return_meta=pairwise_loss_weight > 0,
    )

    val_data = MINDPointwiseSFTDataset(
        behaviors_path=eval_behaviors_path,
        news_path=eval_news_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=min(5000, len(train_data) // 10) if sample <= 0 else min(1000, sample // 10),
        seed=seed,
        max_history=max_history,
        neg_ratio=neg_ratio,
        hard_neg_ratio=hard_neg_ratio,
        use_abstract=use_abstract,
        use_subcategory=use_subcategory,
        use_chat_template=use_chat_template,
        use_recency=use_recency,
        use_profile_summary=use_profile_summary,
        use_impression_timestamp=use_impression_timestamp,
        use_entity=use_entity,
        history_selection=history_selection,
        history_format=history_format,
    )

    print(f"\nTraining with Point-wise SFT:")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Max history: {'unlimited' if max_history == 0 else max_history}")
    print(f"  Neg ratio: {neg_ratio}")
    print(f"  Cutoff length: {cutoff_len}")
    print(f"  Format: Yes/No classification")
    print(f"  Chat template: {'enabled' if use_chat_template else 'disabled (raw text)'}")

    # Prepare training arguments with optional DeepSpeed
    training_args_dict = {
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        # Long sequences (cutoff_len up to 8192) with micro_batch 4 produce very
        # large activation tensors; without checkpointing the forward pass OOMs
        # an 80GB A100 during the loss logits materialization (seen as CUDA OOM
        # in fixed_cross_entropy on the first step). Checkpointing trades a
        # little compute for a large drop in activation memory.
        # On large-VRAM GPUs (B200 180GB) checkpointing is a pure ~2x slowdown:
        # disable via MINIONEREC_NO_GRADIENT_CHECKPOINTING=1.
        "gradient_checkpointing": os.environ.get("MINIONEREC_NO_GRADIENT_CHECKPOINTING", "") != "1",
        "dataloader_num_workers": int(os.environ.get("MINIONEREC_NUM_WORKERS", "0")),
        "warmup_steps": 100,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "logging_first_step": True,
        "eval_strategy": "steps" if val_data else "no",
        "save_strategy": "steps",
        "eval_steps": int(os.environ.get("MINIONEREC_EVAL_STEPS", "256")) if val_data else None,
        "save_steps": int(os.environ.get("MINIONEREC_SAVE_STEPS", "2048")),
        "output_dir": output_dir,
        "save_total_limit": 2,
        "load_best_model_at_end": True if val_data else False,
        "ddp_find_unused_parameters": False if ddp else None,
        # NOTE: `group_by_length` was removed from TrainingArguments in
        # transformers 5.x (it was deprecated in 4.5x). The repo default is
        # False (no length grouping), which is transformers 5.x default too,
        # so dropping it is behaviour-neutral here.
        "report_to": resolve_report_to(wandb_project),
        "run_name": wandb_run_name if wandb_run_name else None,
        "metric_for_best_model": "eval_loss" if val_data else None,
        "greater_is_better": False,
        "disable_tqdm": False,
    }

    # MINIONEREC_MAX_STEPS caps the number of optimizer steps. It exists for
    # smoke tests (does the custom loss path train at all?) and for cheap
    # throughput probes; a full run leaves it unset.
    _max_steps = os.environ.get("MINIONEREC_MAX_STEPS", "").strip()
    if _max_steps:
        training_args_dict["max_steps"] = int(_max_steps)

    if pairwise_loss_weight > 0:
        # The pairing metadata (impression_id / label) is not a model forward
        # argument, so the Trainer's RemoveColumnsCollator would strip it
        # before the collator sees it. Our own collator strips it instead.
        training_args_dict["remove_unused_columns"] = False

    # Add DeepSpeed config if provided
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict["deepspeed"] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    # Muon optimizer (opt-in): orthogonalized momentum updates on hidden 2D
    # weight matrices, AdamW on embeddings / lm_head / scalars (Muon's own
    # guidance — it is not designed for those). SingleDeviceMuonWithAuxAdam
    # keeps one optimizer object so the HF Trainer scheduler and DeepSpeed
    # integration work unchanged. Note Muon lr is in spectral-norm units;
    # 0.02-0.05 is the usual range regardless of model size.
    optimizer_obj = None
    if optimizer == "muon":
        if training_args_dict.get("deepspeed"):
            raise ValueError(
                "optimizer=muon is incompatible with DeepSpeed ZeRO: ZeRO hands the "
                "optimizer flat 1-D fp32 partitions, so Muon's Newton-Schulz step never "
                "sees the 2-D weight matrices (zeropower_via_newtonschulz5 asserts on the "
                "first step). Launch with torchrun and no --deepspeed_config to use Muon."
            )
        from muon import SingleDeviceMuonWithAuxAdam
        # NOTE: membership must be id-based — `p not in list` invokes tensor
        # __eq__ (elementwise), which crashes on shape mismatch.
        _hidden_ids = {
            id(p) for n, p in model.named_parameters()
            if p.ndim >= 2 and p.requires_grad
            and "embed" not in n and "lm_head" not in n
        }
        hidden_matrix_params = [
            p for _, p in model.named_parameters() if id(p) in _hidden_ids
        ]
        other_params = [
            p for _, p in model.named_parameters()
            if p.requires_grad and id(p) not in _hidden_ids
        ]
        param_groups = [
            dict(params=other_params, lr=learning_rate, betas=(0.9, 0.95),
                 eps=1e-10, weight_decay=0.0, use_muon=False),
            dict(params=hidden_matrix_params, lr=muon_lr, momentum=0.95,
                 weight_decay=0.0, use_muon=True),
        ]
        optimizer_obj = SingleDeviceMuonWithAuxAdam(param_groups)
        print(f"Muon optimizer: {len(hidden_matrix_params)} hidden matrix params, "
              f"{len(other_params)} aux params (AdamW lr={learning_rate}, Muon lr={muon_lr})")

    # Initialize trainer
    # Early stopping is gated on completing at least one full epoch: eval-loss
    # often dips and recovers within epoch 1 (LR warmup + data-order effects),
    # and stopping at epoch 0.3 wastes the run. The wrapper below defers the
    # HF EarlyStoppingCallback until trainer.state.epoch >= 1.0.
    class _EpochGatedEarlyStopping(transformers.TrainerCallback):
        def __init__(self, patience, min_epoch=1.0):
            self.inner = EarlyStoppingCallback(early_stopping_patience=patience)
            self.min_epoch = min_epoch
        def on_step_end(self, args, state, control, **kwargs):
            if state.epoch is not None and state.epoch >= self.min_epoch:
                return self.inner.on_step_end(args, state, control, **kwargs)
            return control
        def on_train_begin(self, args, state, control, **kwargs):
            return self.inner.on_train_begin(args, state, control, **kwargs)

    # The Yes/No token ids must match the evaluator's exactly, and the pairwise
    # score must be read from the row that PREDICTS the answer (absolute
    # position L-2) -- the same position the evaluator scores.
    _yes_id, _no_id = yes_no_token_ids(tokenizer)

    trainer = PointwiseTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        args=training_args,
        answer_len=_answer_len,
        pairwise_weight=pairwise_loss_weight,
        yes_id=_yes_id,
        no_id=_no_id,
        **({"optimizers": (optimizer_obj, None)} if optimizer_obj is not None else {}),
        data_collator=MetaCollator(transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )),
        callbacks=[_EpochGatedEarlyStopping(early_stopping_patience)] if val_data else None,
    )
    if trainer._supports_logits_to_keep:
        print(f"  Answer-only logits: ON (logits_to_keep={trainer._logits_to_keep}, "
              f"answer_len={_answer_len}) — skips the full (B,L,V) logit tensor")
    else:
        print(f"  Answer-only logits: OFF (backbone has no logits_to_keep / "
              f"label smoothing / custom loss) — using the stock full-logits path")

    # Length-grouped training batches (opt-in via MINIONEREC_LENGTH_GROUPED=1).
    # transformers 5.x removed `group_by_length`; without it every micro-batch pads
    # to its longest member (~2048) while the median prompt is ~1250 tokens — ~40%
    # wasted compute. This sampler restores the old LengthGroupedSampler behavior:
    # random megabatches sorted by length, so batches are homogeneous.
    #
    # The impression-packed sampler below REPLACES it (its buckets are already
    # length-homogeneous), so the two are mutually exclusive.
    _use_packed = os.environ.get("MINIONEREC_IMPRESSION_PACKED", "") == "1"
    if os.environ.get("MINIONEREC_LENGTH_GROUPED", "") == "1" and not _use_packed:
        from torch.utils.data import DataLoader

        def _length_grouped_dataloader(ds, batch_size, collate_fn, num_workers, seed, world_size, rank):
            # Token lengths are not precomputed (samples store raw history/candidate
            # dicts; tokenization happens in __getitem__), so estimate length from
            # the raw text: chars/4 is a robust token proxy for English news text.
            def _est_len(s):
                cand = s.get("candidate", {}) or {}
                hist = s.get("history", []) or []
                txt = cand.get("text", "") + " ".join(h.get("text", "") for h in hist)
                return len(txt) // 4
            try:
                lengths = [_est_len(s) for s in ds.samples]
            except Exception:
                return None
            import random as _random
            rng = _random.Random(seed)
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            # megabatches of 50*batch_size sorted internally by length
            mega = max(50 * batch_size, batch_size * world_size)
            grouped = []
            for i in range(0, len(indices), mega):
                chunk = sorted(indices[i:i + mega], key=lambda j: lengths[j])
                grouped.extend(chunk)
            # Contiguous micro-batches of the length-sorted order
            batches = [grouped[i:i + batch_size] for i in range(0, len(grouped), batch_size)]

            # CRITICAL: shard at the BATCH level across DDP ranks. The Trainer's
            # default dataloader installs a DistributedSampler; this custom one
            # must do its own sharding, otherwise every rank trains on the FULL
            # dataset (8x duplicated compute, 8x too many optimizer steps).
            class _GroupedBatchSampler(torch.utils.data.Sampler):
                def __init__(self, batches, world_size, rank, seed):
                    self.batches = batches
                    self.world_size = world_size
                    self.rank = rank
                    self.seed = seed
                    self.epoch = 0
                def set_epoch(self, epoch):
                    self.epoch = epoch
                def __iter__(self):
                    # reshuffle batch order each epoch (different batch composition)
                    rng2 = _random.Random(self.seed + 1000 * self.epoch)
                    self.epoch += 1
                    order = list(range(len(self.batches)))
                    rng2.shuffle(order)
                    # trim to an equal shard per rank so all ranks iterate the
                    # same number of batches (avoids DDP/DeepSpeed barrier hang)
                    n = (len(order) // self.world_size) * self.world_size
                    for b in order[:n][self.rank::self.world_size]:
                        yield self.batches[b]
                def __len__(self):
                    return len(self.batches) // self.world_size

            class _GroupedDS(torch.utils.data.Dataset):
                def __init__(self, ds):
                    self.ds = ds
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, i):
                    return self.ds[i]

            return DataLoader(
                _GroupedDS(ds),
                batch_sampler=_GroupedBatchSampler(batches, world_size, rank, seed),
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        _dl = _length_grouped_dataloader(
            train_data,
            training_args.per_device_train_batch_size,
            trainer.data_collator,
            training_args.dataloader_num_workers,
            seed,
            world_size,
            rank,
        )
        if _dl is not None:
            trainer.get_train_dataloader = lambda: _dl
            print(f"Length-grouped train dataloader enabled (world_size={world_size}, rank={rank})")

    # Impression-packed training batches (opt-in via
    # MINIONEREC_IMPRESSION_PACKED=1). This is the fix for the A7.3 pairwise
    # defect: the Bradley-Terry term can only pair a positive with a negative
    # of the SAME impression if both are in the same micro-batch, and under the
    # plain shuffled sampler that happened in 0.25% of micro-batches (measured
    # on the real MIND_small train split) -- so the "within-impression" pairing
    # was really a random global pos/neg pairing across unrelated impressions.
    #
    # It REPLACES the length-grouped sampler (the packed buckets are already
    # length-homogeneous), so the two are mutually exclusive.
    if os.environ.get("MINIONEREC_IMPRESSION_PACKED", "") == "1":
        from torch.utils.data import DataLoader

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        mb = training_args.per_device_train_batch_size
        try:
            buckets = pack_impressions(train_data.samples, mb, seed=seed)
        except Exception as e:
            print(f"WARNING: impression packing failed ({e}); falling back to the default sampler")
            buckets = None

        if buckets:
            # CRITICAL: shard at the BATCH level, never the sample level. A
            # bucket that was split across two ranks would put the positive and
            # the negative of an impression on DIFFERENT ranks, which is
            # exactly the failure the packed sampler exists to fix.
            class _PackedBatchSampler(torch.utils.data.Sampler):
                def __init__(self, buckets, world_size, rank, seed):
                    self.buckets = buckets
                    self.world_size = world_size
                    self.rank = rank
                    self.seed = seed
                    self.epoch = 0
                def set_epoch(self, epoch):
                    self.epoch = epoch
                def __iter__(self):
                    # reshuffle the bucket order each epoch
                    rng2 = random.Random(self.seed + 1000 * self.epoch)
                    self.epoch += 1
                    order = list(range(len(self.buckets)))
                    rng2.shuffle(order)
                    # trim to an equal shard per rank so all ranks iterate the
                    # same number of batches (avoids a DDP/DeepSpeed barrier hang)
                    n = (len(order) // self.world_size) * self.world_size
                    for b in order[:n][self.rank::self.world_size]:
                        yield self.buckets[b]
                def __len__(self):
                    return len(self.buckets) // self.world_size

            class _PackedDS(torch.utils.data.Dataset):
                def __init__(self, ds):
                    self.ds = ds
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, i):
                    return self.ds[i]

            _dl = DataLoader(
                _PackedDS(train_data),
                batch_sampler=_PackedBatchSampler(buckets, world_size, rank, seed),
                collate_fn=trainer.data_collator,
                num_workers=training_args.dataloader_num_workers,
                pin_memory=True,
            )
            trainer.get_train_dataloader = lambda: _dl
            # Report the pairing rate that the packed sampler actually achieves
            # on THIS run's batches -- the number the BT term depends on.
            _n_with_pair = 0
            for b in buckets:
                by = collections.defaultdict(lambda: [0, 0])
                for j in b:
                    s = train_data.samples[j]
                    by[s["impression_id"]][1 if s.get("label") == 1 else 0] += 1
                if any(p > 0 and n > 0 for p, n in by.values()):
                    _n_with_pair += 1
            print(f"Impression-packed train dataloader enabled: {len(buckets)} buckets "
                  f"({len(buckets) * mb} samples), {100.0 * _n_with_pair / max(1, len(buckets)):.2f}% "
                  f"of buckets contain a within-impression pos/neg pair "
                  f"(world_size={world_size}, rank={rank})")

    model.config.use_cache = False

    # Train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Report the DeepSpeed optimizer-step skips. A skipped step is one where
    # DeepSpeed found inf/NaN in the gradients and dropped the update; HF then
    # does not advance the LR scheduler either, so a run that skips most of
    # its steps never warms up and never learns. This is the signal that
    # distinguishes "slow learner" from "broken run".
    if trainer.is_deepspeed_enabled and getattr(trainer, "deepspeed", None) is not None:
        try:
            _skipped = int(trainer.deepspeed.skipped_steps)
            _applied = int(trainer.deepspeed.global_steps) - _skipped
            _total = int(trainer.deepspeed.global_steps)
            print(f"  DeepSpeed optimizer steps: {_applied} applied / {_skipped} SKIPPED "
                  f"(of {_total}) — {_skipped / max(1, _total) * 100.0:.1f}% skipped")
            if _skipped > 0.5 * max(1, _total):
                print("  *** WARNING: most optimizer steps were SKIPPED. The weights barely "
                      "moved and the LR scheduler never advanced past step "
                      f"{_applied}. This run's result is NOT a valid learning curve. ***")
        except Exception:
            pass

    if getattr(trainer, "_grad_diag_enabled", False):
        trainer._report_grad_diag()

    if trainer.pairwise_weight > 0:
        # The pairing rate is the number the whole BT term depends on: if it
        # is near zero the term is a no-op. Print it so every BT run states
        # its own effective pairing rate.
        #
        # The rate that matters is the fraction of MICRO-BATCHES that actually
        # contain a within-impression pair -- that is what decides whether the
        # BT term has any gradient at all. (`pairs/micro-batch` is reported
        # alongside it, but a high mean over a tiny fraction of batches would
        # still mean the term fires on almost nothing.)
        #
        # These counters are PER RANK: each rank sees its own shard of the
        # packed buckets. With the packed sampler every rank sees the same
        # rate, so the numbers are directly comparable across ranks.
        _b = max(1, trainer._bt_batches_seen)
        print(f"Pairwise (Bradley-Terry) term: {trainer._bt_pairs_seen} pairs over "
              f"{trainer._bt_batches_seen} micro-batches "
              f"({trainer._bt_pairs_seen / _b:.3f} pairs/micro-batch, "
              f"{100.0 * trainer._bt_batches_with_pair / _b:.2f}% of micro-batches "
              f"contained a within-impression pair) with lambda={trainer.pairwise_weight}")
        # The BT term's SHARE of the total loss. This is the number that says
        # whether lambda is in a sane range: if the BT share is ~100% the run
        # is training the pairwise objective, not the point-wise one, and the
        # CE term (the thing the leaderboard metric actually measures) is
        # being drowned out.
        if trainer._ce_sum is not None:
            _ce = float(trainer._ce_sum)
            _bt = float(trainer._bt_sum)
            _tot = abs(_ce) + abs(_bt)
            print(f"  Loss components (mean per micro-batch): "
                  f"CE={_ce / _b:.6f}  BT(weighted)={_bt / _b:.6f}  "
                  f"BT share={100.0 * abs(_bt) / _tot if _tot else 0.0:.2f}%")

    # Save final model (use trainer.save_model for proper DeepSpeed ZeRO handling)
    final_checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    if use_lora:
        # Merge adapters into the base weights so final_checkpoint is a plain
        # CausalLM directory loadable by the standard eval pipeline. Only rank 0
        # writes (all ranks writing the same dir on NFS corrupts shards). ZeRO-3
        # partitions params, so gather them before merging on that path.
        zero3 = trainer.is_deepspeed_enabled and \
            trainer.accelerator.state.deepspeed_plugin.zero_stage == 3
        if zero3:
            import deepspeed
            gather = deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=0)
        else:
            import contextlib
            gather = contextlib.nullcontext()
        with gather:
            if trainer.is_world_process_zero():
                merged = model.merge_and_unload()
                merged.save_pretrained(final_checkpoint_path)
                tokenizer.save_pretrained(final_checkpoint_path)
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(final_checkpoint_path)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(final_checkpoint_path)

    print(f"\n✓ Point-wise SFT training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_pointwise.sh {output_dir}/final_checkpoint dev")


if __name__ == "__main__":
    # Reject unknown flags BEFORE training starts. fire.Fire() calls train()
    # first and only errors on the unconsumed flag afterwards (exit 2), so an
    # unknown flag would otherwise train to completion with the default value
    # and only then fail -- invisibly inside a Slurm log. See src/cli_guard.py.
    from cli_guard import check_unknown_flags
    check_unknown_flags(train)
    fire.Fire(train)
