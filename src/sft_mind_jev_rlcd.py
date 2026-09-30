"""
RLCD: RL from Contrastive Distillation for JEV-4B on MIND noul format.

Trains JEV-4B to make the P(A)=P(yes) score for a *clicked* candidate
(preferred) exceed that of an *unclicked* candidate (dispreferred) **within the
same impression**. Preferences are ground truth (clicked > unclicked); the
contrastive margin is the model's own P(A) gap (self-distillation), so no
external teacher is needed.

Loss (contrastive hinge on the single A/B answer logit, beta-weighted to reuse
the published KL coefficient for scale stability):

    D = s(P(A)_pos) - s(P(A)_neg)            (usually log-prob margin)
    L = max(0, margin - D)                   (hinge / contrastive margin loss)

Optionally add beta * |D| smoothing via an in-file KL to keep training stable.
Default is a pure margin hinge (--loss hinge). With --loss kbce it instead
reuses a KL-style objective (see code).

Usage (multi-node, repo root):
    bash scripts/sft_jev_rlcd.sh --data_root ... --output_dir ...
or directly:
    deepspeed --hostfile=hostfile src/sft_mind_jev_rlcd.py --base_model ... \
        --train_behaviors_path ... --train_news_path ... --output_dir ... \
        --deepspeed_config ds_configs/ds_config_zero2.json
"""

import os
import sys
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    Trainer,
)
import fire

# Pre-init DDP with long timeout.
if "LOCAL_RANK" in os.environ and not dist.is_initialized():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jev_noul_pair_data import JEVNoulPairwiseDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PairwiseCollator:
    """Collate pos/neg halves independently (pad each side to its own max)."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _collate_side(self, features, key):
        import transformers
        feats = [
            {
                'input_ids': f[f'input_ids_{key}'],
                'labels': f[f'labels_{key}'],
                'attention_mask': f[f'attention_mask_{key}'],
            }
            for f in features
        ]
        return transformers.DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt', padding=True, label_pad_token_id=-100,
        )(feats)

    def __call__(self, features):
        return {
            'pos': self._collate_side(features, 'pos'),
            'neg': self._collate_side(features, 'neg'),
        }


class ContrastiveSelfDistillCollator:
    """Pads both members into a single batch so the trainer can compute the
    P(A) of each member in parallel (more efficient than two forwards)."""

    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        import transformers
        pos_feats = [self._side(f, 'pos') for f in features]
        neg_feats = [self._side(f, 'neg') for f in features]
        # Stack both members -> batch size 2B; the trainer separates grad.
        all_feats = pos_feats + neg_feats
        batch = transformers.DataCollatorForSeq2Seq(
            self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt', padding=True, label_pad_token_id=-100,
        )(all_feats)
        return batch


class JEVRLCDTrainer(Trainer):
    """Compute the A/B logit margin within an impression and apply a hinge
    contrastive loss (self-distillation, ground-truth preference)."""

    def __init__(self, margin=0.0, kl_beta=0.1, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.kl_beta = kl_beta
        self.tokenizer = tokenizer

    def _answer_logits(self, model, inputs):
        """(B,L,V) -> answer logits at the single label token (like sft_mind_jev)."""
        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=None,
            return_dict=True,
        ).logits
        # Answer = last real (non-pad) token index, from attention_mask.
        real_len = inputs['attention_mask'].sum(dim=1)
        last = real_len - 1
        predict_pos = last - 1
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        ans_logits = logits[batch_idx, predict_pos]  # (B, V)
        return ans_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs is a single collated batch of size 2B: first B = preferred,
        # second B = dispreferred (see ContrastiveSelfDistillCollator).
        b = inputs['input_ids'].shape[0] // 2
        pos_ids = {'input_ids': inputs['input_ids'][:b],
                   'attention_mask': inputs['attention_mask'][:b],
                   'labels': inputs['labels'][:b]}
        neg_ids = {'input_ids': inputs['input_ids'][b:],
                   'attention_mask': inputs['attention_mask'][b:],
                   'labels': inputs['labels'][b:]}

        pos_logits = self._answer_logits(model, pos_ids)   # (b, V)
        neg_logits = self._answer_logits(model, neg_ids)   # (b, V)

        # Ground-truth: preferred is 'A' (column for token "A"), dispref is 'B'.
        # Use the model's P(A) as the score; margin = logP(A)_pos - logP(A)_neg.
        lp_pos = F.log_softmax(pos_logits, dim=-1)
        lp_neg = F.log_softmax(neg_logits, dim=-1)
        a_id = pos_logits.new_tensor(self.tokenizer.convert_tokens_to_ids('A'), dtype=torch.long)
        b_id = neg_logits.new_tensor(self.tokenizer.convert_tokens_to_ids('B'), dtype=torch.long)

        s_pos = lp_pos.gather(1, a_id.unsqueeze(0).expand(b, 1)).squeeze(1)  # logP(A)_pos
        s_neg = lp_neg.gather(1, b_id.unsqueeze(0).expand(b, 1)).squeeze(1)  # logP(B)_neg

        # For a fair contrast, compare P(A)_pos vs P(A)_neg. Recompute P(A) for neg.
        s_neg_a = lp_neg.gather(1, a_id.unsqueeze(0).expand(b, 1)).squeeze(1)  # logP(A)_neg

        D = s_pos - s_neg_a  # margin of preferred over dispreferred on P(A)
        hinge = F.relu(self.margin - D)  # (b,)
        loss = hinge.mean()

        if return_outputs:
            return (loss, {'loss': loss, 'margin': D.mean().detach()})
        return loss


def resolve_report_to(wandb_project):
    if not wandb_project:
        return "none"
    return "wandb" if os.environ.get("MINIONEREC_ENABLE_WANDB", "") == "1" else "none"


def train(
    local_rank: int = -1,
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    use_abstract: bool = True,
    use_subcategory: bool = False,
    max_history: int = 30,
    neg_ratio: float = 1.0,
    hard_neg_ratio: float = 0.5,
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 64,
    micro_batch_size: int = 2,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2048,
    resume_from_checkpoint: str = None,
    wandb_project: str = "",
    wandb_run_name: str = "",
    deepspeed_config: str = "",
    early_stopping_patience: int = 3,
    margin: float = 0.0,
    kl_beta: float = 0.1,
):
    """Train JEV-4B on MIND noul pairwise contrastive (RLCD, self-distill)."""
    set_seed(seed)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // max(micro_batch_size * 2, 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        local_files_only=True if _local else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,
                                              local_files_only=True if _local else None)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    train_data = JEVNoulPairwiseDataset(
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
    )

    print(f"\nJEV-4B RLCD (contrastive self-distill) training:")
    print(f"  Pair samples: {len(train_data)}")
    print(f"  Max history: {max_history}, cutoff: {cutoff_len}")
    print(f"  Margin: {margin}, micro_batch: {micro_batch_size}")

    training_args_dict = {
        'per_device_train_batch_size': micro_batch_size * 2,  # pos+neg in one batch
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'gradient_checkpointing': True,
        'warmup_steps': 100,
        'num_train_epochs': num_epochs,
        'learning_rate': learning_rate,
        'bf16': True,
        'logging_steps': 10,
        'logging_first_step': True,
        'eval_strategy': 'no',
        'save_strategy': 'steps',
        'save_steps': 2048,
        'output_dir': output_dir,
        'save_total_limit': 2,
        'load_best_model_at_end': False,
        'ddp_find_unused_parameters': False if ddp else None,
        'report_to': resolve_report_to(wandb_project),
        'run_name': wandb_run_name if wandb_run_name else None,
        'disable_tqdm': False,
    }
    if deepspeed_config and os.path.exists(deepspeed_config):
        training_args_dict['deepspeed'] = deepspeed_config
        print(f"Using DeepSpeed config: {deepspeed_config}")

    training_args = transformers.TrainingArguments(**training_args_dict)

    collator = ContrastiveSelfDistillCollator(tokenizer, pad_to_multiple_of=8)

    trainer = JEVRLCDTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=collator,
        margin=margin,
        kl_beta=kl_beta,
        tokenizer=tokenizer,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_checkpoint_path = os.path.join(output_dir, 'final_checkpoint')
    trainer.save_model(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

    # Copy chat_template.jinja (JEV stores template there, rank0 only).
    if not ddp or dist.get_rank() == 0:
        if _local:
            src_jinja = os.path.join(base_model, 'chat_template.jinja')
            dst_jinja = os.path.join(final_checkpoint_path, 'chat_template.jinja')
            if os.path.exists(src_jinja) and not os.path.exists(dst_jinja):
                import shutil
                shutil.copy(src_jinja, dst_jinja)
                print('  Copied chat_template.jinja to final_checkpoint')

    print("\n✓ JEV-4B RLCD training completed!")
    print(f"  Model saved to: {final_checkpoint_path}")
    print(f"  Evaluate with: python src/evaluate_mind_jev.py --model_path {final_checkpoint_path} "
          f"--behaviors_path .../dev/behaviors.tsv --news_path .../dev/news.tsv")


if __name__ == '__main__':
    fire.Fire(train)
