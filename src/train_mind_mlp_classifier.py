"""
Train a frozen LLM + MLP classifier on MIND dataset.

Architecture:
  - Qwen3-1.7B backbone: fully frozen
  - MLPHead(hidden_size → mlp_hidden_dim → 1): only component that trains
  - Loss: binary cross-entropy with logits
  - Score at inference: sigmoid(MLP(last_token_hidden_state))

This is a novel paradigm for MIND: hidden-state classifier rather than
generative Yes/No token scoring.  No tokens are generated.

Usage:
    python src/train_mind_mlp_classifier.py \\
        --base_model Qwen/Qwen3-1.7B \\
        --train_behaviors_path data/MIND/train/behaviors.tsv \\
        --train_news_path      data/MIND/train/news.tsv \\
        --eval_behaviors_path  data/MIND/dev/behaviors.tsv \\
        --eval_news_path       data/MIND/dev/news.tsv \\
        --output_dir           output_dir/mlp_classifier \\
        --batch_size 64 --num_epochs 10 --learning_rate 1e-3
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
import fire

# Required so `import data` resolves to repo-root data.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import MINDPointwiseSFTDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# MLP head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Lightweight binary classifier on top of frozen LLM hidden states."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch,)"""
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class MINDMLPDataset(Dataset):
    """
    Wraps MINDPointwiseSFTDataset but returns tokenized prompt + scalar label
    instead of the masked causal-LM targets used for SFT.
    """

    def __init__(
        self,
        behaviors_path: str,
        news_path: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        max_history: int = 30,
        neg_ratio: float = 2.0,
        hard_neg_ratio: float = 0.5,
        use_abstract: bool = False,
        use_chat_template: bool = False,
        use_subcategory: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_chat_template = use_chat_template
        self.use_subcategory = use_subcategory

        # Reuse MINDPointwiseSFTDataset for its sampling logic
        # We only need .samples and ._build_pointwise_prompt
        self._sft = MINDPointwiseSFTDataset(
            behaviors_path=behaviors_path,
            news_path=news_path,
            tokenizer=tokenizer,
            max_len=max_len,
            sample=sample,
            seed=seed,
            max_history=max_history,
            neg_ratio=neg_ratio,
            hard_neg_ratio=hard_neg_ratio,
            use_abstract=use_abstract,
            use_chat_template=use_chat_template,
            use_subcategory=use_subcategory,
        )
        self.samples = self._sft.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Build the same prompt as the SFT script (without " Yes"/" No" target)
        if self.use_subcategory:
            prompt, _ = self._sft._build_pointwise_prompt_subcategory(
                s['history'], s['candidate'], s['label']
            )
        else:
            prompt, _ = self._sft._build_pointwise_prompt(
                s['history'], s['candidate'], s['label'],
                impression_timestamp=s.get('impression_timestamp', ''),
            )

        # Tokenize (no chat template wrapping needed here — backbone is frozen,
        # so the exact format matters less; raw text is fine and faster)
        ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )

        return {
            'input_ids':      torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.ones(len(ids), dtype=torch.long),
            'label':          torch.tensor(s['label'], dtype=torch.float),
        }


def _collate_left_pad(batch, pad_token_id: int):
    """Left-pad a batch so that position [-1] is always the last real token."""
    max_len = max(len(x['input_ids']) for x in batch)
    input_ids_list, attn_list, labels_list = [], [], []

    for x in batch:
        seq = x['input_ids']
        pad_len = max_len - len(seq)
        input_ids_list.append(
            torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long), seq])
        )
        attn_list.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), x['attention_mask']])
        )
        labels_list.append(x['label'])

    return {
        'input_ids':      torch.stack(input_ids_list),
        'attention_mask': torch.stack(attn_list),
        'labels':         torch.stack(labels_list),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    base_model: str = "",
    train_behaviors_path: str = "",
    train_news_path: str = "",
    eval_behaviors_path: str = "",
    eval_news_path: str = "",
    output_dir: str = "",
    # Data
    max_history: int = 30,
    neg_ratio: float = 2.0,
    hard_neg_ratio: float = 0.5,
    use_abstract: bool = False,
    use_subcategory: bool = False,
    use_chat_template: bool = None,   # None = auto-detect from model name
    cutoff_len: int = 2048,
    sample: int = -1,
    # Model
    mlp_hidden_dim: int = 256,
    # Training
    batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 3,
    seed: int = 42,
    # Misc
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    """Train frozen LLM backbone + MLP classifier head on MIND pointwise data."""

    if not base_model:
        raise ValueError("--base_model is required")
    if not train_behaviors_path or not train_news_path:
        raise ValueError("--train_behaviors_path and --train_news_path are required")
    if not output_dir:
        raise ValueError("--output_dir is required")

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Auto-detect chat template
    if use_chat_template is None:
        use_chat_template = "instruct" in base_model.lower() or "chat" in base_model.lower()
        print(f"Chat template: {'enabled (instruct model)' if use_chat_template else 'disabled'}")

    # WandB (optional)
    if wandb_project:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name or None)

    # ------------------------------------------------------------------
    # Load backbone (frozen)
    # ------------------------------------------------------------------
    from pathlib import Path as _Path
    _local = os.path.isabs(base_model) or os.path.isdir(base_model)
    _model_arg = _Path(base_model) if _local else base_model
    _local_kwargs = {"local_files_only": True} if _local else {}

    print(f"Loading backbone: {base_model}")
    backbone = AutoModelForCausalLM.from_pretrained(
        _model_arg,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        output_hidden_states=True,
        **_local_kwargs,
    )
    # Freeze everything
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        _model_arg, trust_remote_code=True, **_local_kwargs
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"   # critical: last real token at index [-1]

    hidden_size = backbone.config.hidden_size
    print(f"Backbone hidden size: {hidden_size}")
    print(f"Backbone parameters frozen: {sum(p.numel() for p in backbone.parameters()):,}")

    # ------------------------------------------------------------------
    # MLP head (the only trainable component)
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_head = MLPHead(hidden_size, mlp_hidden_dim).to(device)
    backbone = backbone.to(device)

    trainable = sum(p.numel() for p in mlp_head.parameters())
    print(f"MLP head parameters (trainable): {trainable:,}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    def make_dataset(behaviors_path, news_path, sample_limit):
        return MINDMLPDataset(
            behaviors_path=behaviors_path,
            news_path=news_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample_limit,
            seed=seed,
            max_history=max_history,
            neg_ratio=neg_ratio,
            hard_neg_ratio=hard_neg_ratio,
            use_abstract=use_abstract,
            use_chat_template=use_chat_template,
            use_subcategory=use_subcategory,
        )

    print("Building training dataset...")
    train_dataset = make_dataset(train_behaviors_path, train_news_path, sample)

    val_dataset = None
    if eval_behaviors_path and eval_news_path:
        val_sample = min(5000, len(train_dataset) // 10) if sample <= 0 else min(1000, sample // 10)
        print(f"Building val dataset (sample={val_sample})...")
        val_dataset = make_dataset(eval_behaviors_path, eval_news_path, val_sample)

    collate = lambda b: _collate_left_pad(b, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate,
    ) if val_dataset else None

    print(f"\nTrain samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples:   {len(val_dataset)}")

    # ------------------------------------------------------------------
    # Optimizer — only MLP parameters
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(mlp_head.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate / 10
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = 0.0
    no_improve = 0
    best_ckpt_path = os.path.join(output_dir, "best_mlp.pt")

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        mlp_head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            # Extract frozen hidden states (no grad through backbone)
            with torch.no_grad():
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            # Last token of last layer: (batch, hidden_size)
            last_hidden = outputs.hidden_states[-1][:, -1, :].float()

            logits = mlp_head(last_hidden)                          # (batch,)
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mlp_head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # ---- Validation ----
        if val_loader:
            mlp_head.eval()
            all_scores, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    outputs = backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    last_hidden = outputs.hidden_states[-1][:, -1, :].float()
                    logits = mlp_head(last_hidden)
                    scores = torch.sigmoid(logits).cpu().tolist()
                    all_scores.extend(scores)
                    all_labels.extend(batch['labels'].tolist())

            val_auc = roc_auc_score(all_labels, all_scores)
            print(f"Epoch {epoch:3d}/{num_epochs}  loss={avg_loss:.4f}  val_AUC={val_auc:.4f}")

            if wandb_project:
                import wandb
                wandb.log({"train_loss": avg_loss, "val_auc": val_auc, "epoch": epoch})

            # Early stopping & checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                no_improve = 0
                torch.save({
                    'mlp_state_dict':     mlp_head.state_dict(),
                    'hidden_size':        hidden_size,
                    'mlp_hidden_dim':     mlp_hidden_dim,
                    'epoch':              epoch,
                    'best_auc':           best_auc,
                    'base_model':         base_model,
                }, best_ckpt_path)
                print(f"  ✓ Saved best checkpoint (AUC={best_auc:.4f})")
            else:
                no_improve += 1
                print(f"  No improvement ({no_improve}/{early_stopping_patience})")
                if no_improve >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{num_epochs}  loss={avg_loss:.4f}")
            # Save every epoch if no validation
            torch.save({
                'mlp_state_dict': mlp_head.state_dict(),
                'hidden_size':    hidden_size,
                'mlp_hidden_dim': mlp_hidden_dim,
                'epoch':          epoch,
                'best_auc':       None,
                'base_model':     base_model,
            }, best_ckpt_path)

    print(f"\n✓ Training complete. Best val AUC: {best_auc:.4f}")
    print(f"  Checkpoint: {best_ckpt_path}")
    print(f"\nTo evaluate:")
    print(f"  bash scripts/eval_mind_mlp_classifier.sh {base_model} {best_ckpt_path} dev")


if __name__ == "__main__":
    fire.Fire(train)
