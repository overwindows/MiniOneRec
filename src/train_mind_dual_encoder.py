"""
Train a from-scratch contrastive dual-encoder on MIND (NRMS-style).

Architecture (maximally orthogonal to the LLM pointwise family):
  - NewsEncoder: learned token embedding + multi-head self-attention encoder
                 -> pooled news embedding (d_model)
  - UserEncoder: additive attention (NRMS-style) over the user's clicked-news
                 embeddings -> user embedding (d_model)
  - Score:       dot(user_emb, news_emb)  -> relevance
  - Loss:        InfoNCE / in-batch contrastive (clicked news = positive,
                 other in-batch news = negatives), plus a binary CE head.

The tokenizer is a simple word-level vocab built from the MIND news corpus, so
the whole model trains from scratch (no HF dependency, cheap, single-GPU).

Eval emits `imp_id<TAB>score1<TAB>score2...` (behaviors-aligned) so it folds
into ensemble_from_scores.py alongside the LLM pointwise members.

Usage:
    python src/train_mind_dual_encoder.py \
        --train_behaviors data_root/train/behaviors.tsv \
        --train_news      data_root/train/news.tsv \
        --eval_behaviors  data_root/dev/behaviors.tsv \
        --eval_news       data_root/dev/news.tsv \
        --output_dir      output_dir/dual_encoder \
        --use_abstract \
        --num_epochs 8 --batch_size 256 --max_history 30 --d_model 128
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Word-level tokenizer built from the MIND news corpus
# ---------------------------------------------------------------------------
PAD, UNK = 0, 1
SPECIALS = ["<pad>", "<unk>", "<eos>", "<bos>"]


class WordTokenizer:
    def __init__(self, min_freq: int = 3):
        self.vocab = {tok: i for i, tok in enumerate(SPECIALS)}
        self.min_freq = min_freq

    def fit(self, texts):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(self._tok(t))
        for w, c in cnt.items():
            if c >= self.min_freq and w not in self.vocab:
                self.vocab[w] = len(self.vocab)
        self.unk_id = self.vocab.get("<unk>", UNK)

    @staticmethod
    def _tok(text):
        # lowercase, split on non-alphanumeric
        return text.lower().split()

    def encode(self, text, max_len: int = 64):
        ids = [self.vocab.get(w, self.unk_id) for w in self._tok(text)[:max_len]]
        return ids

    @property
    def vocab_size(self):
        return len(self.vocab)

    def save(self, path):
        torch.save(self.vocab, path)

    @classmethod
    def load(cls, path):
        t = cls()
        t.vocab = torch.load(path)
        t.unk_id = t.vocab.get("<unk>", UNK)
        return t


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NewsEncoder(nn.Module):
    """Token emb + multi-head self-attention encoder -> pooled news vector."""

    def __init__(self, vocab_size, d_model=128, nhead=4, n_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_embed = nn.Embedding(512, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_model)
        self.emb_norm = math.sqrt(d_model)

    def forward(self, input_ids, mask):
        # input_ids: (B, L), mask: (B, L) 1=real
        B, L = input_ids.shape
        x = self.embed(input_ids) * self.emb_norm
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        key_pad = (mask == 0)  # True where padded
        x = self.encoder(x, src_key_padding_mask=key_pad)
        # mean-pool over real tokens
        maskf = mask.unsqueeze(-1).float()
        pooled = (x * maskf).sum(1) / maskf.sum(1).clamp(min=1.0)
        return self.proj(pooled)  # (B, d)


class UserEncoder(nn.Module):
    """NRMS-style additive attention over user's clicked news embeddings."""

    def __init__(self, d_model=128):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.add_attn = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())

    def forward(self, hist_emb):
        # hist_emb: (B, H, d)
        score = (self.add_attn(hist_emb) * self.query).sum(-1)  # (B, H)
        alpha = F.softmax(score, dim=-1)
        user = (hist_emb * alpha.unsqueeze(-1)).sum(1)  # (B, d)
        return user


class DualEncoder(nn.Module):
    def __init__(self, news_enc, user_enc, temperature=0.07, cls_dim=128):
        super().__init__()
        self.news_enc = news_enc
        self.user_enc = user_enc
        self.temperature = temperature
        self.cls = nn.Sequential(nn.Linear(2 * cls_dim, cls_dim), nn.ReLU(),
                                 nn.Linear(cls_dim, 1))
        self.cls_dim = cls_dim

    def news_embed(self, id_batch, mask_batch):
        return self.news_enc(id_batch, mask_batch)

    def user_embed(self, hist_ids, hist_masks, hist_lens):
        # hist_ids: (B, H, Lc), hist_masks: (B, H, Lc); hist_lens: (B,H)
        B, H, Lc = hist_ids.shape
        flat = hist_ids.reshape(B * H, Lc)
        fmask = hist_masks.reshape(B * H, Lc)
        # embed all rows (padded rows pool to a constant but are zeroed after)
        emb = self.news_enc(flat, fmask)  # (B*H, d)
        emb = emb.reshape(B, H, self.cls_dim)
        # zero out padding history rows
        emb = emb * (hist_lens > 0).unsqueeze(-1).float()
        return self.user_enc(emb)  # (B, d)

    def forward(self, hist_ids, hist_masks, hist_lens, pos_ids, pos_masks,
                neg_ids, neg_masks):
        user = self.user_embed(hist_ids, hist_masks, hist_lens)         # (B, d)
        pos = self.news_embed(pos_ids, pos_masks)                        # (B, d)
        neg = self.news_embed(neg_ids, neg_masks)                        # (B, d*)

        # binary CE head on [user; news] concat
        logits_p = self.cls(torch.cat([user, pos], -1))
        logits_n = self.cls(torch.cat([user, neg], -1))

        # InfoNCE: user vs in-batch positives
        logits = user @ pos.t() / self.temperature           # (B, B)
        labels = torch.arange(user.size(0), device=user.device)
        infonce = F.cross_entropy(logits, labels)

        bce = F.binary_cross_entropy_with_logits(
            torch.cat([logits_p, logits_n], 0),
            torch.cat([torch.ones_like(logits_p), torch.zeros_like(logits_n)], 0),
        )
        return infonce, bce

    def score(self, hist_ids, hist_masks, hist_lens, cand_ids, cand_masks):
        """Pointwise relevance: user_emb dot news_emb for each candidate."""
        user = self.user_embed(hist_ids, hist_masks, hist_lens)   # (B, d)
        # candidates: treat each candidate as its own batch row via per-row history
        cand = self.news_embed(cand_ids, cand_masks)             # (B, d)
        # repeat user per candidate
        return (user * cand).sum(-1)  # (B,)


# ---------------------------------------------------------------------------
# MIND dual-encoder dataset (in-batch contrastive)
# ---------------------------------------------------------------------------
class DualEncoderDataset:
    def __init__(self, behaviors_path, news_path, tokenizer, max_history=30,
                 max_news_len=64, use_abstract=False, sample=-1, seed=42,
                 per_pos_neg=4, history_as_neg=False):
        self.tok = tokenizer
        self.max_history = max_history
        self.max_news_len = max_news_len
        self.use_abstract = use_abstract
        self.per_pos_neg = per_pos_neg
        self.history_as_neg = history_as_neg
        self.seed = seed

        self.news = self._load_news(news_path)
        self._tok_news()
        self.samples = self._load_behaviors(behaviors_path, sample)

    def _load_news(self, news_path):
        news = {}
        with open(news_path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 4:
                    continue
                nid = p[0]
                cat = p[1] if len(p) > 1 else ""
                sub = p[2] if len(p) > 2 else ""
                title = p[3]
                abstract = p[4] if len(p) > 4 else ""
                text = f"{title} {abstract}" if (self.use_abstract and abstract) else title
                news[nid] = {"cat": cat, "sub": sub, "text": text}
        return news

    def _tok_news(self):
        self.news_tok = {}
        for nid, d in self.news.items():
            self.news_tok[nid] = self.tok.encode(d["text"], self.max_news_len), len(
                self.tok.encode(d["text"], self.max_news_len))

    def _load_behaviors(self, behaviors_path, sample_limit):
        samples = []
        with open(behaviors_path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) < 5:
                    continue
                imp = p[0]
                history = p[3].split()
                if self.max_history > 0:
                    history = history[-self.max_history:]
                hist = [h for h in history if h in self.news]
                impressions = p[4].split()
                pos, neg = [], []
                for it in impressions:
                    if "-" in it:
                        nid, lab = it.rsplit("-", 1)
                        lab = int(lab)
                    else:
                        nid, lab = it, 0
                    if nid not in self.news:
                        continue
                    (pos if lab else neg).append(nid)
                if not pos:
                    continue
                # sample negatives per positive
                for _ in range(self.per_pos_neg):
                    if neg:
                        neg_nid = random.Random(self.seed).choice(neg)
                    else:
                        neg_nid = random.Random(self.seed).choice(self.news_tok)
                    # also may add a history item as a hard negative
                    hard = ""
                    if self.history_as_neg and hist:
                        hard = random.Random(self.seed).choice(hist)
                    samples.append({
                        "imp": imp, "history": hist, "pos": pos, "neg": neg_nid,
                        "hard": hard,
                    })
        if sample_limit and sample_limit > 0:
            rng = random.Random(self.seed)
            samples = rng.sample(samples, min(sample_limit, len(samples)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def _pad(seqs, max_len, pad_val=PAD):
        out = np.zeros((len(seqs), max_len), dtype=np.int64) + pad_val
        mask = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, s in enumerate(seqs):
            s = s[:max_len]
            out[i, :len(s)] = s
            mask[i, :len(s)] = 1
        return out, mask


def collate_dual(batch, num_hist, news_tok):
    B = len(batch)
    max_news = 64
    hist_ids = np.zeros((B, num_hist, max_news), dtype=np.int64) + PAD
    hist_mask = np.zeros((B, num_hist, max_news), dtype=np.int64)
    hist_len = np.zeros((B, num_hist), dtype=np.int64)
    pos_ids = np.zeros((B, max_news), dtype=np.int64) + PAD
    pos_mask = np.zeros((B, max_news), dtype=np.int64)
    neg_ids = np.zeros((B, max_news), dtype=np.int64) + PAD
    neg_mask = np.zeros((B, max_news), dtype=np.int64)

    for bi, s in enumerate(batch):
        hist = s["history"]
        # pick up to num_hist most recent
        hist = hist[-num_hist:]
        for hi, nid in enumerate(hist):
            toks = news_tok[nid][0]
            hist_len[bi, hi] = len(toks)
            hist_ids[bi, hi, :len(toks)] = toks
            hist_mask[bi, hi, :len(toks)] = 1
        p, pm = DualEncoderDataset._pad([news_tok[s["pos"][0]][0]], max_news)
        pos_ids[bi] = p[0]; pos_mask[bi] = pm[0]
        n, nm = DualEncoderDataset._pad([news_tok[s["neg"]][0]], max_news)
        neg_ids[bi] = n[0]; neg_mask[bi] = nm[0]

    return {
        "hist_ids": torch.tensor(hist_ids),
        "hist_mask": torch.tensor(hist_mask),
        "hist_len": torch.tensor(hist_len),
        "pos_ids": torch.tensor(pos_ids),
        "pos_mask": torch.tensor(pos_mask),
        "neg_ids": torch.tensor(neg_ids),
        "neg_mask": torch.tensor(neg_mask),
    }


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_behaviors", required=True)
    ap.add_argument("--train_news", required=True)
    ap.add_argument("--eval_behaviors", default="")
    ap.add_argument("--eval_news", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--use_abstract", action="store_true")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_hist", type=int, default=20)
    ap.add_argument("--max_history", type=int, default=50)
    ap.add_argument("--max_news_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=-1)
    ap.add_argument("--per_pos_neg", type=int, default=2)
    ap.add_argument("--history_as_neg", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--infonce_weight", type=float, default=1.0)
    ap.add_argument("--bce_weight", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.07)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Build vocab from train news
    print("Building vocab from train news...")
    from collections import Counter
    all_texts = []
    with open(args.train_news, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                t = p[3]
                if args.use_abstract and len(p) >= 5:
                    t = f"{t} {p[4]}"
                all_texts.append(t)
    tok = WordTokenizer(min_freq=2)
    tok.fit(all_texts)
    tok.unk_id = tok.vocab["<unk>"]
    print(f"Vocab size: {tok.vocab_size}")
    tok.save(os.path.join(args.output_dir, "tokenizer.pt"))

    # 2) Datasets
    print("Loading train dataset...")
    train_ds = DualEncoderDataset(args.train_behaviors, args.train_news, tok,
                                  max_history=args.max_history,
                                  max_news_len=args.max_news_len,
                                  use_abstract=args.use_abstract,
                                  sample=args.sample, seed=args.seed,
                                  per_pos_neg=args.per_pos_neg,
                                  history_as_neg=args.history_as_neg)
    print(f"  {len(train_ds)} train samples")

    from torch.utils.data import DataLoader
    collate = lambda b: collate_dual(b, args.num_hist, train_ds.news_tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True, pin_memory=True)

    # 3) Model
    news_enc = NewsEncoder(tok.vocab_size, d_model=args.d_model)
    user_enc = UserEncoder(d_model=args.d_model)
    model = DualEncoder(news_enc, user_enc, temperature=args.temperature,
                        cls_dim=args.d_model).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_epochs * steps_per_epoch, eta_min=args.lr / 10)

    # 4) Train
    print(f"Training {sum(p.numel() for p in model.parameters()):,} params...")
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        tot_info = tot_bce = 0.0
        nb = 0
        for batch in train_loader:
            hist_ids = batch["hist_ids"].to(device)
            hist_mask = batch["hist_mask"].to(device)
            hist_len = batch["hist_len"].to(device)
            pos_ids = batch["pos_ids"].to(device)
            pos_mask = batch["pos_mask"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            neg_mask = batch["neg_mask"].to(device)

            info, bce = model(hist_ids, hist_mask, hist_len,
                              pos_ids, pos_mask, neg_ids, neg_mask)
            # Note: DualEncoder.forward doesn't use neg in InfoNCE; bce uses neg
            loss = args.infonce_weight * info + args.bce_weight * bce
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            tot_info += info.item(); tot_bce += bce.item(); nb += 1
            global_step += 1
            if nb % 100 == 0:
                print(f"  ep{epoch} step{nb}/{steps_per_epoch} "
                      f"info={tot_info/nb:.4f} bce={tot_bce/nb:.4f}", flush=True)

        print(f"Epoch {epoch} done in {time.time()-t0:.0f}s  "
              f"info={tot_info/max(nb,1):.4f} bce={tot_bce/max(nb,1):.4f}", flush=True)

        ckpt = {
            "news_enc": news_enc.state_dict(),
            "user_enc": user_enc.state_dict(),
            "d_model": args.d_model,
            "tokenizer_path": os.path.join(args.output_dir, "tokenizer.pt"),
            "use_abstract": args.use_abstract,
            "num_hist": args.num_hist,
            "max_news_len": args.max_news_len,
            "max_history": args.max_history,
            "epoch": epoch,
        }
        torch.save(ckpt, os.path.join(args.output_dir, f"dual_enc_ep{epoch}.pt"))
        torch.save(ckpt, os.path.join(args.output_dir, "best_dual.pt"))
        # also save state to a single-file checkpoint for eval
        print(f"  saved ep{epoch}", flush=True)

    print(f"\n[OK] Training complete. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
