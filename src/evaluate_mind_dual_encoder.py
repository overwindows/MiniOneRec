"""
Evaluate a trained MIND contrastive dual-encoder.

Scores each (user, candidate) pair by dot(user_emb, news_emb) and emits
`imp_id<TAB>score1<TAB>score2...` aligned to behaviors.tsv candidate order, so
results fold into ensemble_from_scores.py alongside LLM pointwise members.

All news embeddings are computed ONCE up front (346k news on large), then each
impression just runs user additive-attention on its history embeddings and dots
against candidate embeddings -> fast even for the 376k-impression dev split.

Usage:
    python src/evaluate_mind_dual_encoder.py \
        --checkpoint output_dir/dual_encoder/best_dual.pt \
        --behaviors_path data_root/dev/behaviors.tsv \
        --news_path      data_root/dev/news.tsv \
        --output_scores_file scores_dual.txt \
        --output_file         preds_dual.txt
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mind_dual_encoder as T  # noqa: E402
from mind_utils import auc_score, mrr_score, ndcg_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--behaviors_path", required=True)
    ap.add_argument("--news_path", required=True)
    ap.add_argument("--output_scores_file", default="")
    ap.add_argument("--output_file", default="")
    ap.add_argument("--max_impressions", type=int, default=0)
    ap.add_argument("--news_batch_size", type=int, default=512)
    ap.add_argument("--normalize_per_impression", type=int, default=1,
                    help="Per-impression min-max normalize scores to [0,1] so the "
                         "member contributes a clean ranking signal in the weighted-"
                         "sum ensemble (LLM pointwise members are unbounded log-odds).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    d_model = ckpt.get("d_model", 128)
    num_hist = ckpt.get("num_hist", 20)
    max_history = ckpt.get("max_history", 50)
    max_news_len = ckpt.get("max_news_len", 64)
    use_abstract = ckpt.get("use_abstract", False)

    # Tokenizer path stored in the ckpt may be an absolute scratch path from the
    # training node that no longer exists on the eval node. Fall back to a
    # tokenizer.pt sitting next to the checkpoint if the stored path is missing.
    tok_path = ckpt["tokenizer_path"]
    if not os.path.isfile(tok_path):
        alt = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "tokenizer.pt")
        if os.path.isfile(alt):
            print(f"Stored tokenizer_path missing ({tok_path}); using sibling {alt}")
            tok_path = alt
    tok = T.WordTokenizer.load(tok_path)
    news_enc = T.NewsEncoder(tok.vocab_size, d_model=d_model).to(device)
    user_enc = T.UserEncoder(d_model=d_model).to(device)
    news_enc.load_state_dict(ckpt["news_enc"])
    user_enc.load_state_dict(ckpt["user_enc"])
    news_enc.eval(); user_enc.eval()
    print(f"Loaded dual-encoder  d_model={d_model}  vocab={tok.vocab_size}")

    # ---- load & pre-embed news ----
    print(f"Loading + embedding news from {args.news_path}...")
    news = {}
    with open(args.news_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            nid = p[0]
            title = p[3]
            abstract = p[4] if len(p) > 4 else ""
            text = f"{title} {abstract}" if (use_abstract and abstract) else title
            news[nid] = tok.encode(text, max_news_len)

    news_ids = list(news.keys())
    news_vec = {}
    with torch.no_grad():
        for i in tqdm(range(0, len(news_ids), args.news_batch_size),
                      desc="news embed", unit="batch"):
            chunk = news_ids[i:i + args.news_batch_size]
            texts = [news[nid] for nid in chunk]
            ids = np.zeros((len(chunk), max_news_len), dtype=np.int64) + T.PAD
            mask = np.zeros((len(chunk), max_news_len), dtype=np.int64)
            for r, s in enumerate(texts):
                ids[r, :len(s)] = s
                mask[r, :len(s)] = 1
            ids_t = torch.tensor(ids, device=device)
            mask_t = torch.tensor(mask, device=device)
            emb = news_enc(ids_t, mask_t)  # (B, d)
            emb_cpu = emb.cpu().numpy()
            for j, nid in enumerate(chunk):
                news_vec[nid] = emb_cpu[j]
    print(f"Embedded {len(news_vec):,} news")

    # ---- eval loop ----
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
    predictions, raw_scores = [], []

    total_lines = None
    if not args.max_impressions:
        with open(args.behaviors_path, encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
    total_to_process = total_lines or args.max_impressions or None

    count = 0
    skipped = 0
    print("\nEvaluating dual-encoder...\n")

    user_enc = user_enc.to(device)
    with open(args.behaviors_path, encoding="utf-8") as f:
        pbar = tqdm(total=total_to_process, desc="Eval", unit="imp")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                skipped += 1
                continue
            impression_id = p[0]
            history_ids = p[3].split()
            imp_list = []
            for it in p[4].split():
                if "-" in it:
                    nid, lab = it.rsplit("-", 1)
                    imp_list.append((nid, int(lab)))
                else:
                    imp_list.append((it, 0))

            if max_history > 0:
                history_ids = history_ids[-max_history:]
            hist_vecs = [news_vec[nid] for nid in history_ids if nid in news_vec][-num_hist:]
            cand_vecs = [news_vec.get(nid) for nid, _ in imp_list]

            if not hist_vecs or any(v is None for v in cand_vecs):
                scores = [0.0] * len(imp_list)
            else:
                H = np.array(hist_vecs)          # (H, d)
                C = np.array(cand_vecs)          # (C, d)
                hist_t = torch.tensor(H, device=device).unsqueeze(0)  # (1,H,d)
                with torch.no_grad():
                    user = user_enc(hist_t).squeeze(0)  # (d,)
                    user = user.cpu().numpy()
                sims = C @ user                  # (C,)
                scores = sims.tolist()

            if args.normalize_per_impression and len(scores) > 1:
                lo, hi = min(scores), max(scores)
                if hi - lo > 1e-12:
                    scores = [(s - lo) / (hi - lo) for s in scores]
                else:
                    scores = [0.5] * len(scores)

            labels = [l for _, l in imp_list]
            if sum(labels) > 0:
                aucs.append(auc_score(labels, scores))
                mrrs.append(mrr_score(labels, scores))
                ndcg5s.append(ndcg_score(labels, scores, 5))
                ndcg10s.append(ndcg_score(labels, scores, 10))

            if args.output_file:
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                predictions.append((impression_id, [imp_list[i][0] for i in ranked]))
            if args.output_scores_file:
                raw_scores.append(impression_id + "\t" + "\t".join(f"{s:.6f}" for s in scores))

            count += 1
            pbar.update(1)
            pbar.set_postfix({
                'AUC': f'{np.mean(aucs) if aucs else 0:.4f}',
                'MRR': f'{np.mean(mrrs) if mrrs else 0:.4f}',
            })
        pbar.close()

    def _avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    print("\n" + "=" * 50)
    print("MIND Dual-Encoder Evaluation")
    print("=" * 50)
    print(f"Impressions: {count:,}  (skipped {skipped})")
    print(f"AUC:     {_avg(aucs):.4f}")
    print(f"MRR:     {_avg(mrrs):.4f}")
    print(f"nDCG@5:  {_avg(ndcg5s):.4f}")
    print(f"nDCG@10: {_avg(ndcg10s):.4f}")
    print("=" * 50)

    if args.output_scores_file and raw_scores:
        with open(args.output_scores_file, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_scores) + "\n")
        print(f"Scores saved: {args.output_scores_file}")
    if args.output_file and predictions:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for imp_id, r in predictions:
                f.write(f"{imp_id} {' '.join(r)}\n")
        print(f"Predictions saved: {args.output_file}")


if __name__ == "__main__":
    main()
