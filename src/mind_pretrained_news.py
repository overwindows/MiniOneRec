"""Frozen public word vectors, learned ordered-title CNN and history self-attention."""

from collections import Counter
import re
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TITLE_LENGTH = 30
VOCABULARY_LIMIT = 32768
GLOVE_WORD_ROWS = 400000
TOKEN_PATTERN = re.compile(r"[\w]+|[.,!?;|]")


def tokenize(text):
    if not isinstance(text, str):
        raise ValueError("News titles must be text")
    return TOKEN_PATTERN.findall(text.lower())


def title_vocabulary(corpus, news):
    counts = Counter()
    for name in corpus.names:
        if corpus._metadata_hash(news[name]) != corpus.metadata_sha256[name]:
            raise ValueError("Frozen article metadata changed")
        counts.update(tokenize(news[name]["title"]))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:VOCABULARY_LIMIT]
    return ["<pad>", "<unk>"] + words


def encode_titles(corpus, news, vocabulary, names=None):
    if (len(vocabulary) < 3 or vocabulary[:2] != ["<pad>", "<unk>"]
            or len(set(vocabulary)) != len(vocabulary)):
        raise ValueError("A unique frozen title vocabulary with reserved padding/OOV is required")
    names = corpus.names if names is None else names
    lookup = {word: index for index, word in enumerate(vocabulary)}
    tokens = np.zeros((len(names) + 1, TITLE_LENGTH), dtype=np.int32)
    for row, name in enumerate(names, 1):
        if corpus._metadata_hash(news[name]) != corpus.metadata_sha256[name]:
            raise ValueError("Frozen article metadata changed")
        encoded = [lookup.get(word, 1) for word in tokenize(news[name]["title"])[:TITLE_LENGTH]]
        tokens[row, :len(encoded)] = encoded
    return tokens


def load_public_glove(archive, vocabulary):
    lookup = {word.encode("utf-8"): index for index, word in enumerate(vocabulary[2:], 2)}
    embeddings = np.zeros((len(vocabulary), 300), dtype=np.float32)
    matched = np.zeros(len(vocabulary), dtype=bool)
    rows = 0
    reserved_unknown_rows = 0
    with zipfile.ZipFile(archive) as source:
        member = "glove.6B.300d.txt"
        if (source.namelist().count(member) != 1
                or source.getinfo(member).file_size > 2000000000
                or source.getinfo(member).flag_bits & 1):
            raise ValueError("Expected bounded unencrypted official 300-dimensional GloVe member")
        with source.open(member) as handle:
            for line in handle:
                word, separator, values = line.rstrip(b"\r\n").partition(b" ")
                if not separator or not word:
                    raise ValueError("Malformed public embedding row")
                rows += 1
                if word == b"<unk>":
                    if rows != GLOVE_WORD_ROWS + 1 or reserved_unknown_rows:
                        raise ValueError("Public reserved unknown vector must be the single trailing row")
                    vector = np.asarray(values.split(), dtype=np.float32)
                    if vector.shape != (300,) or not np.isfinite(vector).all():
                        raise ValueError("Invalid public reserved unknown vector")
                    reserved_unknown_rows += 1
                    continue
                if rows > GLOVE_WORD_ROWS:
                    raise ValueError("Unexpected extra public word vector")
                if word not in lookup:
                    continue
                index = lookup[word]
                fields = values.split()
                if len(fields) != 300:
                    raise ValueError("Public embedding dimensionality changed")
                vector = np.asarray(fields, dtype=np.float32)
                if vector.shape != (300,) or not np.isfinite(vector).all() or matched[index]:
                    raise ValueError("Invalid/duplicate matched public word vector")
                embeddings[index] = vector
                matched[index] = True
    if rows - reserved_unknown_rows != GLOVE_WORD_ROWS or not matched[2:].any():
        raise ValueError("Expected complete 400000-word public GloVe 6B corpus with vocabulary overlap")
    unknown = embeddings[matched].mean(axis=0, dtype=np.float64).astype(np.float32)
    embeddings[1:] = np.where(matched[1:, None], embeddings[1:], unknown)
    return embeddings, {
        "public_corpus_rows": rows, "public_word_rows": rows - reserved_unknown_rows,
        "reserved_unknown_rows": reserved_unknown_rows, "vocabulary_words": len(vocabulary) - 2,
        "matched_words": int(matched.sum()), "embedding_dimensions": 300,
        "unmatched_initialization": "mean_of_matched_public_vectors",
        "news_click_labels_read": False, "archive_member_crc_checked_by_complete_stream": True}


def masked_pool(vectors, mask, scorer):
    logits = scorer(vectors).squeeze(-1).masked_fill(~mask, -torch.inf)
    logits = torch.where(mask.any(dim=1, keepdim=True), logits, torch.zeros_like(logits))
    attention = torch.softmax(logits, dim=1) * mask
    return (vectors * attention[:, :, None]).sum(dim=1)


class PretrainedNewsRanker(nn.Module):
    def __init__(self, corpus, tokens, embeddings, vocabulary, priors):
        super().__init__()
        if (tokens.dtype != np.int32 or tokens.shape != (len(corpus.names) + 1, TITLE_LENGTH)
                or np.any(tokens < 0) or np.any(tokens >= len(vocabulary)) or np.any(tokens[0])
                or embeddings.dtype != np.float32 or embeddings.shape != (len(vocabulary), 300)
                or not np.isfinite(embeddings).all() or np.any(embeddings[0])
                or vocabulary[:2] != ["<pad>", "<unk>"] or len(set(vocabulary)) != len(vocabulary)
                or priors.pooled.shape != (len(corpus.names) + 1, 3)
                or not np.isfinite(priors.pooled).all()):
            raise ValueError("Exact finite public title embeddings and zero padding required")
        self.corpus, self.vocabulary, self.priors = corpus, vocabulary, priors
        self.prior_fold = None
        self.register_buffer("word_vectors", torch.from_numpy(embeddings), persistent=False)
        self.register_buffer("tokens", torch.from_numpy(tokens.astype(np.int64)), persistent=False)
        self.register_buffer("categories", torch.from_numpy(corpus.category), persistent=False)
        self.register_buffer("subcategories", torch.from_numpy(corpus.subcategory), persistent=False)
        self.register_buffer("pooled_logits", torch.from_numpy(priors.pooled[:, 0]), persistent=False)
        self.register_buffer("fold_logits", torch.from_numpy(np.stack(
            [table[:, 0] for table in priors.tables])) if hasattr(priors, "tables") else None, persistent=False)
        self.dropout = nn.Dropout(.2)
        self.word_projection = nn.Linear(300, 32, bias=False)
        self.convolution = nn.Conv1d(32, 64, 3, padding=1)
        self.news_attention = nn.Sequential(nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        self.category = nn.Embedding(len(corpus.categories) + 1, 16, padding_idx=0)
        self.subcategory = nn.Embedding(len(corpus.subcategories) + 1, 16, padding_idx=0)
        for embedding in (self.category, self.subcategory):
            nn.init.normal_(embedding.weight, std=.05)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.news_projection = nn.Linear(96, 64, bias=False)
        self.positions = nn.Parameter(torch.zeros(30, 64))
        self.history_attention = nn.MultiheadAttention(64, 4, dropout=.2, batch_first=True)
        self.user_attention = nn.Sequential(nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, 1, bias=False))
        self.head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.mode = "pretrained_title_cnn_history_attention"
        self.cached = None

    @property
    def features(self):
        return self.word_vectors

    def train(self, mode=True):
        if mode:
            self.cached = None
        return super().train(mode)

    def encode(self, news):
        tokens = self.tokens[news]
        projected = self.dropout(F.embedding(
            tokens, self.word_projection(self.word_vectors), padding_idx=0))
        convolved = F.relu(self.convolution(projected.transpose(1, 2))).transpose(1, 2)
        title = masked_pool(self.dropout(convolved), tokens != 0, self.news_attention)
        vectors = self.news_projection(torch.cat((
            title, self.category(self.categories[news]), self.subcategory(self.subcategories[news])), dim=1))
        return F.normalize(vectors, dim=1)

    @torch.no_grad()
    def prepare_inference(self):
        if self.training:
            raise ValueError("Evaluation-only article cache; never freeze encoder updates during training")
        self.cached = torch.cat([
            self.encode(torch.arange(start, min(start + 512, len(self.tokens))))
            for start in range(0, len(self.tokens), 512)])

    def forward(self, histories, candidates, groups):
        if (histories.ndim != 2 or histories.shape[1] != 30 or candidates.ndim != 1
                or groups.shape != candidates.shape
                or any(value.dtype != torch.long for value in (histories, candidates, groups))
                or torch.any(histories < 0) or torch.any(histories >= len(self.tokens))
                or torch.any(candidates < 1) or torch.any(candidates >= len(self.tokens))
                or torch.any(groups < 0) or torch.any(groups >= len(histories))):
            raise ValueError("Complete aligned history/candidate identities required")
        if self.training and (self.prior_fold not in (0, 1) or self.fold_logits is None):
            raise ValueError("Training requires an explicit source fold and opposite-fold click priors")
        if self.cached is None or self.training:
            unique, inverse = torch.unique(torch.cat((histories.reshape(-1), candidates)), return_inverse=True)
            vectors = self.encode(unique)
            history = vectors[inverse[:histories.numel()]].reshape(*histories.shape, 64)
            candidate = vectors[inverse[histories.numel():]]
        else:
            history, candidate = self.cached[histories], self.cached[candidates]
        mask = histories != 0
        safe = mask.clone()
        safe[~mask.any(dim=1), -1] = True
        positioned = history + self.positions[None]
        contextual, _ = self.history_attention(positioned, positioned, positioned,
                                                key_padding_mask=~safe, need_weights=False)
        user = F.normalize(masked_pool(contextual, mask, self.user_attention), dim=1)
        context = user[groups]
        residual = self.head(torch.cat((candidate, context, candidate * context), dim=1)).squeeze(-1)
        prior = (self.fold_logits[1 - self.prior_fold, candidates] if self.training
                 else self.pooled_logits[candidates])
        scores = prior + residual
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Pretrained title/history scores must remain finite")
        return scores

    @torch.no_grad()
    def scores(self, histories, candidates, groups):
        return self(torch.from_numpy(histories.astype(np.int64)),
                    torch.from_numpy(candidates.astype(np.int64)),
                    torch.from_numpy(groups.astype(np.int64))).numpy()

    def extend(self, news):
        if self.training:
            raise ValueError("Extend only frozen inference identities")
        old = len(self.corpus.names)
        self.corpus.extend(news)
        if len(self.corpus.names) > old:
            self.priors.extend(self.corpus)
            tokens = encode_titles(self.corpus, news, self.vocabulary, self.corpus.names[old:])
            self.tokens = torch.cat((self.tokens, torch.from_numpy(tokens[1:].astype(np.int64))))
            self.categories = torch.from_numpy(self.corpus.category)
            self.subcategories = torch.from_numpy(self.corpus.subcategory)
            self.pooled_logits = torch.from_numpy(self.priors.pooled[:, 0])
            self.cached = None
            self.prepare_inference()
