import torch
import torch.nn as nn

from models.utils import SelfAttn, LinearAttn, GlobalAttn


class NRMS(nn.Module):
    def __init__(self, config, bert_model):
        super(NRMS, self).__init__()
        # self.word_emb = nn.Embedding(word2vec_embedding.shape[0], config['word_emb_dim'])
        # self.word_emb.weight = nn.Parameter(torch.tensor(word2vec_embedding, dtype=torch.float32))
        # self.word_emb.weight.requires_grad = True
        self.bert_model = bert_model
        self.news_dim = config['head_num'] * config['head_dim']
        self.user_dim = self.news_dim
        self.global_dim = self.news_dim
        self.key_dim = config['attention_hidden_dim']

        self.news_encoder = NewsEncoder(bert_model=self.bert_model,
                                        drop=config['dropout'],
                                        word_dim=config['word_emb_dim'],
                                        news_dim=self.news_dim,
                                        key_dim=self.key_dim,
                                        head_num=config['head_num'],
                                        head_dim=config['head_dim'])
        user_encoder = PGTEncoder if config['global'] else NewsSetEncoder
        self.user_encoder = user_encoder(news_encoder=self.news_encoder,
                                         drop=config['dropout'],
                                         news_dim=self.news_dim,
                                         user_dim=self.user_dim,
                                         global_dim=self.global_dim,
                                         key_dim=self.key_dim,
                                         head_num=config['head_num'],
                                         head_dim=config['head_dim'])

    def forward(self, x, source):
        if source == 'history':
            his_out = self.user_encoder(x)
            return his_out
        elif source == 'pgt':
            user_out = self.user_encoder(x[0], x[1])
            return user_out
        elif source == 'candidate':
            cand_out = self.news_encoder(x)
            return cand_out


class Encoder(nn.Module):
    def __init__(self, drop, input_dim, output_dim, key_dim, head_num, head_dim):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=drop)
        self.self_attn = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn = LinearAttn(output_dim=output_dim, key_dim=key_dim)

    def forward(self, **kwargs):
        pass


class NewsEncoder(Encoder):
    def __init__(self, bert_model, drop, word_dim, news_dim, key_dim, head_num, head_dim):
        super(NewsEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                          output_dim=news_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.bert_model = bert_model
        self.word_dim = word_dim

    def forward(self, x):
        # print("\nInput dims: ", x['input_ids'].shape)
        bs, hs, ts = x['input_ids'].shape
        x['input_ids'] = x['input_ids'].view(-1, ts)
        # print("\nInput dims: ", x['input_ids'].shape)
        x['attention_mask'] = x['attention_mask'].view(-1,ts)
        x = self.bert_model(**x)[0]
        x = x.view(bs, hs, ts, self.word_dim)
        # print(f"=============\nx {x.shape}\n======================")
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        out = self.linear_attn(out)
        return out


class NewsSetEncoder(Encoder):
    def __init__(self, news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim):
        super(NewsSetEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                             output_dim=user_dim, key_dim=key_dim,
                                             head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder

    def forward(self, x):
        x = self.news_encoder(x)
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        out = self.linear_attn(out)
        return out


class PGTEncoder(Encoder):
    def __init__(self, news_encoder, drop, news_dim, user_dim, global_dim, key_dim, head_num, head_dim):
        super(PGTEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                         output_dim=user_dim, key_dim=key_dim,
                                         head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder
        self.global_encoder = NewsSetEncoder(news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim)
        self.global_attn = GlobalAttn(user_dim, global_dim)

    def forward(self, his, global_pref):
        global_pref = self.global_encoder(global_pref)
        global_pref = self.dropout(global_pref)
        his = self.news_encoder(his)
        his = self.dropout(his)
        his = self.self_attn(QKV=(his, his, his))
        his = self.dropout(his)
        out = self.global_attn(his, global_pref)
        return out
