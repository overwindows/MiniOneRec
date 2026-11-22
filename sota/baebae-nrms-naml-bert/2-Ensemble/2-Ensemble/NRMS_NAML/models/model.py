######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/models/model.py
# - This file includes model classes
#
# Version: 1.0
#######################################################################################################

import torch
import torch.nn as nn

from NRMS_NAML.models.utils import SelfAttn, LinearAttn, GlobalAttn


class NRMS(nn.Module):
    """
    Model class
    """
    def __init__(self, config, word2vec_embedding, num_cat, num_subcat):
        """
        Initialize the class.
        :param config: configuration dictionary
        :param word2vec_embedding: word2vec embedding array
        :param num_cat: number of categories
        :param num_subcat: number of sub-categories
        """
        super(NRMS, self).__init__()
        self.word_emb = nn.Embedding(word2vec_embedding.shape[0], config['word_emb_dim'])
        self.word_emb.weight = nn.Parameter(torch.tensor(word2vec_embedding, dtype=torch.float32))
        self.word_emb.weight.requires_grad = True
        self.cat_dim = config['cat_emb_dim']
        self.news_dim = config['head_num'] * config['head_dim']
        self.user_dim = self.news_dim
        self.global_dim = self.news_dim
        self.key_dim = config['attention_hidden_dim']
        self.aux = config['aux']
        self.num_cat = num_cat
        self.num_subcat = num_subcat

        if self.aux:
            self.news_encoder = AggEncoder(word_emb=self.word_emb,
                                           drop=config['dropout'],
                                           word_dim=config['word_emb_dim'],
                                           news_dim=self.news_dim,
                                           key_dim=self.key_dim,
                                           head_num=config['head_num'],
                                           head_dim=config['head_dim'],
                                           num_self_attn=config['text_self_attn_layer'],
                                           num_cat=self.num_cat,
                                           num_subcat=self.num_subcat,
                                           cat_dim=self.cat_dim)
        else:
            self.news_encoder = TextEncoder(word_emb=self.word_emb,
                                            drop=config['dropout'],
                                            word_dim=config['word_emb_dim'],
                                            news_dim=self.news_dim,
                                            key_dim=self.key_dim,
                                            head_num=config['head_num'],
                                            head_dim=config['head_dim'],
                                            num_self_attn=config['text_self_attn_layer'])
        user_encoder = PGTEncoder if config['global'] else NewsSetEncoder
        self.user_encoder = user_encoder(news_encoder=self.news_encoder,
                                         drop=config['dropout'],
                                         news_dim=self.news_dim,
                                         user_dim=self.user_dim,
                                         global_dim=self.global_dim,
                                         key_dim=self.key_dim,
                                         head_num=config['head_num'],
                                         head_dim=config['head_dim'],
                                         num_self_attn=config['history_self_attn_layer'])

    def forward(self, x, source):
        """
        Forward the model given data batch
        :param x: data batch
        :param source: type of data source
        :return: forwarded results
        """
        if source == 'history':
            his_out = self.user_encoder(x)
            return his_out
        elif source == 'pgt':
            user_out = self.user_encoder(x[0], x[1])
            return user_out
        elif source == 'candidate':
            cand_out = self.news_encoder(x)
            return cand_out


class DenseLayer(nn.Module):
    """
    Dense layer class
    """
    def __init__(self, emb_dim, output_dim, num_emb):
        """
        Initialize the class.
        :param emb_dim: dimension of embeddings
        :param output_dim: dimension of outputs
        :param num_emb: number of embeddings
        """
        super(DenseLayer, self).__init__()
        self.emb = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim)
        self.W = nn.Linear(emb_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward the layer given data batch
        :param x: data batch
        :return: forwarded results
        """
        x = self.emb(x)
        out = self.activation(self.W(x))
        return out


class Encoder(nn.Module):
    """
    Encoder class
    """
    def __init__(self, drop, input_dim, output_dim, key_dim, head_num, head_dim):
        """
        Initialize the class.
        :param drop: dropout ratio
        :param input_dim: dimension of inputs
        :param output_dim: dimension of outputs
        :param key_dim: dimension of keys
        :param head_num: number of heads
        :param head_dim: dimension of heads
        """
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=drop)
        self.self_attn = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn = LinearAttn(output_dim=output_dim, key_dim=key_dim)

    def forward(self, **kwargs):
        pass


class AggEncoder(Encoder):
    """
    Aggregation encoder class (title, abstract, category, and sub-category are used)
    """
    def __init__(self, word_emb, drop, word_dim, news_dim, key_dim, head_num, head_dim, num_self_attn,
                 num_cat, num_subcat, cat_dim):
        """
        Initialize the class.
        :param word_emb: word embeddings
        :param drop: dropout ratio
        :param word_dim: dimension of words
        :param news_dim: dimension of news
        :param key_dim: dimension of keys
        :param head_num: number of heads
        :param head_dim: dimension of heads
        :param num_self_attn: number of self attention layers
        :param num_cat: number of categories
        :param num_subcat: number of sub-categories
        :param cat_dim: dimension of category embeddings
        """
        super(AggEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                         output_dim=news_dim, key_dim=key_dim,
                                         head_num=head_num, head_dim=head_dim)
        self.word_emb = word_emb
        self.title_encoder = TextEncoder(word_emb=word_emb,
                                         drop=drop,
                                         word_dim=word_dim,
                                         news_dim=news_dim,
                                         key_dim=key_dim,
                                         head_num=head_num,
                                         head_dim=head_dim,
                                         num_self_attn=num_self_attn)
        self.abs_encoder = TextEncoder(word_emb=word_emb,
                                       drop=drop,
                                       word_dim=word_dim,
                                       news_dim=news_dim,
                                       key_dim=key_dim,
                                       head_num=head_num,
                                       head_dim=head_dim,
                                       num_self_attn=num_self_attn)
        self.cat_encoder = DenseLayer(emb_dim=cat_dim,
                                      output_dim=news_dim,
                                      num_emb=num_cat)
        self.subcat_encoder = DenseLayer(emb_dim=cat_dim,
                                         output_dim=news_dim,
                                         num_emb=num_subcat)

    def forward(self, x):
        """
        Forward the encoder given data batch
        :param x: data batch
        :return: forwarded results
        """
        title = self.title_encoder(x[0])
        abs = self.abs_encoder(x[1])
        cat = self.cat_encoder(x[2])
        subcat = self.subcat_encoder(x[3])
        out = torch.stack((title, abs, cat, subcat), dim=2)
        out = self.linear_attn(out)
        return out


class TextEncoder(Encoder):
    """
    Text encoder class
    """
    def __init__(self, word_emb, drop, word_dim, news_dim, key_dim, head_num, head_dim, num_self_attn):
        """
        Initialize the class.
        :param word_emb: word embeddings
        :param drop: dropout ratio
        :param word_dim: dimension of words
        :param news_dim: dimension of news
        :param key_dim: dimension of keys
        :param head_num: number of heads
        :param head_dim: dimension of heads
        :param num_self_attn: number of self attention layers
        """
        super(TextEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                          output_dim=news_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)
        self.num_self_attn = num_self_attn
        self.word_emb = word_emb

        self.fnn2 = nn.Linear(news_dim, news_dim)
        self.self_attn2 = SelfAttn(head_num=head_num,
                                   head_dim=head_dim,
                                   input_dim=news_dim)
        self.acti = nn.ReLU()

    def forward(self, x):
        """
        Forward the text encoder given data batch
        :param x: data batch
        :return: forwarded results
        """
        x = self.word_emb(x)
        ###
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        ###

        if self.num_self_attn == 2:
            out = self.acti(self.fnn2(out))
            out = self.self_attn2(QKV=(out, out, out))
            out = self.dropout(out)

        out = self.linear_attn(out)
        return out


class NewsSetEncoder(Encoder):
    """
    News set encoder class
    """
    def __init__(self, news_encoder, drop, news_dim, user_dim, global_dim, key_dim, head_num, head_dim, num_self_attn):
        """
        Initialize the class.
        :param news_encoder: news encoder
        :param drop: dropout ratio
        :param news_dim: dimension of news embeddings
        :param user_dim: dimension of user embeddings
        :param global_dim: dimension of global encoding vectors
        :param key_dim: dimension of keys
        :param head_num: number of heads
        :param head_dim: dimension of heads
        :param num_self_attn: number of self attention layers
        """
        super(NewsSetEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                             output_dim=user_dim, key_dim=key_dim,
                                             head_num=head_num, head_dim=head_dim)
        self.num_self_attn = num_self_attn
        self.news_encoder = news_encoder

        self.fnn2 = nn.Linear(user_dim, user_dim)
        self.self_attn2 = SelfAttn(head_num=head_num,
                                   head_dim=head_dim,
                                   input_dim=user_dim)
        self.acti = nn.ReLU()

    def forward(self, x):
        """
        Forward the news set encoder given data batch
        :param x: data batch
        :return: forwarded results
        """
        x = self.news_encoder(x)
        ###
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        ###

        if self.num_self_attn == 2:
            out = self.acti(self.fnn2(out))
            out = self.self_attn2(QKV=(out, out, out))
            out = self.dropout(out)

        out = self.linear_attn(out)
        return out


class PGTEncoder(Encoder):
    """
    PGT encoder class
    """
    def __init__(self, news_encoder, drop, news_dim, user_dim, global_dim, key_dim, head_num, head_dim, num_self_attn):
        """
        Initialize the class.
        :param news_encoder: news encoder
        :param drop: dropout ratio
        :param news_dim: dimension of news embeddings
        :param user_dim: dimension of user embeddings
        :param global_dim: dimension of global encoding vectors
        :param key_dim: dimension of keys
        :param head_num: number of heads
        :param head_dim: dimension of heads
        :param num_self_attn: number of self attention layers
        """
        super(PGTEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                         output_dim=user_dim, key_dim=key_dim,
                                         head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder
        self.global_encoder = NewsSetEncoder(news_encoder, drop, news_dim, user_dim, global_dim, key_dim, head_num, head_dim)
        self.global_attn = GlobalAttn(user_dim, global_dim)

    def forward(self, his, global_pref):
        """
        Forward the PGT encoder given data batch
        :param x: data batch
        :return: forwarded results
        """
        global_pref = self.global_encoder(global_pref)
        global_pref = self.dropout(global_pref)
        his = self.news_encoder(his)
        his = self.dropout(his)
        his = self.self_attn(QKV=(his, his, his))
        his = self.dropout(his)
        out = self.global_attn(his, global_pref)
        return out
