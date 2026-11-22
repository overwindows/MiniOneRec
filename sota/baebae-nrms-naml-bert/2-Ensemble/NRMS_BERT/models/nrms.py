######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/models/model.py
# - Definition of our nrms_bert model.
#
# Version: 1.0
#######################################################################################################
import sys
sys.path.append("../")
sys.path.append("./")
import torch
import torch.nn as nn

from models.utils import SelfAttn, LinearAttn, GlobalAttn, AdditiveAttention


class NRMS(nn.Module):
    """
    Our NRMS_BERT model with the 2-layers self-attention
    """
    def __init__(self, config, word2vec_embedding):
        """
        initialize the model, reading behaviors.tsv and news.tsv
        :param config: configuration
        :param word2vec_embedding: word embedding file for GLUE if use
        """
        super(NRMS, self).__init__()
        self.word_emb = nn.Embedding(word2vec_embedding.shape[0], config['word_emb_dim'])
        self.word_emb.weight = nn.Parameter(torch.tensor(word2vec_embedding, dtype=torch.float32))
        self.word_emb.weight.requires_grad = True
        self.news_dim = config['head_num'] * config['head_dim']
        self.user_dim = self.news_dim
        self.global_dim = self.news_dim
        self.key_dim = config['attention_hidden_dim']

        self.news_encoder = NewsEncoder(word_emb=self.word_emb,
                                        drop=config['dropout'],
                                        word_dim=config['word_emb_dim'],
                                        news_dim=self.news_dim,
                                        key_dim=self.key_dim,
                                        head_num=config['head_num'],
                                        head_dim=config['head_dim'])
        user_encoder = PGTEncoder if config['global'] else NewsSetEncoder
        if config['global']:
            self.user_encoder = user_encoder(news_encoder=self.news_encoder,
                                             drop=config['dropout'],
                                             news_dim=self.news_dim,
                                             user_dim=self.user_dim,
                                             global_dim=self.global_dim,
                                             key_dim=self.key_dim,
                                             head_num=config['head_num'],
                                             head_dim=config['head_dim'])
        else:
            self.user_encoder = user_encoder(news_encoder=self.news_encoder,
                                             drop=config['dropout'],
                                             news_dim=self.news_dim,
                                             user_dim=self.user_dim,
                                             key_dim=self.key_dim,
                                             head_num=config['head_num'],
                                             head_dim=config['head_dim'])
    def forward(self, x, source):
        """
        Get model output
        :param x
        :param source: using PGT or not
        :return: cand_out, output
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


class Encoder(nn.Module):
    """
    Parent class for Encoder classes
    """
    def __init__(self, drop, input_dim, output_dim, key_dim, head_num, head_dim):
        """
        Initialize the encoder
        :param drop: dropout ratio
        :param input_dim: input dimension
        :param output_dim: output dimension
        :param key_dim: key dimension
        :param head_num: number of heads for multi-head attention
        :param head_dim: dimension of heads
        """
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=drop)
        self.self_attn = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn = LinearAttn(output_dim=output_dim, key_dim=key_dim)

        self.self_attn11 = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.MLP1 = nn.Linear(output_dim, input_dim)
        self.self_attn12 = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn1 = LinearAttn(output_dim=output_dim, key_dim=key_dim)

        self.dropout2 = nn.Dropout(p=drop)

        self.self_attn21 = SelfAttn(head_num=head_num,
                                   head_dim=head_dim,
                                   input_dim=input_dim)
        self.MLP2 = nn.Linear(output_dim, input_dim)
        self.self_attn22 = SelfAttn(head_num=head_num,
                                  head_dim=head_dim,
                                  input_dim=input_dim)
        self.linear_attn2 = LinearAttn(output_dim=output_dim, key_dim=key_dim)

        self.final_attention = LinearAttn(output_dim=output_dim,
                                                 key_dim=key_dim)

    def forward(self, **kwargs):
        pass


class NewsEncoder(Encoder):
    """
    Encoder for each news
    """
    def __init__(self, word_emb, drop, word_dim, news_dim, key_dim, head_num, head_dim):
        """
        Initialize the NewsEncoder
        :param word_emb: size of the word embedding
        :param drop: dropout ratio
        :param word_dim: dimension of the word
        :param news_dim: dimension of the news
        :param key_dim: dimension of the key
        :param head_num: number of heads
        :param head_dim: dimension of each head
        """
        super(NewsEncoder, self).__init__(drop=drop, input_dim=word_dim,
                                          output_dim=news_dim, key_dim=key_dim,
                                          head_num=head_num, head_dim=head_dim)

        bert_dim = 1024
        # self.word_emb = word_emb
        self.word_dim = word_dim
        t_hid_dim1 = (bert_dim + self.word_dim) * 2 // 3
        t_hid_dim2 = (bert_dim + self.word_dim) * 1 // 3
        self.t_l1 = nn.Linear(bert_dim, t_hid_dim1)
        self.t_l2 = nn.Linear(t_hid_dim1, t_hid_dim2)
        self.t_l3 = nn.Linear(t_hid_dim2, self.word_dim)

        c_hid_dim = (bert_dim + news_dim) // 2
        self.c_l1 = nn.Linear(bert_dim, c_hid_dim)
        self.c_l2 = nn.Linear(c_hid_dim, news_dim)

        sc_hid_dim = (bert_dim + news_dim) // 2
        self.sc_l1 = nn.Linear(bert_dim, sc_hid_dim)
        self.sc_l2 = nn.Linear(sc_hid_dim, news_dim)

        abs_hid_dim1 = (bert_dim + self.word_dim) * 1 // 3
        abs_hid_dim2 = (bert_dim + self.word_dim) * 2 // 3
        self.abs_l1 = nn.Linear(bert_dim, abs_hid_dim1)
        self.abs_l2 = nn.Linear(abs_hid_dim1, abs_hid_dim2)
        self.abs_l3 = nn.Linear(abs_hid_dim2, self.word_dim)

        self.relu = nn.ReLU()


    def forward(self, inputs):
        """
        Get output
        :param inputs:
        :return: news_vectorL encoding for the news
        """
        t_emb = inputs['title'].float()
        c_emb = inputs['category'].float()
        sc_emb = inputs['subcategory'].float()
        abs_emb = inputs['abstract'].float()

        # process title embedding
        t = self.t_l1(t_emb)
        t = self.relu(t)
        t = self.t_l2(t)
        t = self.relu(t)
        t = self.t_l3(t)
        t = self.dropout(t)
        t = self.self_attn11(QKV=(t, t, t))
        t = self.dropout(t)
        t = self.MLP1(t)
        t = self.relu(t)
        t = self.self_attn12(QKV=(t,t,t))
        self.dropout(t)
        t = self.linear_attn1(t)

        # process category embedding
        c = self.c_l1(c_emb)
        c = self.relu(c)
        c = self.c_l2(c)
        c = self.relu(c)

        # process subcategory embedding
        sc = self.sc_l1(sc_emb)
        sc = self.relu(sc)
        sc = self.sc_l2(sc)
        sc = self.relu(sc)

        # process abs embedding
        abs = self.abs_l1(abs_emb)
        abs = self.relu(abs)
        abs = self.abs_l2(abs)
        abs = self.relu(abs)
        abs = self.abs_l3(abs)
        abs = self.dropout2(abs)
        abs = self.self_attn21(QKV=(abs, abs, abs))
        abs = self.dropout2(abs)
        abs = self.MLP2(abs)
        abs = self.relu(abs)
        abs = self.self_attn22(QKV=(abs, abs, abs))
        abs = self.dropout2(abs)
        abs = self.linear_attn2(abs)

        # Final attention using all features
        stacked_news_vector = torch.stack([c, sc, t, abs], dim=2)
        news_vector = self.final_attention(stacked_news_vector)
        # news_vector = self.final_attention(stacked_news_vector)

        return news_vector


class NewsSetEncoder(Encoder):
    """
    News set encoder
    """
    def __init__(self, news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim):
        """
        Initialize the NewsSetEncoder
        :param news_encoder:
        :param drop: dropout ratio
        :param news_dim: dimension of news
        :param user_dim: dimension of user
        :param key_dim: dimension of the key
        :param head_num: number of heads
        :param head_dim: dimension of the head
        """
        super(NewsSetEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                             output_dim=user_dim, key_dim=key_dim,
                                             head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder

    def forward(self, x):
        """
        Get output
        :param x: input
        :return: out
        """
        x = self.news_encoder(x)
        out = self.dropout(x)
        out = self.self_attn(QKV=(out, out, out))
        out = self.dropout(out)
        out = self.linear_attn(out)
        return out


class PGTEncoder(Encoder):
    """
    Encoder with PGT
    """
    def __init__(self, news_encoder, drop, news_dim, user_dim, global_dim, key_dim, head_num, head_dim):
        """
        Initialize the PGTEncoder
        :param news_encoder:
        :param drop: dropout ratio
        :param news_dim: dimension of the news data
        :param user_dim: dimension of the user
        :param global_dim: global dimension
        :param key_dim: dimension of key
        :param head_num: number of heads
        :param head_dim: dimension of heads
        """
        super(PGTEncoder, self).__init__(drop=drop, input_dim=news_dim,
                                         output_dim=user_dim, key_dim=key_dim,
                                         head_num=head_num, head_dim=head_dim)
        self.news_encoder = news_encoder
        self.global_encoder = NewsSetEncoder(news_encoder, drop, news_dim, user_dim, key_dim, head_num, head_dim)
        self.global_attn = GlobalAttn(user_dim, global_dim)

    def forward(self, his, global_pref):
        """
        Get output
        :param his: history
        :param global_pref: global preference
        :return: out
        """
        global_pref = self.global_encoder(global_pref)
        global_pref = self.dropout(global_pref)
        his = self.news_encoder(his)
        his = self.dropout(his)
        his = self.self_attn(QKV=(his, his, his))
        his = self.dropout(his)
        out = self.global_attn(his, global_pref)
        return out
