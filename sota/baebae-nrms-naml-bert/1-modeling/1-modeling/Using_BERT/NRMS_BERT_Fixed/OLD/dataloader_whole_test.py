import os
import torch
import re
import numpy as np
from random import sample
from tqdm import tqdm
import timeit
import pandas as pd
import pickle

from utils.utils import parallelize_dataframe, process_behaviors
from utils.bert import get_word_embedding_bert


# def word_tokenize(sent):
#     """ Split sentence into word list using regex.
#     Args:
#         sent (str): Input sentence
#
#     Return:
#         list: word list
#     """
#     pat = re.compile(r"[\w]+|[.,!?;|]")
#     if isinstance(sent, str):
#         return pat.findall(sent.lower())
#     else:
#         return []


class DataSet(torch.utils.data.Dataset):
    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, selector, config,
                 col_spliter="\t"):
        self.word2idx = word2idx
        self.uid2idx = uid2idx
        self.selector = selector
        self.col_spliter = col_spliter
        self.title_size = config['title_size']
        self.his_size = config['his_size']
        self.npratio = config['npratio']
        self.num_cores = config['num_cores']
        self.nid2idx, self.news_title_index = self.init_news(news_file)
        self.histories, self.imprs, self.labels, self.raw_impr_idxs,\
        self.impr_idxs, self.uidxs, self.times, self.pops, self.freshs = \
            self.init_behaviors(behaviors_file)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def init_news(self, news_file):
        """ init news information given news file, such as news_title_index and nid2index.
        Args:
            news_file: path of news file
        Outputs:
            nid2index: news id to index
            news_title_index: word_ids of titles
        """

        tokens = news_file.split('/')
        pickle_file = '/'.join(tokens[:-1]) + f"/BERT/bert_title_{self.title_size}.pickle"

        assert os.path.isfile(pickle_file)
        # if not os.path.isfile(pickle_file):
        #     get_word_embedding_bert(news_file, pickle_file, title_size=self.title_size, n=30)

        with open(pickle_file, 'rb') as f:
            nid2index, news_title_index = pickle.load(f)
        news_title_index.requires_grad=False

        return nid2index, news_title_index

    def init_behaviors(self, behaviors_file):
        """ init behavior logs given behaviors file.

        Args:
            behaviors_file: path of behaviors file

        Outputs:
            histories: nids of histories
            imprs: nids of impressions
            labels: labels of impressions
            impr_indexes: index of the behavior
            uindexes: index of users
        """
        st = timeit.default_timer()
        df = pd.read_csv(behaviors_file, sep='\t', header=None)
        df = parallelize_dataframe(df, process_behaviors, num_cores=self.num_cores,
                                   class_pointer=self)

        histories = df['histories'].tolist()
        imprs = df['imprs'].tolist()
        labels = df['labels'].tolist()
        raw_impr_indexes = df['raw_impr_indexes'].tolist()
        # impr_indexes = df['impr_indexes'].tolist()
        impr_indexes = list(range(df.shape[0]))
        uindexes = df['uindexes'].tolist()
        times = df['times'].tolist()
        pops = df['pops'].tolist()
        freshs = df['freshs'].tolist()
        print(f"Initializing behavior end ({timeit.default_timer() - st}s)")

        return histories, imprs, labels, raw_impr_indexes, impr_indexes, uindexes, times, pops, freshs


class DataSetTrn(DataSet):
    nid2idx = None
    news_title_index = None
    histories = None
    imprs = None
    labels = None
    impr_idxs = None
    uidxs = None
    times = None

    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, selector, config):
        super().__init__(news_file, behaviors_file, word2idx, uid2idx, selector, config)
        # unfolding
        self.histories_unfold = []
        self.impr_idxs_unfold = []
        self.uidxs_unfold = []
        self.pos_unfold = []
        self.neg_unfold = []
        self.times_unfold = []
        self.pop_unfold = []
        self.fresh_unfold = []

        for line in range(len(self.uidxs)):
            neg_idxs = [i for i, x in enumerate(self.labels[line]) if x == 0]
            pos_idxs = [i for i, x in enumerate(self.labels[line]) if x == 1]
            if len(pos_idxs) < 1:
                continue
            for pos_idx in pos_idxs:
                self.pos_unfold.append([self.imprs[line][pos_idx]])
                if len(neg_idxs) == 0:
                    negs = [0] * self.npratio
                else:
                    negs = [self.imprs[line][i] for i in neg_idxs]
                    if len(neg_idxs) < self.npratio:
                        negs += [0] * (self.npratio - len(neg_idxs))
                self.neg_unfold.append(negs)
                self.histories_unfold.append(self.histories[line])
                self.impr_idxs_unfold.append(self.impr_idxs[line])
                self.uidxs_unfold.append(self.uidxs[line])
                self.times_unfold.append(self.times[line])
                self.pop_unfold.append(self.pops[line])
                self.fresh_unfold.append(self.freshs[line])

    def __getitem__(self, idx):
        negs = sample(self.neg_unfold[idx], self.npratio)
        his = self.news_title_index[self.histories_unfold[idx]]
        pos = self.news_title_index[self.pos_unfold[idx]]
        neg = self.news_title_index[negs]
        pop = self.news_title_index[self.pop_unfold[idx]]
        fresh = self.news_title_index[self.fresh_unfold[idx]]
        return his, pos, neg, pop, fresh

        # return torch.tensor(his).long(), torch.tensor(pos).long(), torch.tensor(neg).long(),\
        #        torch.tensor(pop).long(), torch.tensor(fresh).long()

    def __len__(self):
        return len(self.uidxs_unfold)


class DataSetTest(DataSet):
    nid2idx = None
    news_title_index = None
    histories = None
    imprs = None
    labels = None
    impr_idxs = None
    uidxs = None
    times = None

    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, selector, config, label_known=True):
        self.label_known = label_known
        super().__init__(news_file, behaviors_file, word2idx, uid2idx, selector, config)

        self.histories_words = []
        self.imprs_words = []
        self.pops_words = []
        self.freshs_words = []

        for i in range(len(self.histories)):
            self.histories_words.append(self.news_title_index[self.histories[i]])
            self.imprs_words.append(self.news_title_index[self.imprs[i]])
            self.pops_words.append(self.news_title_index[self.pops[i]])
            self.freshs_words.append(self.news_title_index[self.freshs[i]])

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.uidxs)