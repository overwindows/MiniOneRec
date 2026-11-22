import torch
import re
import numpy as np
from random import sample
from tqdm import tqdm
import pandas as pd
import timeit
import os
from utils.utils import parallelize_dataframe, process_behaviors


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
    def __init__(self, news_file, behaviors_file, tokenizer, uid2idx, selector, config,
                 col_spliter="\t"):
        # self.word2idx = word2idx
        self.tokenizer = tokenizer # I fixed
        self.uid2idx = uid2idx
        self.selector = selector
        self.col_spliter = col_spliter
        self.title_size = config['title_size']
        self.his_size = config['his_size']
        self.npratio = config['npratio']
        self.num_cores = config['num_cores']
        self.nid2idx, self.input_ids, self.attention_mask = self.init_news(news_file)
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

        st = timeit.default_timer()
        news_title = [""]
        df = pd.read_csv(news_file, sep=self.col_spliter, header=None)
        titles = news_title + df[3].map(lambda x: '[CLS]' + x + '[SEP]').tolist()
        inputs = self.tokenizer(titles, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.title_size,
                                add_special_tokens=False)
        nid2index = {}
        for idx, nid in enumerate(df[0]):
            nid2index[nid] = idx+1

        print(f"Initializing News End ({timeit.default_timer() - st}s)")
        return nid2index, inputs['input_ids'], inputs['attention_mask']

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
    input_ids = None
    attention_mask = None
    histories = None
    imprs = None
    labels = None
    impr_idxs = None
    uidxs = None
    times = None

    def __init__(self, news_file, behaviors_file, tokenizer, uid2idx, selector, config):
        super().__init__(news_file, behaviors_file, tokenizer, uid2idx, selector, config)

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

        his_in = self.input_ids[self.histories_unfold[idx]]
        his_att = self.attention_mask[self.histories_unfold[idx]]
        his = {'input_ids': his_in, 'attention_mask':his_att}

        pos_in = self.input_ids[self.pos_unfold[idx]]
        pos_att = self.attention_mask[self.pos_unfold[idx]]
        pos = {'input_ids': pos_in, 'attention_mask': pos_att}

        neg_in = self.input_ids[negs]
        neg_att = self.attention_mask[negs]
        neg = {'input_ids': neg_in, 'attention_mask': neg_att}

        pop_in = self.input_ids[self.pop_unfold[idx]]
        pop_att = self.attention_mask[self.pop_unfold[idx]]
        pop = {'input_ids': pop_in, 'attention_mask': pop_att}

        fresh_in = self.input_ids[self.fresh_unfold[idx]]
        fresh_att = self.attention_mask[self.fresh_unfold[idx]]
        fresh = {'input_ids': fresh_in, 'attention_mask': fresh_att}

        return his, pos, neg, pop, fresh

    def __len__(self):
        return len(self.uidxs_unfold)


class DataSetTest(DataSet):
    nid2idx = None
    input_ids = None
    attention_mask = None
    histories = None
    imprs = None
    labels = None
    impr_idxs = None
    uidxs = None
    times = None

    def __init__(self, news_file, behaviors_file, tokenizer, uid2idx, selector, config, label_known=True):
        self.label_known = label_known
        super().__init__(news_file, behaviors_file, tokenizer, uid2idx, selector, config)

        self.histories_words = []
        self.imprs_words = []
        self.pops_words = []
        self.freshs_words = []

        for i in range(len(self.histories)):
            self.histories_words.append(
                {
                    'input_ids': self.input_ids[self.histories[i]],
                    'attention_mask': self.attention_mask[self.histories[i]]
                }
            )
            self.imprs_words.append(
                {
                    'input_ids': self.input_ids[self.imprs[i]],
                    'attention_mask': self.attention_mask[self.imprs[i]]
                }
            )

            self.pops_words.append(
                {
                    'input_ids': self.input_ids[self.pops[i]],
                    'attention_mask': self.attention_mask[self.pops[i]]
                }
            )

            self.freshs_words.append(
                {
                    'input_ids': self.input_ids[self.freshs[i]],
                    'attention_mask': self.attention_mask[self.freshs[i]]
                }
            )

            # self.histories_words.append(self.news_title_index[self.histories[i]])
            # self.imprs_words.append(self.news_title_index[self.imprs[i]])
            # self.pops_words.append(self.news_title_index[self.pops[i]])
            # self.freshs_words.append(self.news_title_index[self.freshs[i]])

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.uidxs)