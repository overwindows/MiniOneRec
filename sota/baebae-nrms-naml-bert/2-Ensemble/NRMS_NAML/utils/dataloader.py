######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/utils/dataloader.py
# - This file includes utility functions useful for defining dataloaders
#
# Version: 1.0
#######################################################################################################

import torch
import re
import numpy as np
from random import sample
from tqdm import tqdm


def word_tokenize(sent):
    """
    Split sentence into word list using regex.
    :param sent: input sequence
    :return: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


class DataSet(torch.utils.data.Dataset):
    """
    Dataset class for mind datasets
    """
    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, cat2idx, subcat2idx, selector, config,
                 label_known=True,
                 col_splitter="\t"):
        """
        Initialize the class.
        :param news_file: news file path
        :param behaviors_file: behavior file path
        :param word2idx: word to index dictionary
        :param uid2idx: uid to index dictionary
        :param cat2idx: category to index dictionary
        :param subcat2idx: sub-category to index dictionary
        :param selector: popular/fresh news selector
        :param config: configuration dictionary
        :param label_known: whether the dataset includes labels of impressions
        :param col_splitter: column splitter of the file
        """
        self.word2idx = word2idx
        self.uid2idx = uid2idx
        self.cat2idx = cat2idx
        self.subcat2idx = subcat2idx
        self.selector = selector
        self.col_spliter = col_splitter
        self.title_size = config['title_size']
        self.abstract_size = config['abstract_size']
        self.his_size = config['his_size']
        self.npratio = config['npratio']
        self.label_known = label_known
        self.nid2idx, self.news_title_index, self.news_abstract_index,\
        self.news_cat_index, self.news_subcat_index = self.init_news(news_file)
        self.histories, self.imprs, self.labels, self.raw_impr_idxs,\
        self.impr_idxs, self.uidxs, self.times, self.pops, self.freshs = \
            self.init_behaviors(behaviors_file)

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def init_news(self, news_file):
        """
        Initialize news information given news file.
        :param news_file: news file path
        :return: information of each news (index, title, abstract, category, sub-category)
        """
        nid2index = {}
        news_title = ['']
        news_abstract = ['']
        news_cat = ['']
        news_subcat = ['']

        with open(news_file, 'r') as rd:
            for line in tqdm(rd, desc='Init news'):
                nid, cat, subcat, title, abstract,\
                url, title_ent, abstract_ent = line.strip("\n").split(self.col_spliter)

                if nid in nid2index:
                    continue

                nid2index[nid] = len(nid2index) + 1
                title = word_tokenize(title)
                abstract = word_tokenize(abstract)
                news_title.append(title)
                news_abstract.append(abstract)
                news_cat.append(cat)
                news_subcat.append(subcat)

        news_title_index = np.zeros((len(news_title), self.title_size), dtype='int32')
        news_abstract_index = np.zeros((len(news_abstract), self.abstract_size), dtype='int32')
        news_cat_index = np.zeros(len(news_cat), dtype='int32')
        news_subcat_index = np.zeros(len(news_subcat), dtype='int32')

        for news_index in range(len(news_title)):
            title = news_title[news_index]
            for word_index in range(min(self.title_size, len(title))):
                if title[word_index] in self.word2idx:
                    news_title_index[news_index, word_index] = \
                    self.word2idx[title[word_index].lower()]

        for news_index in range(len(news_abstract)):
            abstract = news_abstract[news_index]
            for word_index in range(min(self.abstract_size, len(abstract))):
                if abstract[word_index] in self.word2idx:
                    news_abstract_index[news_index, word_index] = \
                    self.word2idx[abstract[word_index].lower()]

        for news_index in range(len(news_cat)):
            cat = news_cat[news_index]
            news_cat_index[news_index] = self.cat2idx[cat] if cat in self.cat2idx else 0

        for news_index in range(len(news_subcat)):
            cat = news_subcat[news_index]
            news_subcat_index[news_index] = self.subcat2idx[cat] if cat in self.subcat2idx else 0

        return nid2index, news_title_index, news_abstract_index, news_cat_index, news_subcat_index

    def init_behaviors(self, behaviors_file):
        """
        Initialize behavior logs given a behavior file.
        :param behaviors_file: behavior file path
        :return: information of each behavior (history, impressions, labels of impressions, times,
                                               popular news, fresh news)
        """
        raw_impr_indexes = []
        histories = []
        imprs = []
        labels = []
        impr_indexes = []
        uindexes = []
        times = []
        pops = []
        freshs = []

        with open(behaviors_file, 'r') as rd:
            impr_index = 0
            for line in tqdm(rd, desc='Init behaviors'):
                raw_impr_id, uid, time, history, impr = line.strip("\n").split(
                    self.col_spliter)[-5:]

                history = [self.nid2idx[i] for i in history.split()]
                # padding
                history = [0]*(self.his_size-len(history)) + history[:self.his_size]

                impr_news = [self.nid2idx[i.split("-")[0]] for i in impr.split()]
                if self.label_known:
                    label = [int(i.split("-")[1]) for i in impr.split()]
                else:
                    label = None
                uindex = self.uid2idx[uid] if uid in self.uid2idx else 0

                pop = [self.nid2idx[i] for i in self.selector.get_pop_recommended(time)]
                # pop = [self.nid2idx[i] for i in self.selector.get_pop_clicked(time)]
                fresh = [self.nid2idx[i] for i in self.selector.get_fresh(time)]

                histories.append(history)
                imprs.append(impr_news)
                labels.append(label)
                raw_impr_indexes.append(raw_impr_id)
                impr_indexes.append(impr_index)
                uindexes.append(uindex)
                times.append(time)
                pops.append(pop)
                freshs.append(fresh)
                impr_index += 1

        return histories, imprs, labels, raw_impr_indexes, impr_indexes, uindexes, times, pops, freshs


class DataSetTrn(DataSet):
    """
    Dataset class for mind training dataset
    """
    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, cat2idx, subcat2idx, selector, config):
        """
        Initialize the class.
        :param news_file: news file path
        :param behaviors_file: behavior file path
        :param word2idx: word to index dictionary
        :param uid2idx: uid to index dictionary
        :param cat2idx: category to index dictionary
        :param subcat2idx: sub-category to index dixtionary
        :param selector: popular/fresh news selector
        :param config: configuration dictionary
        """
        super().__init__(news_file, behaviors_file, word2idx, uid2idx, cat2idx, subcat2idx, selector, config)

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
        """
        Policy of getting an item given an index
        :param idx: index of dataset
        :return: title/abstarch/category/sub-category of
                 history, positive news, negative news, popular news, fresh news
        """
        negs = sample(self.neg_unfold[idx], self.npratio)

        # title
        his_title = self.news_title_index[self.histories_unfold[idx]]
        pos_title = self.news_title_index[self.pos_unfold[idx]]
        neg_title = self.news_title_index[negs]
        pop_title = self.news_title_index[self.pop_unfold[idx]]
        fresh_title = self.news_title_index[self.fresh_unfold[idx]]

        # abstract
        his_abs = self.news_abstract_index[self.histories_unfold[idx]]
        pos_abs = self.news_abstract_index[self.pos_unfold[idx]]
        neg_abs = self.news_abstract_index[negs]
        pop_abs = self.news_abstract_index[self.pop_unfold[idx]]
        fresh_abs = self.news_abstract_index[self.fresh_unfold[idx]]

        # category
        his_cat = self.news_cat_index[self.histories_unfold[idx]]
        pos_cat = self.news_cat_index[self.pos_unfold[idx]]
        neg_cat = self.news_cat_index[negs]
        pop_cat = self.news_cat_index[self.pop_unfold[idx]]
        fresh_cat = self.news_cat_index[self.fresh_unfold[idx]]

        # sub-category
        his_subcat = self.news_subcat_index[self.histories_unfold[idx]]
        pos_subcat = self.news_subcat_index[self.pos_unfold[idx]]
        neg_subcat = self.news_subcat_index[negs]
        pop_subcat = self.news_subcat_index[self.pop_unfold[idx]]
        fresh_subcat = self.news_subcat_index[self.fresh_unfold[idx]]

        return torch.tensor(his_title).long(), torch.tensor(pos_title).long(), torch.tensor(neg_title).long(),\
               torch.tensor(pop_title).long(), torch.tensor(fresh_title).long(), \
               torch.tensor(his_abs).long(), torch.tensor(pos_abs).long(), torch.tensor(neg_abs).long(), \
               torch.tensor(pop_abs).long(), torch.tensor(fresh_abs).long(), \
               torch.tensor(his_cat).long(), torch.tensor(pos_cat).long(), torch.tensor(neg_cat).long(), \
               torch.tensor(pop_cat).long(), torch.tensor(fresh_cat).long(), \
               torch.tensor(his_subcat).long(), torch.tensor(pos_subcat).long(), torch.tensor(neg_subcat).long(), \
               torch.tensor(pop_subcat).long(), torch.tensor(fresh_subcat).long()

    def __len__(self):
        """
        Return length of dataset.
        :return: length of dataset
        """
        return len(self.uidxs_unfold)


class DataSetTest(DataSet):
    """
    Dataset class for mind test dataset
    """
    def __init__(self, news_file, behaviors_file, word2idx, uid2idx, cat2idx, subcat2idx, selector, config, label_known=True):
        """
        Initialize the class.
        :param news_file: news file path
        :param behaviors_file: behavior file path
        :param word2idx: word to index dictionary
        :param uid2idx: uid to index dictionary
        :param cat2idx: category to index dictionary
        :param subcat2idx: sub-category to index dictionary
        :param selector: popular/fresh news selector
        :param config: configuration dictionary
        :param label_known: whether the dataset includes labels of impressions
        """
        super().__init__(news_file, behaviors_file, word2idx, uid2idx, cat2idx, subcat2idx, selector, config, label_known)

        # title
        self.histories_title = []
        self.imprs_title = []
        self.pops_title = []
        self.freshs_title = []

        # abstract
        self.histories_abs = []
        self.imprs_abs = []
        self.pops_abs = []
        self.freshs_abs = []

        # category
        self.histories_cat = []
        self.imprs_cat = []
        self.pops_cat = []
        self.freshs_cat = []

        # sub-category
        self.histories_subcat = []
        self.imprs_subcat = []
        self.pops_subcat = []
        self.freshs_subcat = []

        for i in range(len(self.histories)):
            # title
            self.histories_title.append(self.news_title_index[self.histories[i]])
            self.imprs_title.append(self.news_title_index[self.imprs[i]])
            self.pops_title.append(self.news_title_index[self.pops[i]])
            self.freshs_title.append(self.news_title_index[self.freshs[i]])

            # abstract
            self.histories_abs.append(self.news_abstract_index[self.histories[i]])
            self.imprs_abs.append(self.news_abstract_index[self.imprs[i]])
            self.pops_abs.append(self.news_abstract_index[self.pops[i]])
            self.freshs_abs.append(self.news_abstract_index[self.freshs[i]])

            # category
            self.histories_cat.append(self.news_cat_index[self.histories[i]])
            self.imprs_cat.append(self.news_cat_index[self.imprs[i]])
            self.pops_cat.append(self.news_cat_index[self.pops[i]])
            self.freshs_cat.append(self.news_cat_index[self.freshs[i]])

            # sub-category
            self.histories_subcat.append(self.news_subcat_index[self.histories[i]])
            self.imprs_subcat.append(self.news_subcat_index[self.imprs[i]])
            self.pops_subcat.append(self.news_subcat_index[self.pops[i]])
            self.freshs_subcat.append(self.news_subcat_index[self.freshs[i]])

    def __getitem__(self, idx):
        pass

    def __len__(self):
        """
        Return length of dataset.
        :return: length of dataset
        """
        return len(self.uidxs)
