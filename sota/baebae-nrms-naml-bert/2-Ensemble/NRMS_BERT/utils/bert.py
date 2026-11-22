######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/utils/bert.py
# - Save word embeddings for each news.tsv files using BERT.
#
# Version: 1.0
#######################################################################################################

import sys
sys.path.append("../")
sys.path.append("./")

import pandas as pd
import torch
import itertools
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pickle
import os, sys
import csv

def get_title_embedding_bert(df, tokenizer, model, pickle_file, title_size=30, n=30):
    """
    Get word embedding for the titles
    :param df: dataframe of the news
    :param tokenizer: BERT tokenizer to use
    :param model: BERT model to use
    :param pickle_file: output pickle file path
    :param title_size: number of tokens to save
    :param n: number of workers
    """
    nid2index = {}
    titles = ['']

    for r in tqdm(range(df.shape[0])):
        nid = df[0][r]
        title = df[3][r]

        if nid in nid2index:
            continue
        nid2index[nid] = len(nid2index) + 1
        titles.append('[CLS]' + title + '[SEP]')

    tokenized_titles = tokenizer(titles, max_length =title_size, padding=True, truncation=True,
                                 add_special_tokens=False, return_tensors='pt')

    def get_tokenized_titles(t, _from, _to):
        res = {}
        for key in t:
            res[key] = t[key][_from:_to]
        return res

    num_news = len(titles)
    gap = num_news // (n-1)

    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_title_embeds = (model(**get_tokenized_titles(tokenized_titles, ranges[i], ranges[i+1]))[0])
        news_title_embeds = news_title_embeds.half()
        with open(pickle_file + f'{i}', 'wb') as f:
            pickle.dump(news_title_embeds, f)

    for i in range(n):
        file_name = pickle_file + f'{i}'
        with open(file_name, 'rb') as f:
            if i == 0:
                z = pickle.load(f)
            else:
                z = torch.cat((z, pickle.load(f)))
        os.remove(file_name) # remove used file
        res = [nid2index, z]
    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)


def get_category_embedding_bert(df, tokenizer, model, pickle_file, n=30):
    """
         Get word embedding for the categories
         :param df: dataframe of the news
         :param tokenizer: BERT tokenizer to use
         :param model: BERT model to use
         :param pickle_file: output pickle file path
         :param n: number of workers
    """
    dict = {'foodanddrink': 'food'}
    def processing_cats(cat):
        if cat in dict:
            return dict[cat]
        else:
            return cat

    cats = df[1].map(lambda x: processing_cats(x)).unique().tolist()
    cat2idx = {}
    for i, cat in enumerate(cats):
        cat2idx[cat] = i

    tokenized_cats = tokenizer(cats, return_tensors='pt', padding=True, truncation=True, max_length=3)
    cats_embedding = (model(**tokenized_cats)[0][:, 1, :])
    cats_embedding = cats_embedding.half()

    res = (cat2idx, cats_embedding, df[1].map(lambda x: processing_cats(x)))

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)


def get_subcategory_embedding_bert(df, tokenizer, model, pickle_file, n=30):
    """
         Get word embedding for the subcategories
         :param df: dataframe of the news
         :param tokenizer: BERT tokenizer to use
         :param model: BERT model to use
         :param pickle_file: output pickle file path
         :param n: number of workers
    """
    subcats = df[2]

    # Read mapping file
    mapping_df = pd.read_csv('subcategory_mapping.tsv', header=None, sep='\t')
    _dict = {}
    for r in range(mapping_df.shape[0]):
        _dict[mapping_df[0][r]] = mapping_df[1][r]

    # Do mapping using pre-defined dictionary
    subcats = subcats.map(lambda x: _dict[x]).unique().tolist()
    subcat2idx = {}
    for i, cat in enumerate(subcats):
        subcat2idx[cat] = i

    tokenized_subcats = tokenizer(subcats, return_tensors='pt', padding=True, max_length=10)
    subcat_embedding = (model(**tokenized_subcats)[0])[:,0,:]
    subcat_embedding = subcat_embedding.half()

    res = (subcat2idx, subcat_embedding, df[2].map(lambda x: _dict[x]))

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)

def get_abs_embedding_bert(df, tokenizer, model, pickle_file, abs_size=30, n=30):
    """
    Get word embedding for the abstracts
    :param df: dataframe of the news
    :param tokenizer: BERT tokenizer to use
    :param model: BERT model to use
    :param pickle_file: output pickle file path
    :param abs_size: number of tokens to save
    :param n: number of workers
    """
    abs = df[4]
    abs = abs.fillna('')
    def process_abs(x):
        if len(x) > 500:
            return '[CLS]' + x[0:500] + '[SEP]'
        elif x== '':
            return ''
        else:
            return '[CLS]' + x + '[SEP]'
    abs = abs.map(lambda x: process_abs(x)).tolist()  # [CLS], [SEP]
    tokenized_abs = tokenizer(abs, max_length =abs_size, padding=True, truncation=True,
                                 add_special_tokens=False, return_tensors='pt')

    def get_tokenized_abs(t, _from, _to):
        res = {}
        for key in t:
            res[key] = t[key][_from:_to]
        return res

    num_news = len(abs)
    gap = num_news // (n-1)

    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_abs_embeds = (model(**get_tokenized_abs(tokenized_abs, ranges[i], ranges[i+1]))[0])
        news_abs_embeds = news_abs_embeds.half()
        with open(pickle_file + f'{i}', 'wb') as f:
            pickle.dump(news_abs_embeds, f)

    for i in range(n):
        file_name = pickle_file + f'{i}'
        with open(file_name, 'rb') as f:
            if i == 0:
                z = pickle.load(f)
            else:
                z = torch.cat((z, pickle.load(f)))
        os.remove(file_name) # remove used file
        res = z

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)


if __name__ == "__main__":
    print('start')
    model_name = 'bert-large-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # [demo, large], ['train', 'dev']
    dt1, dt2 = 'demo', 'train'
    for dt1 in ['demo', 'large']:
        for dt2 in ['train', 'dev','test']:
            if dt1 == 'demo' and dt2 == 'test':
                continue
            news_file = f'/data/mind/MIND{dt1}_{dt2}/news.tsv'
            df = pd.read_csv(news_file, sep='\t', header=None, quoting= csv.QUOTE_NONE)

            # get title embedding
            title_size = 30
            n=100
            title_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_title_{title_size}.pickle'
            # get_title_embedding_bert(df, tokenizer, model, title_pickle_file, title_size=title_size, n=100)
            print(f"{dt1}-{dt2}/ get title embedding done")

            # get category embedding
            category_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_category2.pickle'
            get_category_embedding_bert(df, tokenizer, model, category_pickle_file, n=100)
            print(f"{dt1}-{dt2}/ get category embedding done")

            # get subcategory embedding
            subcategory_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_subcategory2.pickle'
            get_subcategory_embedding_bert(df, tokenizer, model, subcategory_pickle_file, n=100)
            print(f"{dt1}-{dt2}/ get subcategory embedding done")

            # get abstract embedding
            abs_size = 30
            abs_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_abs_{title_size}.pickle'
            # get_abs_embedding_bert(df, tokenizer, model, abs_pickle_file, abs_size=abs_size, n=100)
            print(f"{dt1}-{dt2}/ get abstract embedding done")
            print(f"{dt1}-{dt2} done")
