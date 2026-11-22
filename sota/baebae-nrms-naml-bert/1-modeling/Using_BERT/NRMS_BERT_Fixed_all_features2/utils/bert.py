import pandas as pd
import torch
import itertools
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pickle
import os, sys
import csv

def get_title_embedding_bert(df, tokenizer, model, pickle_file, title_size=30, n=30):
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
    dict = {'foodanddrink': 'food'}
    def processing_cats(cat):
        if cat in dict:
            return dict[cat]
        else:
            return cat

    cats = df[1].map(lambda x: processing_cats(x)).tolist()

    tokenized_cats = tokenizer(cats, return_tensors='pt', padding=True, truncation=True, max_length=3)

    def get_tokenized_cats(c, _from, _to):
        res = {}
        for key in c:
            res[key] = c[key][_from:_to]
        return res

    num_news = len(cats)
    gap = num_news // (n-1)

    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_cats_embeds = (model(**get_tokenized_cats(tokenized_cats, ranges[i], ranges[i+1]))[0][:,1,:])
        news_cats_embeds = news_cats_embeds.half()
        with open(pickle_file + f'{i}', 'wb') as f:
            pickle.dump(news_cats_embeds, f)

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


def get_category_embedding_bert2(df, tokenizer, model, pickle_file, n=30):
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

    # cat --> cat2idx[cat] --> cats_embedding[cat2idx[cat]]
    # res = torch.stack(df[1].map(lambda x: cats_embedding[cat2idx[processing_cats(x)]]).tolist())
    res = (cat2idx, cats_embedding, df[1].map(lambda x: processing_cats(x)))

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)


def get_subcategory_embedding_bert(df, tokenizer, model, pickle_file, n=30):
    subcats = df[2]

    # Read mapping file
    mapping_df = pd.read_csv('subcategory_mapping.tsv', header=None, sep='\t')
    _dict = {}
    for r in range(mapping_df.shape[0]):
        _dict[mapping_df[0][r]] = mapping_df[1][r]

    # Do mapping
    subcats = subcats.map(lambda x: _dict[x]).tolist()
    tokenized_cats = tokenizer(subcats, return_tensors='pt', padding=True, max_length=10)

    def get_tokenized_subcats(c, _from, _to):
        res = {}
        for key in c:
            res[key] = c[key][_from:_to]
        return res

    num_news = len(subcats)
    gap = num_news // (n - 1)

    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_subcat_embeds = (model(**get_tokenized_subcats(tokenized_cats, ranges[i], ranges[i + 1]))[0][:, 0, :])
        news_subcat_embeds = news_subcat_embeds.half()
        with open(pickle_file + f'{i}', 'wb') as f:
            pickle.dump(news_subcat_embeds, f)

    for i in range(n):
        file_name = pickle_file + f'{i}'
        with open(file_name, 'rb') as f:
            if i == 0:
                z = pickle.load(f)
            else:
                z = torch.cat((z, pickle.load(f)))
        os.remove(file_name)  # remove used file
        res = z

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)


def get_subcategory_embedding_bert2(df, tokenizer, model, pickle_file, n=30):
    subcats = df[2]

    # Read mapping file
    mapping_df = pd.read_csv('subcategory_mapping.tsv', header=None, sep='\t')
    _dict = {}
    for r in range(mapping_df.shape[0]):
        _dict[mapping_df[0][r]] = mapping_df[1][r]

    # Do mapping
    subcats = subcats.map(lambda x: _dict[x]).unique().tolist()
    subcat2idx = {}
    for i, cat in enumerate(subcats):
        subcat2idx[cat] = i

    tokenized_subcats = tokenizer(subcats, return_tensors='pt', padding=True, max_length=10)
    subcat_embedding = torch.mean(model(**tokenized_subcats)[0], dim=1)
    subcat_embedding = subcat_embedding.half()
    # print(df[2].shape)
    # print(model(**tokenized_subcats)[0].shape)
    # print(subcat_embedding.shape)

    # res = torch.stack(df[2].map(lambda x: subcat_embedding[subcat2idx[_dict[x]]]).tolist())

    res = (subcat2idx, subcat_embedding, df[2].map(lambda x: _dict[x]))
    # print(res.shape)
    # print(res.shape)
    # print(res[0:10])

    with open(pickle_file, 'wb') as f:
        pickle.dump(res, f, protocol=4)

def get_abs_embedding_bert(df, tokenizer, model, pickle_file, abs_size=30, n=30):
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
    # print(abs.head(10))
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
    # for dt1 in ['demo', 'large']:
    for dt1 in ['demo']:
        # for dt2 in ['train', 'dev' ,'test']:
        for dt2 in ['dev', 'train']:
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
            get_category_embedding_bert2(df, tokenizer, model, category_pickle_file, n=100)
            print(f"{dt1}-{dt2}/ get category embedding done")

            # get subcategory embedding
            subcategory_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_subcategory2.pickle'
            get_subcategory_embedding_bert2(df, tokenizer, model, subcategory_pickle_file, n=100)
            print(f"{dt1}-{dt2}/ get subcategory embedding done")

            # get abstract embedding
            abs_size = 30
            abs_pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/large_bert_abs_{title_size}.pickle'
            # get_abs_embedding_bert(df, tokenizer, model, abs_pickle_file, abs_size=abs_size, n=100)
            print(f"{dt1}-{dt2}/ get abstract embedding done")

            # checks
            # with open(title_pickle_file, 'rb') as tf:
            #     nid2idx, title_embeddings = pickle.load(tf)
            # with open(category_pickle_file, 'rb') as cf:
            #     cats = pickle.load(cf)
            # with open(subcategory_pickle_file, 'rb') as sf:
            #     subcats = pickle.load(cf)
            # print(nid2idx.shape)
            # print(cats.shape)
            # print(subcats.shape)
            print(f"{dt1}-{dt2} done")

