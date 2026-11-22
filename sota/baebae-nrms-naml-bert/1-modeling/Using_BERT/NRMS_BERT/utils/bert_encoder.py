import pandas as pd
import torch
import itertools
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pickle
import os

def get_word_embedding_bert(dt1='large', dt2='train', title_size=30, n=30):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # data_file = f'/data/mind/MIND{dt1}_{dt2}/integrated_news.tsv'
    data_file = f'/data/mind/MIND{dt1}_{dt2}/news.tsv' #TODO: Here
    pickle_file = f'/data/mind/MIND{dt1}_{dt2}/BERT/bert_title2.pickle' #TODO: Here
    # 1: integrated_news.tsv 2: news.tsv 3: news + [sep]
    df = pd.read_csv(data_file, sep='\t', header=None) #TODO: Here

    nid2index = {}
    titles = ['']

    # Read dataframe and make the list of titles
    for r in tqdm(range(df.shape[0])):
        # nid = df['nid'][r]
        # title = df['title'][r]
        nid = df[0][r] #TODO: Here
        title = df[3][r] #TODO: Here

        if nid in nid2index:
            continue

        nid2index[nid] = len(nid2index) + 1
        titles.append(title) # TODO: Here

    tokenized_titles = tokenizer(titles, max_length=title_size, padding=True, truncation=True,
                                add_special_tokens=False , return_tensors='pt')

    def get_tokenized_titles(t, _from, _to):
        res = {}
        for key in t:
            res[key] = t[key][_from:_to]
        return res

    # Make partitioned pickle files because of the memory overflow error
    num_news = len(titles)
    n=30
    gap = num_news// (n-1)
    ranges = [i * gap for i in range(n)]
    ranges.append(num_news)
    for i in tqdm(range(n)):
        news_title_embeds = (model(**get_tokenized_titles(tokenized_titles, ranges[i], ranges[i+1] ))[0])
        with open(pickle_file +f'{i}', 'wb') as f:
            pickle.dump(news_title_embeds, f)


    # Generate a integrated pickle file
    for i in range(n):
        file_name = pickle_file + f'{i}'
        with open(file_name, 'rb') as f:
            if i == 0:
                z = pickle.load(f)
            else:
                z = torch.cat((z,pickle.load(f)))
    res_file = [nid2index, z]
    with open(pickle_file, 'wb') as f:
        pickle.dump(res_file, f, protocol=4) # protocol 4 enables to handle large files (> 4GB)

    # Check the final file
    with open(pickle_file, 'rb') as f:
        b, a = pickle.load(f)
    print(a.shape)
    print(a[0])
    print(len(b.keys()))

    # Remove redundant files
    for i in range(n):
        os.remove(pickle_file+f'{i}')


if __name__=='__main__':
    get_word_embedding_bert(dt1='large', dt2='dev', title_size=30, n=30)
