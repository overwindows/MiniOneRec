import os, pickle
import torch

title_pickle_file = f"/data/mind/MINDdemo_train/BERT/large_bert_title_30.pickle"
assert os.path.isfile(title_pickle_file)
# if not os.path.isfile(pickle_file):
#     get_word_embedding_bert(news_file, pickle_file, title_size=self.title_size, n=30)
with open(title_pickle_file, 'rb') as f:
    nid2index, title_embeddings = pickle.load(f)
title_embeddings.requires_grad = False

abs_pickle_file = "/data/mind/MINDdemo_train/BERT/large_bert_abs_30.pickle"
assert os.path.isfile(abs_pickle_file)
with open(abs_pickle_file, 'rb') as f:
    abs_embeddings = pickle.load(f)
abs_embeddings.requires_grad = False
#
# subcat_pickle_file = "/data/mind/MINDdemo_dev/BERT/large_bert_subcategory.pickle"
# assert os.path.isfile(subcat_pickle_file)
# with open(subcat_pickle_file, 'rb') as f:
#     subcat_embeddings = pickle.load(f)
# subcat_embeddings.requires_grad = False

print(title_embeddings.shape)
print(abs_embeddings.shape)

print(title_embeddings[0])
print(abs_embeddings[0])
z = torch.zeros(1,30,1024).half()
abs_embeddings = torch.cat((z, abs_embeddings))
print(abs_embeddings[0:3])
print(abs_embeddings.shape)

