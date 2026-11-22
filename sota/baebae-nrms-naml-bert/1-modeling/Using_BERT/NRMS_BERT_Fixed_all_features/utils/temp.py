import pandas as pd
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

dt1s = ['large', 'demo']
dt2s = ['train', 'dev']
total_df = pd.DataFrame()
for dt1, dt2 in zip(dt1s, dt2s):
    news_file = f'/data/mind/MIND{dt1}_{dt2}/news.tsv'
    df = pd.read_csv(news_file, sep='\t', header=None)
    total_df = total_df.append(df)

news_file = '/data/mind/MINDlarge_test/news.tsv'
df = pd.read_csv(news_file, sep='\t', header=None)
total_df = total_df.append(df)


total_abs = total_df[4]
total_abs = total_abs.fillna('')
# print(total_abs[0:50])
total_abs = total_abs.map(lambda x: x[0:500] if (len(x)> 500) else x) #[CLS], [SEP]
tokenized_abs = total_abs.map(lambda x: len(tokenizer(x)['input_ids']))
print(tokenized_abs.value_counts)
print(tokenized_abs.decribe())

# print("============ Number of cats: ", total_cats.shape[0], "=======================")
# for r in range(total_cats.shape[0]):
#     print(total_cats.index[r], total_cats[r])


# print("============ Number of subcats: ", total_subcats.shape[0], "=======================")
# for r in range(total_subcats.shape[0]):
#     print(total_subcats.index[r], total_subcats[r])




