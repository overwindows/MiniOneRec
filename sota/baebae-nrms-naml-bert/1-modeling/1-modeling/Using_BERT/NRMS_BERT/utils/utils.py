# from multiprocessing import Pool
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import itertools
import torch

"""
Functions for parallelizing init_behaviors
"""
def parallelize_dataframe(df, func, num_cores=20, class_pointer=None):
    df_split = np.array_split(df, num_cores)
    pool = mp.Pool(num_cores)
    # df = pd.concat(pool.map(func, df_split))
    if class_pointer == None:
        res = pd.concat(pool.starmap(func, df_split))
    else:
        res = pd.concat(pool.starmap(func, zip(df_split, itertools.repeat(class_pointer))))
    pool.close()
    pool.join()
    return res


def _p_hist(x, dt):
    # print(x)
    if type(x) == np.float:
        return [0] * dt.his_size
    else:
        x = [dt.nid2idx[i] for i in x.split(' ')]
        x = [0] * (dt.his_size-len(x)) + x[:dt.his_size]
        return x
def get_imprs(x, dt):
    x = [dt.nid2idx[i.split("-")[0]] for i in x.split(' ')]
    return x
def get_labels(x):
    x = [int(i.split('-')[1]) for i in x.split(' ')]
    return x
def get_uidx(x, dt):
    if x in dt.uid2idx:
        return dt.uid2idx[x]
    else:
        return 0
def get_pop(x, dt): # x: time
    pop = [dt.nid2idx[i] for i in dt.selector.get_pop_recommended(x)]
    return pop
def get_fresh(x, dt):
    fresh = [dt.nid2idx[i] for i in dt.selector.get_fresh(x)]
    return fresh

def process_behaviors(df, dt):
    new_df = {}
    # print(df[4])
    new_df['imprs'] = df[4].map(lambda x:get_imprs(x, dt))
    new_df['labels'] = df[4].map(lambda x:get_labels(x))
    new_df['raw_impr_indexes'] = list(df[0])
    new_df['uindexes'] = df[1].map(lambda x:get_uidx(x, dt))
    new_df['times'] = df[2]
    new_df['pops'] = df[2].map(lambda x: get_pop(x, dt))
    new_df['freshs'] = df[2].map(lambda x: get_fresh(x, dt))
    # new_df['impr_indexes'] = range(df.shape[0])
    new_df['histories'] = df[3].map(lambda x: _p_hist(x,dt))
    new_df = pd.DataFrame.from_dict(new_df)
    return new_df

def process_news(df, class_pt):
    new_df = {}
    new_df['out'] = df[3].map(lambda x: class_pt.tokenizer(x,return_tensors="pt", padding=True,
                                                            truncation=True, max_length=30))
    new_df['input_ids'] =new_df['out'].map(lambda x: x['input_ids'])
    new_df['attention_mask'] = new_df['out'].map(lambda x: x['attention_mask'])
    new_df.drop('out',axis=1)
    return new_df

# """
# Functions for parallelizing evaluation
# """
# import os
# from utils.evaluation import ndcg_score, mrr_score
# from sklearn.metrics import roc_auc_score
#
# # metrics, vlds = {'his', 'pop', 'fresh', 'impr_idx', 'impr', 'label'}, config, DEVICE, model
# def parallelize_eval(_list, func, metrics, vlds, config, DEVICE, model, num_cores=20):
#     list_split = np.array_split(_list, num_cores)
#     pool = mp.Pool(num_cores)
#     # df = pd.concat(pool.map(func, df_split))
#     res = pd.concat(pool.starmap(func, zip(list_split,
#                                            itertools.repeat(metrics),
#                                            itertools.repeat(vlds),
#                                            itertools.repeat(config),
#                                            itertools.repeat(DEVICE),
#                                            itertools.repeat(model)
#                                            )))
#     pool.close()
#     pool.join()
#     for metric, _ in metrics.items():
#         metrics[metric] = res[metric].sum()
#
#     return metrics
#
# def evaluation(j_list, metrics, vlds, config, DEVICE, model):
#     res_df = pd.DataFrame()
#     for j in j_list:
#         impr_idx_j = vlds['impr_idx'][j]
#         vld_his_j, vld_pop_j, vld_fresh_j, vld_global_j = {}, {}, {}, {}
#         for key in ['input_ids', 'attention_mask']:
#             # print("=====================111111111================================")
#             # print("j", j)
#             # print("vlds['his'][j]", vlds['his'][j][key])
#             # (vlds['his'][j][key]).to(DEVICE)
#             # print("========  TO  DEVICE DONE =========")
#             # (vlds['his'][j][key]).to(DEVICE).unsqueeze(0)
#             # print("========= Unsqueeze DONE ===========")
#
#             # vld_his_j[key] = (vlds['his'][j][key]).to(DEVICE).unsqueeze(0)  # TODO: Here
#             # # print("=====================2222222222================================")
#             # vld_pop_j[key] = (vlds['pop'][j][key]).to(DEVICE).unsqueeze(0)  # TODO: Here
#             # # print("=====================333333333================================")
#             # vld_fresh_j[key] = (vlds['fresh'][j][key]).to(DEVICE).unsqueeze(0)  # TODO: Here
#             # # print("=====================444444444================================")
#             vld_his_j[key] = (vlds['his'][j][key]).unsqueeze(0)  # TODO: Here
#             vld_pop_j[key] = (vlds['pop'][j][key]).unsqueeze(0)  # TODO: Here
#             vld_fresh_j[key] = (vlds['fresh'][j][key]).unsqueeze(0)  # TODO: Here
#             vld_pop_j[key] = vld_pop_j[key][:, :config['pop'], :]
#             vld_fresh_j[key] = vld_fresh_j[key][:, :config['fresh'], :]
#             vld_global_j[key] = torch.cat((vld_pop_j[key], vld_fresh_j[key]), dim=1)
#
#         if config['global']:
#             vld_user_out_j = model((vld_his_j, vld_global_j), source='pgt')
#         else:
#             vld_user_out_j = model(vld_his_j, source='history')
#         vld_cand_j = {}
#         for key in ['input_ids', 'attention_mask']:
#             # vld_cand_j[key] = (vlds['impr'][j][key]).to(DEVICE).unsqueeze(0)  # TODO: Here
#             vld_cand_j[key] = (vlds['impr'][j][key]).unsqueeze(0)  # TODO: Here
#
#         vld_cand_out_j = model(vld_cand_j, source='candidate')
#
#         scores_j = torch.matmul(vld_cand_out_j, vld_user_out_j.unsqueeze(2)).squeeze()
#         scores_j = scores_j.detach().cpu().numpy()
#         argmax_idx = (-scores_j).argsort()
#         ranks = np.empty_like(argmax_idx)
#         ranks[argmax_idx] = np.arange(1, scores_j.shape[0] + 1)
#         ranks_str = ','.join([str(r) for r in list(ranks)])
#         # f.write(f'{impr_idx_j} [{ranks_str}]\n')
#
#         vld_gt_j = np.array(vlds['label'][j])
#
#         new_df = {'idx':j}
#
#         for metric, _ in metrics.items():
#             if metric == 'auc':
#                 score = roc_auc_score(vld_gt_j, scores_j)
#                 new_df[metric] = score
#             elif metric == 'mrr':
#                 score = mrr_score(vld_gt_j, scores_j)
#                 new_df[metric] = score
#             elif metric.startswith('ndcg'):  # format like: ndcg@5;10
#                 k = int(metric.split('@')[1])
#                 score = ndcg_score(vld_gt_j, scores_j, k=k)
#                 new_df[metric] = score
#         new_df["impr_idx_j"] = impr_idx_j
#         new_df["ranks_str"] = ranks_str
#         res_df.append(new_df, ignore_index=True)
#     return res_df
