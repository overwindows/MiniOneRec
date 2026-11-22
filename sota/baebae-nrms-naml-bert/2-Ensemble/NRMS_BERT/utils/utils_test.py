######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/utils/utils_test.py
# - utility functions for the tester.
#
# Version: 1.0
#######################################################################################################
import os, sys
sys.path.append('./')
sys.path.append('../')

import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import itertools
import torch

"""
Functions for parallelizing init_behaviors
"""
def parallelize_dataframe_test(df, func, num_cores=20, class_pointer=None):
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

################# For TEST dataset ###################################

def get_imprs(x, dt):
    res = []
    for nid in x.split(' '):
        res.append(dt.nid2idx[nid])
    return res

def get_labels(x):
    x = [int(0) for i in x.split(' ')]
    return x

###########################################################

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

def process_behaviors_test(df, dt):
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
