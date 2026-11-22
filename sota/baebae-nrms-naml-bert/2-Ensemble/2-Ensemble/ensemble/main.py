######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/ensemble/train.py
# - The main file for ensembling
#
# Version: 1.0
#######################################################################################################
import sys
sys.path.append('./')

import os
import pickle
import pandas as pd
import numpy as np
import torch
import csv
from sklearn.metrics import roc_auc_score

from utils import *
from evaluation import ndcg_score, mrr_score
from tqdm import tqdm
import click

@click.command()
@click.option('--dev_test', type=str, default='test')
@click.option('--model_comb', type=str, default='')
@click.option('--weight_comb', type=str, default='')
@click.option('--merge_type', type=str, default='rank')
@click.option('--out_path', type=str, default='../out')
@click.option('--idx', type=int, default=10)
def main(dev_test, model_comb, weight_comb, merge_type, out_path, idx):
    # Get list of models for ensembling
    model_list = model_comb.split(',')
    if weight_comb == '999':
        weight_list = [1./len(model_list) for _ in range(len(model_list))]
    else:
        weight_list = [float(w) for w in weight_comb.split(',')]

    assert len(model_list) == len(weight_list)
    max_model_num = 10
    str_setting = f'{idx},{dev_test.upper()},'
    for i in range(max_model_num):
        if i < len(model_list):
            str_setting += model_list[i]+','
        else:
            str_setting += ','
    str_setting+= f',{merge_type}'    
    
    '''
    Load data
    '''
    behavior_file = f'/data/mind/MINDlarge_{dev_test}/behaviors.tsv'
    bdf = pd.read_csv(behavior_file, sep='\t', header=None, quoting=csv.QUOTE_NONE)
    labels = get_labels(bdf, dev_test) # zeros for test data

    data_path = f'/data/mind/ensemble/{dev_test.upper()}'
    score_files = [os.path.join( data_path, model+'_score.pickle') for model in model_list]

    dicts = []
    for score_file in score_files:
        # print(score_file)
        with open(score_file, 'rb') as f:
            new_dict = (pickle.load(f))
        if 'TANR' in score_file:
            _dict = {}
            for key in new_dict:
                _dict[int(key.item())] = np.array(new_dict[key])
            new_dict = _dict
        else:
            _dict = {}
            for key in new_dict:
                _dict[int(key)] = np.array(new_dict[key])
            new_dict = _dict
            
        dicts.append(new_dict)

    '''
    Scoring
    '''
    out_path = os.path.join(out_path, dev_test.upper())
    output_file = os.path.join(out_path, f"my_prediction{idx}.txt")
    log_file = os.path.join(out_path, "log.txt")
    os.makedirs(out_path, exist_ok=True)

    tot_num = len(dicts[0])
    metrics = {metric: 0. for metric in ['auc', 'mrr', 'ndcg@5', 'ndcg@10']}
    scores_dict = {}
    scores_pickle_file = os.path.join(out_path,f"ensemble_scores{idx}.pickle")
    with open(output_file, 'w') as f:
        for j in tqdm(range(1, tot_num+1)):
            gt_labels_j = labels[j-1]
            scores_j = [np.array(dicts[method_id][j]) for method_id in range(len(model_list))]
            final_score_j = get_final_score(scores_j, by=merge_type, weight_list=weight_list) # performing enemble
            scores_dict[j] = final_score_j

            # Get ranks for prediction.txt
            argmax_idx = (-final_score_j).argsort()
            ranks = np.empty_like(argmax_idx)
            if ranks.shape[0] == 1:
                ranks[argmax_idx] = np.array([1])
            else:
                ranks[argmax_idx] = np.arange(1, final_score_j.shape[0] + 1)
            ranks_str = ','.join([str(r) for r in list(ranks)])
            f.write(f"{j} [{ranks_str}]\n")

            # print(gt_labels_j)
            # print(final_score_j)

            # Skip scoring process for the test data
            if dev_test == 'test':
                continue 
            for metric, _ in metrics.items():
                if metric == 'auc':
                    score = roc_auc_score(gt_labels_j, final_score_j)
                    metrics[metric] += score
                elif metric == 'mrr':
                    score = mrr_score(gt_labels_j, final_score_j)
                    metrics[metric] += score
                elif metric.startswith('ndcg'):
                    k = int(metric.split('@')[1])
                    score = ndcg_score(gt_labels_j, final_score_j, k=k)
                    metrics[metric] += score
    result = ''
    with open(log_file, 'a') as lf:
        for metric, _ in metrics.items():
            metrics[metric] /= tot_num
            result += (f"{metrics[metric]:.4f},")
        print(result)
        lf.write(str_setting+',')
        lf.write(result+'\n')
    print()
    
    with open(scores_pickle_file, 'wb') as spf:
        pickle.dump(scores_dict, spf, protocol=4)

if __name__ == '__main__':
    main()
