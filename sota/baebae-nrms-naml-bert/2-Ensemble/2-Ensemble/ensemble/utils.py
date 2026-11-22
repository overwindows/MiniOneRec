######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/ensemble/utils.py
# - Utility functions for performing ensemble.
#
# Version: 1.0
#######################################################################################################
import sys
import pandas as pd
import numpy as np
from subprocess import Popen, PIPE

def get_labels(bdf, dev_test):
    """
    Get labels of each impressions
    :param bdf: behavior df
    :param dev_test: dev file or test file
    :return: list of labels
    """
    imprs = bdf[4].map(lambda x: x.split(' '))
    if dev_test == 'dev':
        labels = imprs.map(lambda x: [float(xx.split('-')[1]) for xx in x])
    elif dev_test == 'test':
        # dummy labels for the test dataset
        labels = imprs.map(lambda x: [float(0) for _ in x])
    return labels.tolist()


def _softmax(arr):
    exp_arr = np.exp(arr)
    sum_exp_arr = np.sum(exp_arr)
    res = exp_arr / sum_exp_arr
    return res

def _z_norm(arr):
    """
    Performing z normalization for the given array
    :param arr:
    :return: normalized array
    """
    n_arr = np.array(arr)
    mu = np.mean(arr)
    sigma = np.std(arr)
    return (n_arr-mu)/sigma

def _get_ranks(score):
    argmax_idx = score.argsort()
    ranks = np.empty_like(argmax_idx)
    if ranks.shape[0] == 1:
        ranks[argmax_idx] = np.array([1])
    else:
        ranks[argmax_idx] = np.arange(1, score.shape[0]+1)
    return ranks

def get_final_score(scores, by='rank', weight_list=[]):
    assert len(weight_list ) > 0
    if len(np.array(scores).shape) == 1:
       return np.array([1])
    if by == 'rank':
        ranks = np.array([_get_ranks(score) for score in scores])
        res = np.zeros(ranks.shape[1])
        for r in range(ranks.shape[0]):
            res += weight_list[r] * ranks[r]

    if by == 'score':
        soft_scores = np.array([_softmax(score) for score in scores])
        # print(soft_scores)
        # print(soft_scores.shape)
        res = np.zeros(soft_scores.shape[1])
        for r in range(soft_scores.shape[0]):
            res += weight_list[r] * soft_scores[r]

    if by == 'z_score':
        normed_scores = np.array([_z_norm(score) for score in scores])
        # print(soft_scores)
        # print(soft_scores.shape)
        res = np.zeros(normed_scores.shape[1])
        for r in range(normed_scores.shape[0]):
            res += weight_list[r] * normed_scores[r]

    if by == 'max':
        normed_scores = np.array([_softmax(score) for score in scores])
        res = np.zeros(normed_scores.shape[1])
        for c in range(normed_scores.shape[1]):
            res[c] = np.max(normed_scores[:,c])

    return res
   

def _process_command_list_subprocess(cmd_list, ERROR_LOG_FILE):
    prev_stdin = None
    count = 0
    for str_cmd in cmd_list:
        count += 1
        try:
            print("START[{}/{}]: {}".format(count, len(cmd_list), str_cmd))

            p = Popen(str_cmd, stdout=None, stderr=PIPE, shell=True)
            output, error = p.communicate()
        except KeyboardInterrupt:
            sys.exit()

        if p.returncode != 0:
            with open(ERROR_LOG_FILE, 'a') as ff:
                ff.write("Command: " + str_cmd + '\n')
                ff.write(str(error.decode('utf-8')) + '\n\n')


