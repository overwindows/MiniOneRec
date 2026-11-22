######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/utils/evaluation.py
# - This file includes utility functions useful for evaluation
#
# Version: 1.0
#######################################################################################################

import numpy as np


def mrr_score(y_true, y_score):
    """
    Score the results based on Mean Reciprocal Ranking.
    :param y_true: ground truth values
    :param y_score: predicted values
    :return: MRR score
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    """
    Score the results based on Discounted Cumulative Gain at k.
    :param y_true: ground truth values
    :param y_score: predicted values
    :param k: how many recommendation would be considered
    :return: DCG score
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """
    Score the results based on Normalized Discounted Cumulative Gain at k.
    :param y_true: ground truth values
    :param y_score: predicted values
    :param k: how many recommendation would be considered
    :return: NDCG score
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best
