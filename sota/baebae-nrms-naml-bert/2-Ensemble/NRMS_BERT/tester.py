######################################################################################################
# Mind 2020
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_BERT/tester.py
# - The tester file for evaluation using saved model.
#
# Version: 1.0
#######################################################################################################

import sys, os
sys.path.append('./')
import pickle
import time

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict

from models.nrms2 import NRMS
from utils.config import prepare_config
from utils.dataloader import DataSetTrn, DataSetTest
from utils.evaluation import ndcg_score, mrr_score
from utils.selector import NewsSelector



def set_data_paths(path):
    """
    set path for data
    :param path
    :return: paths
    """
    paths = {'behaviors': os.path.join(path, 'behaviors.tsv'),
             'news': os.path.join(path, 'news.tsv'),
             'entity': os.path.join(path, 'entity_embedding.vec'),
             'relation': os.path.join(path, 'relation_embedding.vec')}
    return paths


def set_util_paths(path):
    """
    set path for util files
    :param path
    :return: paths
    """
    paths = {'embedding': os.path.join(path, 'embedding.npy'),
             'uid2index': os.path.join(path, 'uid2index.pkl'),
             'word_dict': os.path.join(path, 'word_dict.pkl')}
    return paths


def load_dict(file_path):
    """
    load dictionary file
    :param file_path:
    :return: loaded dictionary file included in the pickle file
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def set_seed(seed):
    """
    set random seed for numpy and pytorch
    :param seed:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


@click.command()
@click.option('--gpu', type=str, default='3')
@click.option('--data_path', type=str, default='/data/mind')
@click.option('--data', type=str, default='large')
@click.option('--out_path', type=str, default='./out/evaluation/nrms_fixed_bert')
@click.option('--config_path', type=str, default='./config.yaml')
@click.option('--test_dev', type=str, default='test')
@click.option('--eval_every', type=int, default=2)
def main(gpu, data_path, data, out_path, config_path, test_dev, eval_every):
    ##########################   CHECK SETTING     ##############################
    data = 'large' # large or demo
    gpu = 3 # set a gpu to use
    test_dev = 'test' # test or dev

    # Set model path
    epoch = 6
    model_pth = f'/home/ant6si/MIND/EXP2/mind2020/NRMS_BERT_Fixed_all_features/out'+\
                f'/exp20/MINDlarge_dev_pop0_fresh0/nrms-bert-{epoch}.pth'

    # Set output path (prediction will be saved in thie path)
    out_path = f'/home/ant6si/MIND/EXP2/mind2020/NRMS_BERT_Fixed_all_features/out/evaluation-{epoch}'

    #############################################################################

    # ignore outputs without the first process
    DEVICE = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    # read paths
    vld_data = os.path.join(data_path, f'MIND{data}_{test_dev}')
    util_data = os.path.join(data_path, 'utils')

    vld_paths = set_data_paths(vld_data)
    util_paths = set_util_paths(util_data)

    # read configuration file
    config = prepare_config(config_path,
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'])

    # out path
    num_global = config['pop']
    num_fresh = config['fresh']
    out_path = os.path.join(out_path, f'MIND{data}_{test_dev}_pop{num_global}_fresh{num_fresh}')
    os.makedirs(out_path, exist_ok=True)

    # set
    seed = config['seed']
    set_seed(seed)
    metrics = {metric: 0. for metric in config['metrics']}

    # load dictionaries
    word2idx = load_dict(config['wordDict_file'])
    uid2idx = load_dict(config['userDict_file'])

    vld_selector = NewsSelector(data_type1=data, data_type2=test_dev,
                                num_pop=20,
                                num_fresh=20)
    vld_set = DataSetTest(vld_paths['news'], vld_paths['behaviors'],
                          word2idx=word2idx, uid2idx=uid2idx,
                          selector=vld_selector, config=config,
                          label_known=True)

    vld_loader = DataLoader(vld_set, batch_size=1, num_workers=1, shuffle=False)

    word2vec_emb = np.load(config['wordEmb_file'])

    # Load model
    # Load model using code below because of the apex
    model = NRMS(config, word2vec_emb).to(DEVICE)
    loaded_model = torch.load(model_pth, map_location=DEVICE)
    new_dict = OrderedDict()
    for key in loaded_model.keys():
        new_dict[key[7:]] = loaded_model[key]
    model.load_state_dict(new_dict)
    model.eval()

    scores_df = {}
    pickle_file = os.path.join(out_path, 'scores.pickle')
    with open(pickle_file, 'wb') as sf:
        pickle.dump(scores_df, sf, protocol=4)

    # Start evaluation
    with open(os.path.join(out_path, f'prediction.txt'), 'w') as f:
        for j, (impr_idx_j, vld_his_j, vld_cand_j, vld_label_j, vld_pop_j, vld_fresh_j) \
            in tqdm(enumerate(vld_loader), desc='Evaluation', total=len(vld_loader)):
            # Get model output
            impr_idx_j = impr_idx_j.item()
            vld_global_j = {}
            for key in vld_his_j.keys():
                vld_his_j[key], vld_pop_j[key], vld_fresh_j[key], vld_cand_j[key] = \
                vld_his_j[key].to(DEVICE), vld_pop_j[key].to(DEVICE), \
                vld_fresh_j[key].to(DEVICE), vld_cand_j[key].to(DEVICE)

                vld_pop_j[key] = vld_pop_j[key][:, :config['pop'], :]
                vld_fresh_j[key] = vld_fresh_j[key][:, :config['fresh'], :]
                vld_global_j[key] = torch.cat((vld_pop_j[key], vld_fresh_j[key]), dim=1)
            if config['global']:
                vld_user_out_j = model((vld_his_j, vld_global_j), source='pgt')
            else:
                vld_user_out_j = model(vld_his_j, source='history')
            vld_cand_out_j = model(vld_cand_j, source='candidate')

            # Get model output end
            scores_j = torch.matmul(vld_cand_out_j, vld_user_out_j.unsqueeze(2)).squeeze()
            scores_j = scores_j.detach().cpu().numpy()
            scores_df[impr_idx_j] = scores_j
            argmax_idx = (-scores_j).argsort()
            ranks = np.empty_like(argmax_idx)
            if ranks.shape[0] == 1:
                ranks[argmax_idx] = np.array([1])
            else:
                ranks[argmax_idx] = np.arange(1, scores_j.shape[0] + 1)
            ranks_str = ','.join([str(r) for r in list(ranks)])
            f.write(f'{impr_idx_j} [{ranks_str}]\n')
            
            if test_dev == 'dev':
                vld_gt_j = np.array(vld_label_j)
                # vld_gt_j = np.array(vld_label[j])

                for metric, _ in metrics.items():
                    if metric == 'auc':
                        score = roc_auc_score(vld_gt_j, scores_j)
                        metrics[metric] += score
                    elif metric == 'mrr':
                        score = mrr_score(vld_gt_j, scores_j)
                        metrics[metric] += score
                    elif metric.startswith('ndcg'):  # format like: ndcg@5;10
                        k = int(metric.split('@')[1])
                        score = ndcg_score(vld_gt_j, scores_j, k=k)
                        metrics[metric] += score
            
    if test_dev == 'dev':
        for metric, _ in metrics.items():
            metrics[metric] /= len(vld_loader) # len(vld_impr)

    pickle_file = os.path.join(out_path, 'scores.pickle')
    with open(pickle_file, 'wb') as sf:
        pickle.dump(scores_df, sf, protocol=4)
    
    end_time = time.time()
    result = ''
    for enum, (metric, _) in enumerate(metrics.items(), start=1):
        result += f'{metric}:{metrics[metric]:.4f}'
        if enum < len(metrics):
            result += ', '
    print(result)

if __name__ == '__main__':
    main()
