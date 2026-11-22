######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/test.py
# - This file evaluates a trained model on datasets.
#
# Version: 1.0
#######################################################################################################

import os
import pickle
import time

import click
import numpy as np
import torch
from tqdm import tqdm

from NRMS_NAML.models.model import NRMS
from NRMS_NAML.utils.config import prepare_config
from NRMS_NAML.utils.dataloader import DataSetTest
from NRMS_NAML.utils.selector_test import NewsSelectorTest

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_data_paths(path):
    """
    Set paths for data.
    :param path: directory path
    :return: file paths
    """
    paths = {'behaviors': os.path.join(path, 'behaviors.tsv'),
             'news': os.path.join(path, 'news.tsv'),
             'entity': os.path.join(path, 'entity_embedding.vec'),
             'relation': os.path.join(path, 'relation_embedding.vec')}
    return paths


def set_util_paths(path):
    """
    Set paths for utility files.
    :param path: directory path
    :return: file paths
    """
    paths = {'embedding': os.path.join(path, 'embedding.npy'),
             'uid2index': os.path.join(path, 'uid2index.pkl'),
             'word_dict': os.path.join(path, 'word_dict.pkl'),
             'cat_dict': os.path.join(path, 'cat_dict.pkl'),
             'subcat_dict': os.path.join(path, 'subcat_dict.pkl')}
    return paths


def load_dict(file_path):
    """
    Read a dictionary.
    :param file_path: path of file to be read
    :return: dictionary
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def set_seed(seed):
    """
    Set random seed for torch and numpy.
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


@click.command()
@click.option('--data_path', type=str, default='/data/mind')
@click.option('--data', type=str, default='large')
@click.option('--out_path', type=str, default='../out')
@click.option('--config_path', type=str, default='./config.yaml')
@click.option('--epoch', type=int, default=7)
def main(data_path, data, out_path, config_path, epoch):
    start_time = time.time()

    # read paths
    test_data = os.path.join(data_path, f'MIND{data}_test')
    util_data = os.path.join(data_path, 'utils')

    test_paths = set_data_paths(test_data)
    util_paths = set_util_paths(util_data)

    # read config file
    config = prepare_config(config_path,
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'],
                            catDict_file=util_paths['cat_dict'],
                            subcatDict_file=util_paths['subcat_dict'])

    # set out path
    out_path = os.path.join(out_path, f'MIND{data}_results')
    os.makedirs(out_path, exist_ok=True)
    bucket_size = config['bucket_size']
    his_size = config['his_size']
    title_size = config['title_size']
    abs_size = config['abstract_size']
    test_pickle_path = os.path.join(test_data, f'dataset_bucket{bucket_size}_his{his_size}_title{title_size}_abs{abs_size}.pickle')
    model_path = os.path.join(out_path, f'model-{epoch}.pt')

    # set random seed
    seed = config['seed']
    set_seed(seed)

    # load dictionaries
    word2idx = load_dict(config['wordDict_file'])
    uid2idx = load_dict(config['userDict_file'])
    cat2idx = load_dict(config['catDict_file'])
    subcat2idx = load_dict(config['subcatDict_file'])
    cat2idx[''] = 0
    subcat2idx[''] = 0

    # load datasets and define dataloaders
    if os.path.exists(test_pickle_path):
        with open(test_pickle_path, 'rb') as f:
            test_set = pickle.load(f)
    else:
        test_selector = NewsSelectorTest(data_type1=data, data_type2='test',
                                         num_pop=20,
                                         num_fresh=20,
                                         bucket_size=bucket_size)
        test_set = DataSetTest(test_paths['news'], test_paths['behaviors'],
                               word2idx=word2idx, uid2idx=uid2idx,
                               cat2idx=cat2idx, subcat2idx=subcat2idx,
                               selector=test_selector, config=config,
                               label_known=False)
        with open(test_pickle_path, 'wb') as f:
            pickle.dump(test_set, f)

    test_impr_idx, test_his_title, test_impr_title, test_label, test_pop_title, test_fresh_title = \
        test_set.raw_impr_idxs, test_set.histories_title, test_set.imprs_title, \
        test_set.labels, test_set.pops_title, test_set.freshs_title
    test_his_abs, test_impr_abs, test_pop_abs, test_fresh_abs =\
        test_set.histories_abs, test_set.imprs_abs, test_set.pops_abs, test_set.freshs_abs
    test_his_cat, test_impr_cat, test_pop_cat, test_fresh_cat =\
        test_set.histories_cat, test_set.imprs_cat, test_set.pops_cat, test_set.freshs_cat
    test_his_subcat, test_impr_subcat, test_pop_subcat, test_fresh_subcat =\
        test_set.histories_subcat, test_set.imprs_subcat, test_set.pops_subcat, test_set.freshs_subcat

    # define model, optimizer, loss
    word2vec_emb = np.load(config['wordEmb_file'])
    model = NRMS(config, word2vec_emb, len(cat2idx), len(subcat2idx)).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(f'[{time.time()-start_time:5.2f} Sec] Ready for test...')

    scores_dict = {}
    with open(os.path.join(out_path, f'test-prediction-{epoch}.txt'), 'w') as f:
        for j in tqdm(range(len(test_impr_title)),
                      desc=f'Epoch {epoch:3d} test evaluation',
                      total=len(test_impr_title)):
            impr_idx_j = test_impr_idx[j]
            test_his_title_j = torch.tensor(test_his_title[j]).long().to(
                DEVICE).unsqueeze(0)
            test_pop_title_j = torch.tensor(test_pop_title[j]).long().to(
                DEVICE).unsqueeze(0)
            test_fresh_title_j = torch.tensor(test_fresh_title[j]).long().to(
                DEVICE).unsqueeze(0)
            test_pop_title_j = test_pop_title_j[:, :config['pop'], :]
            test_fresh_title_j = test_fresh_title_j[:, :config['fresh'], :]
            test_cand_title_j = torch.tensor(test_impr_title[j]).long().to(
                DEVICE).unsqueeze(0)
            test_global_title_j = torch.cat(
                (test_pop_title_j, test_fresh_title_j), dim=1)

            if config['aux']:
                test_his_abs_j = torch.tensor(test_his_abs[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_pop_abs_j = torch.tensor(test_pop_abs[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_fresh_abs_j = torch.tensor(test_fresh_abs[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_pop_abs_j = test_pop_abs_j[:, :config['pop'], :]
                test_fresh_abs_j = test_fresh_abs_j[:, :config['fresh'], :]
                test_cand_abs_j = torch.tensor(test_impr_abs[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_global_abs_j = torch.cat(
                    (test_pop_abs_j, test_fresh_abs_j), dim=1)

                test_his_cat_j = torch.tensor(test_his_cat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_pop_cat_j = torch.tensor(test_pop_cat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_fresh_cat_j = torch.tensor(test_fresh_cat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_pop_cat_j = test_pop_cat_j[:, :config['pop']]
                test_fresh_cat_j = test_fresh_cat_j[:, :config['fresh']]
                test_cand_cat_j = torch.tensor(test_impr_cat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_global_cat_j = torch.cat(
                    (test_pop_cat_j, test_fresh_cat_j), dim=1)

                test_his_subcat_j = torch.tensor(test_his_subcat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_pop_subcat_j = torch.tensor(test_pop_subcat[j]).long().to(
                    DEVICE).unsqueeze(0)
                test_fresh_subcat_j = torch.tensor(
                    test_fresh_subcat[j]).long().to(DEVICE).unsqueeze(0)
                test_pop_subcat_j = test_pop_subcat_j[:, :config['pop']]
                test_fresh_subcat_j = test_fresh_subcat_j[:, :config['fresh']]
                test_cand_subcat_j = torch.tensor(
                    test_impr_subcat[j]).long().to(DEVICE).unsqueeze(0)
                test_global_subcat_j = torch.cat(
                    (test_pop_subcat_j, test_fresh_subcat_j), dim=1)

                test_his_j = [test_his_title_j, test_his_abs_j, test_his_cat_j,
                              test_his_subcat_j]
                test_cand_j = [test_cand_title_j, test_cand_abs_j,
                               test_cand_cat_j, test_cand_subcat_j]
                test_global_j = [test_global_title_j, test_global_abs_j,
                                 test_global_cat_j, test_global_subcat_j]
            else:
                test_his_j = test_his_title_j
                test_cand_j = test_cand_title_j
                test_global_j = test_global_title_j

            if config['global']:
                test_user_out_j = model((test_his_j, test_global_j),
                                        source='pgt')
            else:
                test_user_out_j = model(test_his_j, source='history')

            test_cand_out_j = model(test_cand_j, source='candidate')

            scores_j = torch.matmul(test_cand_out_j,
                                    test_user_out_j.unsqueeze(2)).squeeze()
            scores_j = scores_j.detach().cpu().numpy()
            scores_dict[impr_idx_j] = scores_j
            argmax_idx = (-scores_j).argsort()
            ranks = np.empty_like(argmax_idx)
            if ranks.shape[0] == 1:
                ranks[argmax_idx] = np.array([1])
            else:
                ranks[argmax_idx] = np.arange(1, scores_j.shape[0] + 1)
            ranks_str = ','.join([str(r) for r in list(ranks)])
            f.write(f'{impr_idx_j} [{ranks_str}]\n')

    # scores to file
    with open(os.path.join(out_path, f'test-scores-{epoch}.pickle'), 'wb') as f:
        pickle.dump(scores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()