import os
import pickle
import time
import sys

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.nrms import NRMS
from utils.config import prepare_config
from utils.dataloader import DataSetTrn, DataSetTest
from utils.evaluation import ndcg_score, mrr_score
from utils.selector import NewsSelector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_data_paths(path):
    paths = {'behaviors': os.path.join(path, 'behaviors.tsv'),
             'news': os.path.join(path, 'news.tsv'),
             'entity': os.path.join(path, 'entity_embedding.vec'),
             'relation': os.path.join(path, 'relation_embedding.vec')}
    return paths


def set_util_paths(path):
    paths = {'embedding': os.path.join(path, 'embedding.npy'),
             'uid2index': os.path.join(path, 'uid2index.pkl'),
             'word_dict': os.path.join(path, 'word_dict.pkl'),
             'cat_dict': os.path.join(path, 'cat_dict.pkl'),
             'subcat_dict': os.path.join(path, 'subcat_dict.pkl')}
    return paths


def load_dict(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@click.command()
@click.option('--data_path', type=str, default='/data/mind')
@click.option('--data', type=str, default='large')
@click.option('--out_path', type=str, default='../out')
@click.option('--config_path', type=str, default='./config.yaml')
@click.option('--eval_every', type=int, default=3)
def main(data_path, data, out_path, config_path, eval_every):
    start_time = time.time()

    # read paths
    trn_data = os.path.join(data_path, f'MIND{data}_train')
    # vld_data = os.path.join(data_path, f'MIND{data}_dev')
    vld_data = os.path.join(data_path, f'MIND{data}_train')
    util_data = os.path.join(data_path, 'utils')

    trn_paths = set_data_paths(trn_data)
    vld_paths = set_data_paths(vld_data)
    util_paths = set_util_paths(util_data)

    model_path = None

    # read configuration file
    config = prepare_config(config_path,
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'],
                            catDict_file=util_paths['cat_dict'],
                            subcatDict_file=util_paths['subcat_dict'])

    # out path
    num_pop = config['pop']
    num_fresh = config['fresh']
    aux = config['aux']
    text_self_attn_layer = config['text_self_attn_layer']
    history_self_attn_layer = config['history_self_attn_layer']
    out_path = os.path.join(out_path, f'Final_MIND{data}_NRMS_aux{aux}_text{text_self_attn_layer}_history{history_self_attn_layer}')
    os.makedirs(out_path, exist_ok=True)
    bucket_size = config['bucket_size']
    his_size = config['his_size']
    title_size = config['title_size']
    abs_size = config['abstract_size']
    trn_pickle_path = os.path.join(trn_data, f'dataset_bucket{bucket_size}_his{his_size}_title{title_size}_abs{abs_size}.pickle')
    # vld_pickle_path = os.path.join(vld_data, f'dataset_bucket{bucket_size}_his{his_size}_title{title_size}_abs{abs_size}.pickle')
    vld_pickle_path = os.path.join(trn_data, f'list_dataset_bucket{bucket_size}_his{his_size}_title{title_size}_abs{abs_size}.pickle')
    verbose_path = os.path.join(out_path, 'verbose.txt')
    # model_path = os.path.join(out_path, 'model-30.pt')

    # set
    seed = config['seed']
    set_seed(seed)
    epochs = config['epochs']
    metrics = {metric: 0. for metric in config['metrics']}

    # load dictionaries
    word2idx = load_dict(config['wordDict_file'])
    uid2idx = load_dict(config['userDict_file'])
    cat2idx = load_dict(config['catDict_file'])
    subcat2idx = load_dict(config['subcatDict_file'])
    cat2idx[''] = 0
    subcat2idx[''] = 0

    # load datasets and define dataloaders
    if os.path.exists(trn_pickle_path):
        with open(trn_pickle_path, 'rb') as f:
            trn_set = pickle.load(f)
    else:
        trn_selector = NewsSelector(data_type1=data, data_type2='train',
                                    num_pop=20,
                                    num_fresh=20,
                                    bucket_size=bucket_size)
        trn_set = DataSetTrn(trn_paths['news'], trn_paths['behaviors'],
                             word2idx=word2idx, uid2idx=uid2idx,
                             cat2idx=cat2idx, subcat2idx=subcat2idx,
                             selector=trn_selector, config=config)
        with open(trn_pickle_path, 'wb') as f:
            pickle.dump(trn_set, f)

    if os.path.exists(vld_pickle_path):
        with open(vld_pickle_path, 'rb') as f:
            vld_set = pickle.load(f)
    else:
        vld_selector = NewsSelector(data_type1=data, data_type2='train',    ######################
                                    num_pop=20,
                                    num_fresh=20,
                                    bucket_size=bucket_size)
        vld_set = DataSetTest(vld_paths['news'], vld_paths['behaviors'],
                              word2idx=word2idx, uid2idx=uid2idx,
                              cat2idx=cat2idx, subcat2idx=subcat2idx,
                              selector=vld_selector, config=config,
                              label_known=True)
        with open(vld_pickle_path, 'wb') as f:
            pickle.dump(vld_set, f)
    exit()

    trn_loader = DataLoader(trn_set, batch_size=config['batch_size'],
                            shuffle=True, num_workers=8)

    vld_impr_idx, vld_his_title, vld_impr_title, vld_label, vld_pop_title, vld_fresh_title = \
        vld_set.raw_impr_idxs, vld_set.histories_title, vld_set.imprs_title, \
        vld_set.labels, vld_set.pops_title, vld_set.freshs_title
    vld_his_abs, vld_impr_abs, vld_pop_abs, vld_fresh_abs =\
        vld_set.histories_abs, vld_set.imprs_abs, vld_set.pops_abs, vld_set.freshs_abs
    vld_his_cat, vld_impr_cat, vld_pop_cat, vld_fresh_cat =\
        vld_set.histories_cat, vld_set.imprs_cat, vld_set.pops_cat, vld_set.freshs_cat
    vld_his_subcat, vld_impr_subcat, vld_pop_subcat, vld_fresh_subcat =\
        vld_set.histories_subcat, vld_set.imprs_subcat, vld_set.pops_subcat, vld_set.freshs_subcat

    # define models, optimizer, loss
    word2vec_emb = np.load(config['wordEmb_file'])
    model = NRMS(config, word2vec_emb, len(cat2idx), len(subcat2idx)).to(DEVICE)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']),
                           weight_decay=float(config['weight_decay']))

    criterion = nn.CrossEntropyLoss()

    print(f'[{time.time()-start_time:5.2f} Sec] Ready for training...')

    # train and evaluate
    for epoch in range(1, epochs+1):
        model.train()
        start_time = time.time()
        batch_loss = 0.
        '''
        training
        '''
        for i, (trn_his_title, trn_pos_title, trn_neg_title, trn_pop_title, trn_fresh_title,
                trn_his_abs, trn_pos_abs, trn_neg_abs, trn_pop_abs, trn_fresh_abs,
                trn_his_cat, trn_pos_cat, trn_neg_cat, trn_pop_cat, trn_fresh_cat,
                trn_his_subcat, trn_pos_subcat, trn_neg_subcat, trn_pop_subcat, trn_fresh_subcat) \
                in tqdm(enumerate(trn_loader), desc=f'Epoch {epoch:3d} training', total=len(trn_loader)):
            # ready for training
            optimizer.zero_grad()

            # prepare data
            trn_his_title, trn_pos_title, trn_neg_title, trn_pop_title, trn_fresh_title = \
                trn_his_title.to(DEVICE), trn_pos_title.to(DEVICE), trn_neg_title.to(DEVICE),\
                trn_pop_title.to(DEVICE), trn_fresh_title.to(DEVICE)

            trn_pop_title = trn_pop_title[:, :config['pop'], :]
            trn_fresh_title = trn_fresh_title[:, :config['fresh'], :]
            trn_cand_title = torch.cat((trn_pos_title, trn_neg_title), dim=1)
            trn_global_title = torch.cat((trn_pop_title, trn_fresh_title), dim=1)
            trn_gt = torch.zeros(size=(trn_cand_title.shape[0],)).long().to(DEVICE)

            if config['aux']:
                trn_his_abs, trn_pos_abs, trn_neg_abs, trn_pop_abs, trn_fresh_abs = \
                    trn_his_abs.to(DEVICE), trn_pos_abs.to(DEVICE), trn_neg_abs.to(DEVICE),\
                    trn_pop_abs.to(DEVICE), trn_fresh_abs.to(DEVICE)
                trn_his_cat, trn_pos_cat, trn_neg_cat, trn_pop_cat, trn_fresh_cat = \
                    trn_his_cat.to(DEVICE), trn_pos_cat.to(DEVICE), trn_neg_cat.to(DEVICE),\
                    trn_pop_cat.to(DEVICE), trn_fresh_cat.to(DEVICE)
                trn_his_subcat, trn_pos_subcat, trn_neg_subcat, trn_pop_subcat, trn_fresh_subcat = \
                    trn_his_subcat.to(DEVICE), trn_pos_subcat.to(DEVICE), trn_neg_subcat.to(DEVICE),\
                    trn_pop_subcat.to(DEVICE), trn_fresh_subcat.to(DEVICE)

                trn_pop_abs = trn_pop_abs[:, :config['pop'], :]
                trn_fresh_abs = trn_fresh_abs[:, :config['fresh'], :]
                trn_cand_abs = torch.cat((trn_pos_abs, trn_neg_abs), dim=1)
                trn_global_abs = torch.cat((trn_pop_abs, trn_fresh_abs), dim=1)

                trn_pop_cat = trn_pop_cat[:, :config['pop']]
                trn_fresh_cat = trn_fresh_cat[:, :config['fresh']]
                trn_cand_cat = torch.cat((trn_pos_cat, trn_neg_cat), dim=1)
                trn_global_cat = torch.cat((trn_pop_cat, trn_fresh_cat), dim=1)

                trn_pop_subcat = trn_pop_subcat[:, :config['pop']]
                trn_fresh_subcat = trn_fresh_subcat[:, :config['fresh']]
                trn_cand_subcat = torch.cat((trn_pos_subcat, trn_neg_subcat), dim=1)
                trn_global_subcat = torch.cat((trn_pop_subcat, trn_fresh_subcat), dim=1)

                trn_his = [trn_his_title, trn_his_abs, trn_his_cat, trn_his_subcat]
                trn_cand = [trn_cand_title, trn_cand_abs, trn_cand_cat, trn_cand_subcat]
                trn_global = [trn_global_title, trn_global_abs, trn_global_cat, trn_global_subcat]

            else:
                trn_his = trn_his_title
                trn_cand = trn_cand_title
                trn_global = trn_global_title

            # inference
            if config['global']:
                trn_user_out = model((trn_his, trn_global), source='pgt')
            else:
                trn_user_out = model(trn_his, source='history')
            trn_cand_out = model(trn_cand, source='candidate')
            prob = torch.matmul(trn_cand_out, trn_user_out.unsqueeze(2)).squeeze()

            # training
            loss = criterion(prob, trn_gt)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()

        inter_time = time.time()
        epoch_loss = batch_loss/(i+1)

        # if epoch % eval_every != 0:
        if epoch < 6:
            result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f}Sec]' \
                     f', TrnLoss:{epoch_loss:.4f}'
            print(result)
            continue

        '''
        model save
        '''
        torch.save(model.state_dict(), os.path.join(out_path, f'model-{epoch}.pt'))

        '''
        evaluation
        '''
        model.eval()
        scores_dict = {}
        with open(os.path.join(out_path, f'vld-prediction-{epoch}.txt'), 'w') as f:
            for j in tqdm(range(len(vld_impr_title)), desc=f'Epoch {epoch:3d} vld evaluation', total=len(vld_impr_title)):
                impr_idx_j = vld_impr_idx[j]
                vld_his_title_j = torch.tensor(vld_his_title[j]).long().to(DEVICE).unsqueeze(0)
                vld_pop_title_j = torch.tensor(vld_pop_title[j]).long().to(DEVICE).unsqueeze(0)
                vld_fresh_title_j = torch.tensor(vld_fresh_title[j]).long().to(DEVICE).unsqueeze(0)
                vld_pop_title_j = vld_pop_title_j[:, :config['pop'], :]
                vld_fresh_title_j = vld_fresh_title_j[:, :config['fresh'], :]
                vld_cand_title_j = torch.tensor(vld_impr_title[j]).long().to(DEVICE).unsqueeze(0)
                vld_global_title_j = torch.cat((vld_pop_title_j, vld_fresh_title_j), dim=1)

                if config['aux']:
                    vld_his_abs_j = torch.tensor(vld_his_abs[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_abs_j = torch.tensor(vld_pop_abs[j]).long().to(DEVICE).unsqueeze(0)
                    vld_fresh_abs_j = torch.tensor(vld_fresh_abs[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_abs_j = vld_pop_abs_j[:, :config['pop'], :]
                    vld_fresh_abs_j = vld_fresh_abs_j[:, :config['fresh'], :]
                    vld_cand_abs_j = torch.tensor(vld_impr_abs[j]).long().to(DEVICE).unsqueeze(0)
                    vld_global_abs_j = torch.cat((vld_pop_abs_j, vld_fresh_abs_j), dim=1)

                    vld_his_cat_j = torch.tensor(vld_his_cat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_cat_j = torch.tensor(vld_pop_cat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_fresh_cat_j = torch.tensor(vld_fresh_cat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_cat_j = vld_pop_cat_j[:, :config['pop']]
                    vld_fresh_cat_j = vld_fresh_cat_j[:, :config['fresh']]
                    vld_cand_cat_j = torch.tensor(vld_impr_cat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_global_cat_j = torch.cat((vld_pop_cat_j, vld_fresh_cat_j), dim=1)

                    vld_his_subcat_j = torch.tensor(vld_his_subcat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_subcat_j = torch.tensor(vld_pop_subcat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_fresh_subcat_j = torch.tensor(vld_fresh_subcat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_pop_subcat_j = vld_pop_subcat_j[:, :config['pop']]
                    vld_fresh_subcat_j = vld_fresh_subcat_j[:, :config['fresh']]
                    vld_cand_subcat_j = torch.tensor(vld_impr_subcat[j]).long().to(DEVICE).unsqueeze(0)
                    vld_global_subcat_j = torch.cat((vld_pop_subcat_j, vld_fresh_subcat_j), dim=1)

                    vld_his_j = [vld_his_title_j, vld_his_abs_j, vld_his_cat_j, vld_his_subcat_j]
                    vld_cand_j = [vld_cand_title_j, vld_cand_abs_j, vld_cand_cat_j, vld_cand_subcat_j]
                    vld_global_j = [vld_global_title_j, vld_global_abs_j, vld_global_cat_j, vld_global_subcat_j]
                else:
                    vld_his_j = vld_his_title_j
                    vld_cand_j = vld_cand_title_j
                    vld_global_j = vld_global_title_j

                if config['global']:
                    vld_user_out_j = model((vld_his_j, vld_global_j), source='pgt')
                else:
                    vld_user_out_j = model(vld_his_j, source='history')

                vld_cand_out_j = model(vld_cand_j, source='candidate')

                scores_j = torch.matmul(vld_cand_out_j, vld_user_out_j.unsqueeze(2)).squeeze()
                scores_j = scores_j.detach().cpu().numpy()
                scores_dict[impr_idx_j] = scores_j
                argmax_idx = (-scores_j).argsort()
                ranks = np.empty_like(argmax_idx)
                ranks[argmax_idx] = np.arange(1, scores_j.shape[0]+1)
                ranks_str = ','.join([str(r) for r in list(ranks)])
                f.write(f'{impr_idx_j} [{ranks_str}]\n')

                vld_gt_j = np.array(vld_label[j])

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

        # scores to file
        with open(os.path.join(out_path, f'vld-scores-{epoch}.pickle'), 'wb') as f:
            pickle.dump(scores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        for metric, _ in metrics.items():
            metrics[metric] /= len(vld_impr_title)

        end_time = time.time()

        result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f} / {end_time - inter_time:5.2f} Sec]' \
                 f', TrnLoss:{epoch_loss:.4f}, '
        for enum, (metric, _) in enumerate(metrics.items(), start=1):
            result += f'{metric}:{metrics[metric]:.4f}'
            if enum < len(metrics):
                result += ', '
        print(result)
        with open(verbose_path, 'a') as f:
            f.write(result + '\n')


if __name__ == '__main__':
    main()