import sys, os
import pickle
import time

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from apex.parallel import DistributedDataParallel as DDP

torch.distributed.init_process_group(backend='nccl', init_method='env://')


from models.nrms2 import NRMS # two layers
from utils.config import prepare_config
from utils.dataloader import DataSetTrn, DataSetTest
from utils.evaluation import ndcg_score, mrr_score
from utils.selector import NewsSelector



def set_data_paths(path):
    paths = {'behaviors': os.path.join(path, 'behaviors.tsv'),
             'news': os.path.join(path, 'news.tsv'),
             'entity': os.path.join(path, 'entity_embedding.vec'),
             'relation': os.path.join(path, 'relation_embedding.vec')}
    return paths


def set_util_paths(path):
    paths = {'embedding': os.path.join(path, 'embedding.npy'),
             'uid2index': os.path.join(path, 'uid2index.pkl'),
             'word_dict': os.path.join(path, 'word_dict.pkl')}
    return paths


def load_dict(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@click.command()
@click.option('--hyp', type=int, default=0)
@click.option('--lr', type=float, default=0.0001)
@click.option('--dropout', type=float, default=0.2)
@click.option('--word_dim', type=int, default=300)
@click.option('--exp_num', type=str, default='0')
@click.option('--gpus', type=str, default='0')
@click.option('--local_rank', type=int, default=0)
@click.option('--data_path', type=str, default='/data/mind')
@click.option('--data', type=str, default='demo')
@click.option('--out_path', type=str, default='./out/nrms_bert_fixed_all_features_')
@click.option('--config_path', type=str, default='./config.yaml')
@click.option('--eval_every', type=int, default=1)
def main(hyp, lr, dropout, word_dim, exp_num, gpus, local_rank, data_path, data, out_path, config_path, eval_every):
    # ignore outputs without the first process
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')

    gpu_list = gpus.strip().split(',')
    DEVICE = torch.device(f'cuda:{gpu_list[local_rank]}' if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    # read paths
    trn_data = os.path.join(data_path, f'MIND{data}_train')
    vld_data = os.path.join(data_path, f'MIND{data}_dev')
    util_data = os.path.join(data_path, 'utils')

    trn_paths = set_data_paths(trn_data)
    vld_paths = set_data_paths(vld_data)
    util_paths = set_util_paths(util_data)

    trn_pickle_path = os.path.join(trn_data, 'dataset_bert.pickle')
    vld_pickle_path = os.path.join(vld_data, 'dataset_bert.pickle')

    # read configuration file
    config = prepare_config(config_path,
                            wordEmb_file=util_paths['embedding'],
                            wordDict_file=util_paths['word_dict'],
                            userDict_file=util_paths['uid2index'])

    ############ for hyperparameter searching #####################
    if hyp == 1:
        config['learning_rate'] = lr
        config['dropout'] = dropout
        config['word_emb_dim'] = word_dim
    ########################################################

    # out path
    num_global = config['pop']
    num_fresh = config['fresh']
    out_path += str(exp_num)
    out_path = os.path.join(out_path, f'MIND{data}_dev_pop{num_global}_fresh{num_fresh}')
    os.makedirs(out_path, exist_ok=True)

    log_file = out_path + '/log.txt'
    with open(log_file, 'a') as lf:
        lf.write(f"\n{lr},{dropout},{word_dim}\n")

    # set
    seed = config['seed']
    set_seed(seed)
    epochs = config['epochs']
    metrics = {metric: 0. for metric in config['metrics']}

    # load dictionaries
    word2idx = load_dict(config['wordDict_file'])
    uid2idx = load_dict(config['userDict_file'])

    # load datasets and define dataloaders
    # if os.path.exists(trn_pickle_path):
    #     with open(trn_pickle_path, 'rb') as f:
    #         trn_set = pickle.load(f)
    if False:
        pass
    else:
        trn_selector = NewsSelector(data_type1=data, data_type2='train',
                                    num_pop=20,
                                    num_fresh=20)
        trn_set = DataSetTrn(trn_paths['news'], trn_paths['behaviors'],
                             word2idx=word2idx, uid2idx=uid2idx,
                             selector=trn_selector, config=config)
        # with open(trn_pickle_path, 'wb') as f:
        #     pickle.dump(trn_set, f, protocol=4)

    # if os.path.exists(vld_pickle_path):
    #     with open(vld_pickle_path, 'rb') as f:
    #         vld_set = pickle.load(f)
    if False:
        pass
    else:
        vld_selector = NewsSelector(data_type1=data, data_type2='dev',
                                    num_pop=20,
                                    num_fresh=20)
        vld_set = DataSetTest(vld_paths['news'], vld_paths['behaviors'],
                              word2idx=word2idx, uid2idx=uid2idx,
                              selector=vld_selector, config=config,
                              label_known=True)

        # with open(vld_pickle_path, 'wb') as f:
        #     pickle.dump(vld_set, f, protocol=4)

    trn_sampler = DistributedSampler(trn_set)
    trn_loader = DataLoader(trn_set, batch_size=config['batch_size'],
                            num_workers=8, sampler=trn_sampler)

    # vld_impr_idx, vld_his, vld_impr, vld_label, vld_pop, vld_fresh =\
    #     vld_set.raw_impr_idxs, vld_set.histories_words, vld_set.imprs_words,\
    #     vld_set.labels, vld_set.pops_words, vld_set.freshs_words
    vld_loader = DataLoader(vld_set, batch_size=1, num_workers=1, shuffle=False)
    # define models, optimizer, loss
    # TODO: w2v --> BERT model
    word2vec_emb = np.load(config['wordEmb_file'])
    model = NRMS(config, word2vec_emb).to(DEVICE)
    model = DDP(model, delay_allreduce=True)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']),
                           weight_decay=float(config['weight_decay']))
    criterion = nn.CrossEntropyLoss()
    print(f'[{time.time()-start_time:5.2f} Sec] Ready for training...')

    # train and evaluate
    for epoch in range(1, epochs+1):
        start_time = time.time()
        batch_loss = 0.
        '''
        training
        '''
        for i, (trn_his, trn_pos, trn_neg, trn_pop, trn_fresh) \
                in tqdm(enumerate(trn_loader), desc='Training', total=len(trn_loader)):
            # ready for training
            model.train()
            optimizer.zero_grad()

            trn_cand, trn_global = {}, {}
            for key in trn_his.keys():
                # prepare data
                trn_his[key], trn_pos[key], trn_neg[key], trn_pop[key], trn_fresh[key] = \
                    trn_his[key].to(DEVICE), trn_pos[key].to(DEVICE), trn_neg[key].to(DEVICE),\
                    trn_pop[key].to(DEVICE), trn_fresh[key].to(DEVICE)
                trn_pop[key] = trn_pop[key][:, :config['pop'], :]
                trn_fresh[key] = trn_fresh[key][:, :config['fresh'], :]
                trn_cand[key] = torch.cat((trn_pos[key], trn_neg[key]), dim=1)
                trn_global[key] = torch.cat((trn_pop[key], trn_fresh[key]), dim=1)
            trn_gt = torch.zeros(size=(trn_cand['title'].shape[0],)).long().to(DEVICE)

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

        if epoch<6 or epoch % eval_every != 0:
            result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f}Sec]' \
                     f', TrnLoss:{epoch_loss:.4f}'
            print(result)
            continue

        '''
        evaluation
        '''
        model.eval()
        if local_rank==0:
            with open(os.path.join(out_path, f'prediction-{epoch}.txt'), 'w') as f:
                # for j in tqdm(range(len(vld_impr)), desc='Evaluation', total=len(vld_impr)):
                for j, (impr_idx_j, vld_his_j, vld_cand_j, vld_label_j, vld_pop_j, vld_fresh_j) \
                    in tqdm(enumerate(vld_loader), desc='Evaluation', total=len(vld_loader)):

                    # Get model output
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
                    argmax_idx = (-scores_j).argsort()
                    ranks = np.empty_like(argmax_idx)
                    if ranks.shape[0] == 1:
                        ranks[argmax_idx] = np.array([1])
                    else:
                        ranks[argmax_idx] = np.arange(1, scores_j.shape[0]+1)
                    ranks_str = ','.join([str(r) for r in list(ranks)])
                    f.write(f'{impr_idx_j.item()} [{ranks_str}]\n')
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

            for metric, _ in metrics.items():
                metrics[metric] /= len(vld_loader) # len(vld_impr)

            end_time = time.time()

            result = f'Epoch {epoch:3d} [{inter_time - start_time:5.2f} / {end_time - inter_time:5.2f} Sec]' \
                     f', TrnLoss:{epoch_loss:.4f}, '
            for enum, (metric, _) in enumerate(metrics.items(), start=1):
                result += f'{metric}:{metrics[metric]:.4f}'
                if enum < len(metrics):
                    result += ', '
            print(result)
            with open(log_file, 'a') as lf:
                lf.write(result+"\n")

if __name__ == '__main__':
    main()