######################################################################################################
# Mind 2020 competition
# Authors: Hyunsik Jeon(jeon185@snu.ac.kr), SeungCheol Park(ant6si@snu.ac.kr),
#          Yuna Bae(yunabae482@gmail.com), U kang(ukang@snu.ac.kr)
# File: src/NRMS_NAML/cat2dict.py
# - This file finds sets of categories/subcategories from datasets
#
# Version: 1.0
#######################################################################################################

import os
import pickle

import click
from tqdm import tqdm


def get_set(path):
    """
    Find all categories/sub-categories from the file.
    :param path: file path
    :return: sets of category and sub-category
    """
    cat_set = set()
    subcat_set = set()

    with open(path, 'r') as rd:
        for line in tqdm(rd, desc='Init news'):
            nid, cat, subcat, title, abstract, \
            url, title_ent, abstract_ent = line.strip("\n").split('\t')

            cat_set.add(cat)
            subcat_set.add(subcat)

    return cat_set, subcat_set


@click.command()
@click.option('--data_path', type=str, default='/data/mind')
@click.option('--out_path', type=str, default='../out')
def main(data_path, out_path):
    # file paths
    trn_demo_data = os.path.join(data_path, 'MINDdemo_train')
    vld_demo_data = os.path.join(data_path, 'MINDdemo_dev')
    trn_large_data = os.path.join(data_path, 'MINDlarge_train')
    vld_large_data = os.path.join(data_path, 'MINDlarge_dev')

    # define out paths
    util_data = os.path.join(data_path, 'utils')
    cat_path = os.path.join(util_data, 'cat_dict.pkl')
    subcat_path = os.path.join(util_data, 'subcat_dict.pkl')

    cat_set = set()
    subcat_set = set()

    cat, subcat = get_set(os.path.join(trn_demo_data, 'news.tsv'))
    cat_set.update(cat)
    subcat_set.update(subcat)

    cat, subcat = get_set(os.path.join(vld_demo_data, 'news.tsv'))
    cat_set.update(cat)
    subcat_set.update(subcat)

    cat, subcat = get_set(os.path.join(trn_large_data, 'news.tsv'))
    cat_set.update(cat)
    subcat_set.update(subcat)

    cat, subcat = get_set(os.path.join(vld_large_data, 'news.tsv'))
    cat_set.update(cat)
    subcat_set.update(subcat)

    cat_dict = {c: i for (i, c) in enumerate(cat_set, start=1)}
    subcat_dict = {c: i for (i, c) in enumerate(subcat_set, start=1)}

    with open(cat_path, 'wb') as pkl:
        pickle.dump(cat_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    with open(subcat_path, 'wb') as pkl:
        pickle.dump(subcat_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()