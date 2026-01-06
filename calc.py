import os
import fire
import math
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_recommendation_metrics(path, item_path):
    """
    Calculate NDCG and Hit Rate (HR) metrics for recommendation results.

    Args:
        path: Single result JSON file path or list of paths
        item_path: Path to item catalog file (.txt)

    Returns:
        None (prints metrics to stdout)
    """
    if not isinstance(path, list):
        path = [path]

    # Validate input files exist
    for p in path:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Result file not found: {p}")

    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    if not os.path.exists(f"{item_path}.txt"):
        raise FileNotFoundError(f"Item catalog file not found: {item_path}.txt")

    unknown_items_count = 0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        for index, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[index]['output']) == list:
                target_item = test_data[index]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]['output'].strip(" \n\"")
            minID = 1000000
            for i in range(len(sample)):
                if sample[i] not in item_dict:
                    unknown_items_count += 1
                    print(f"WARNING: Unknown item in predictions: '{sample[i]}' (target: '{target_item}')")

                if sample[i] == target_item:
                    minID = i
                    break
            
            for index, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[index] = ALLNDCG[index] + (1 / math.log(minID + 2))
                    ALLHR[index] = ALLHR[index] + 1
        print(f"\n{'='*60}")
        print(f"Results for: {p}")
        print(f"{'='*60}")
        print(f"Number of beams: {n_beam}")
        print(f"Valid top-k values: {valid_topk}")
        print(f"Total samples: {len(text)}")
        print(f"Unknown items count: {unknown_items_count}")
        print(f"\nNDCG@k: {ALLNDCG / len(text) / (1.0 / math.log(2))}")
        print(f"HR@k:   {ALLHR / len(text)}")
        print(f"{'='*60}\n")

if __name__ == '__main__':
    fire.Fire(calculate_recommendation_metrics)
