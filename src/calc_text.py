import math
import json
import fire


def evaluate_text_predictions(path, item_path):
    """
    Evaluate text generation predictions using NDCG and HR metrics.
    
    Args:
        path: Path(s) to JSON files containing prediction results
        item_path: Path to item list file (items.txt)
    
    Outputs:
        Prints NDCG and HR metrics for various cutoff values (1, 3, 5, 10, 20, 50)
    """
    # Convert path to list if it's a single string
    if type(path) != list:
        path = [path]
    
    # Normalize item_path by removing .txt extension if present
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]

    # Load item names from the item list file
    with open(f"{item_path}.txt", "r") as f:
        items = f.readlines()

    # Extract item names from the file (use second column if available, else first)
    item_names = []
    for line in items:
        parts = line.split("\t")
        if len(parts) >= 2:
            item_names.append(parts[1].strip())
        else:
            item_names.append(parts[0].strip())

    # Create item ID mapping (ID = index in the item list)
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = {}
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:
            item_dict[item_names[i]].append(item_ids[i])

    # Define evaluation cutoff values for top-k metrics
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1

    # Process each prediction file
    for p in path:
        with open(p, "r") as f:
            test_data = json.load(f)

        # Extract predicted items (as list of strings) from each sample
        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        valid_topk = None
        all_ndcg = None
        all_hr = None
        
        # Determine valid cutoff values based on beam size (number of predictions)
        for sample in text:
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                all_ndcg = [0.0] * len(valid_topk)
                all_hr = [0.0] * len(valid_topk)

        # Calculate metrics for each sample
        for index, sample in enumerate(text):
            # Extract the target/ground truth item from the test data
            if isinstance(test_data[index]["output"], list):
                target_item = test_data[index]["output"][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]["output"].strip(" \n\"")

            # Find the rank of the target item in predictions (1-based index)
            min_id = 1000000
            for i in range(len(sample)):
                if sample[i] == target_item:
                    min_id = i
                    break

            # Update NDCG and HR metrics if target found within top-k
            for idx, topk in enumerate(valid_topk):
                if min_id < topk:
                    # NDCG uses log discount: 1 / log(rank + 2)
                    all_ndcg[idx] += 1 / math.log(min_id + 2)
                    # HR is binary hit (found in top-k or not)
                    all_hr[idx] += 1

        # Normalize metrics by number of samples and print results
        print(n_beam)
        print(valid_topk)
        ndcg = [v / len(text) / (1.0 / math.log(2)) for v in all_ndcg]
        hr = [v / len(text) for v in all_hr]
        print(f"NDCG:\t{ndcg}")
        print(f"HR\t{hr}")


if __name__ == "__main__":
    fire.Fire(evaluate_text_predictions)
