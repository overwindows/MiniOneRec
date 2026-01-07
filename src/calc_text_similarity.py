import math
import json
import fire
from difflib import SequenceMatcher
from tqdm import tqdm


def calculate_similarity(str1, str2):
    """
    Calculate fuzzy string similarity score using SequenceMatcher.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize strings: lowercase and strip whitespace
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()

    # Calculate similarity ratio
    ratio = SequenceMatcher(None, s1, s2).ratio()
    return ratio


def find_best_matching_item(prediction, item_names):
    """
    Find the best matching item in the catalog for a given prediction.
    Always returns the most similar item, regardless of similarity score.

    Args:
        prediction: Generated text prediction
        item_names: List of all item names in the catalog

    Returns:
        Tuple of (best_match_index, best_similarity_score)
    """
    best_match_idx = 0
    best_similarity = 0.0

    # Optimization: check exact match first (much faster)
    prediction_normalized = prediction.lower().strip()
    for idx, item_name in enumerate(item_names):
        if item_name.lower().strip() == prediction_normalized:
            return idx, 1.0

    # Fuzzy matching: find the most similar item
    for idx, item_name in enumerate(item_names):
        # Early exit if we already have a perfect match
        if best_similarity >= 1.0:
            break

        similarity = calculate_similarity(prediction, item_name)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_idx = idx

    return best_match_idx, best_similarity


def evaluate_text_predictions(path, item_path, similarity_threshold=0.85, use_similarity=True):
    """
    Evaluate text generation predictions using NDCG and HR metrics.
    Supports both exact matching and similarity-based catalog matching.

    Evaluation approaches:
    1. Exact matching (use_similarity=False):
       - Directly compare prediction text with ground truth text
       - Match if strings are identical

    2. Similarity-based catalog matching (use_similarity=True) - RECOMMENDED:
       - For each prediction, find the most similar item in the full catalog using fuzzy matching
       - Compare the matched item's ID with ground truth item's ID
       - This simulates real-world retrieval: map generated text → catalog item → check correctness

    Args:
        path: Path(s) to JSON files containing prediction results
        item_path: Path to item list file (items.txt)
        similarity_threshold: Minimum similarity score (0.0-1.0) for catalog matching (default: 0.85)
        use_similarity: If True, use catalog matching; if False, use exact text matching (default: True)

    Outputs:
        Prints NDCG and HR metrics for various cutoff values (1, 3, 5, 10, 20, 50)
    """
    # Convert path to list if it's a single string
    if type(path) != list:
        path = [path]

    # Load item names from the item list file
    # item_path can be with or without .txt extension
    if not item_path.endswith(".txt"):
        # Try to find the file with suffix pattern
        import glob
        import os
        dir_path = os.path.dirname(item_path) if os.path.dirname(item_path) else "."
        base_name = os.path.basename(item_path)
        matching_files = glob.glob(f"{dir_path}/{base_name}*.txt")
        if matching_files:
            item_path = matching_files[0]
        else:
            item_path = f"{item_path}.txt"

    with open(item_path, "r") as f:
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

    print(f"\n{'='*60}")
    print(f"Evaluation Mode: {'SIMILARITY-BASED' if use_similarity else 'EXACT MATCH'}")
    if use_similarity:
        print(f"Similarity Threshold: {similarity_threshold:.2f}")
    print(f"{'='*60}\n")

    # Process each prediction file
    for p in path:
        with open(p, "r") as f:
            test_data = json.load(f)

        # Extract predicted items (as list of strings) from each sample
        text = [[_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        valid_topk = None
        all_ndcg = None
        all_hr = None

        # Tracking statistics
        exact_matches = 0
        fuzzy_matches = 0
        no_matches = 0

        # Determine valid cutoff values based on beam size (number of predictions)
        for sample in text:
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                all_ndcg = [0.0] * len(valid_topk)
                all_hr = [0.0] * len(valid_topk)

        # Calculate metrics for each sample
        if use_similarity:
            print(f"\n⚠️  NOTE: Similarity-based catalog matching is computationally expensive!")
            print(f"   Processing {len(text)} samples × {n_beam} predictions × {len(item_names)} catalog items")
            print(f"   This may take several minutes... Please be patient.\n")

        for index, sample in enumerate(tqdm(text, desc="Evaluating samples", disable=not use_similarity)):
            # Extract the target/ground truth item from the test data
            if isinstance(test_data[index]["output"], list):
                target_item = test_data[index]["output"][0].strip("\"").strip(" ")
            else:
                target_item = test_data[index]["output"].strip(" \n\"")

            # Get ground truth item ID
            target_item_ids = item_dict.get(target_item, None)
            if target_item_ids is None:
                # Ground truth item not in catalog - skip this sample
                no_matches += 1
                continue

            # Find the rank where we match the ground truth item
            min_id = 1000000
            match_type = "none"

            if use_similarity:
                # NEW APPROACH: For each prediction, find best matching item in catalog using similarity
                # Then check if matched item's ID equals ground truth item's ID
                for i in range(len(sample)):
                    prediction = sample[i]

                    # Step 1: Find best matching item in the catalog using fuzzy matching
                    best_match_idx, best_similarity = find_best_matching_item(
                        prediction, item_names
                    )

                    # Step 2: Check if the matched item ID equals ground truth item ID (exact match)
                    if best_match_idx in target_item_ids:
                        min_id = i
                        match_type = "fuzzy"
                        break
            else:
                # Exact matching: prediction text must exactly match ground truth text
                for i in range(len(sample)):
                    if sample[i] == target_item:
                        min_id = i
                        match_type = "exact"
                        break

            # Update statistics
            if match_type == "exact":
                exact_matches += 1
            elif match_type == "fuzzy":
                fuzzy_matches += 1
            else:
                no_matches += 1

            # Update NDCG and HR metrics if target found within top-k
            for idx, topk in enumerate(valid_topk):
                if min_id < topk:
                    # NDCG uses log discount: 1 / log(rank + 2)
                    all_ndcg[idx] += 1 / math.log(min_id + 2)
                    # HR is binary hit (found in top-k or not)
                    all_hr[idx] += 1

        # Normalize metrics by number of samples and print results
        print(f"Number of beams: {n_beam}")
        print(f"Valid top-k values: {valid_topk}")
        print(f"Total items in catalog: {len(item_names)}")
        print(f"\nMatch Statistics:")
        if use_similarity:
            print(f"  Matched via similarity → ground truth ID: {fuzzy_matches}/{len(text)} ({100*fuzzy_matches/len(text):.2f}%)")
        else:
            print(f"  Exact text matches: {exact_matches}/{len(text)} ({100*exact_matches/len(text):.2f}%)")
        print(f"  No matches: {no_matches}/{len(text)} ({100*no_matches/len(text):.2f}%)")

        ndcg = [v / len(text) / (1.0 / math.log(2)) for v in all_ndcg]
        hr = [v / len(text) for v in all_hr]

        print(f"\nMetrics:")
        print(f"NDCG:\t{[f'{v:.4f}' for v in ndcg]}")
        print(f"HR:\t{[f'{v:.4f}' for v in hr]}")

        # Print formatted table
        print(f"\n{'='*60}")
        print(f"{'Metric':<10} {'@1':<10} {'@3':<10} {'@5':<10} {'@10':<10} {'@20':<10} {'@50':<10}")
        print(f"{'='*60}")
        print(f"{'NDCG':<10} {' '.join([f'{v*100:>8.2f}%' for v in ndcg])}")
        print(f"{'HR':<10} {' '.join([f'{v*100:>8.2f}%' for v in hr])}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    fire.Fire(evaluate_text_predictions)
