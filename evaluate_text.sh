#!/usr/bin/env bash
set -euo pipefail

# Industrial_and_Scientific
# Office_Products
for category in "Industrial_and_Scientific"
do
    # your model path
    exp_name="output_dir/sft_text_Industrial_and_Scientific/final_checkpoint"

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (TEXT MODE)"

    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    item_meta_file=$(ls ./data/Amazon/index/${category}.item.json 2>/dev/null | head -1)

    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi

    output_dir="./results_text/${exp_name_clean}"
    mkdir -p "$output_dir"

    result_json="$output_dir/final_result_${category}.json"

    if [[ -f "$item_meta_file" ]]; then
        python ./evaluate_text.py \
            --base_model "$exp_name" \
            --category ${category} \
            --test_data_path "$test_file" \
            --item_meta_path "$item_meta_file" \
            --result_json_data "$result_json" \
            --batch_size 4 \
            --num_beams 20 \
            --max_new_tokens 256 \
            --length_penalty 0.0
    else
        python ./evaluate_text.py \
            --base_model "$exp_name" \
            --category ${category} \
            --test_data_path "$test_file" \
            --result_json_data "$result_json" \
            --batch_size 4 \
            --num_beams 20 \
            --max_new_tokens 256 \
            --length_penalty 0.0
    fi

    python ./calc_text.py \
        --path "$result_json" \
        --item_path "$info_file"

done

echo "All categories processed!"
