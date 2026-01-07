#!/usr/bin/env bash
set -euo pipefail

# Industrial_and_Scientific
# Office_Products
for category in "Industrial_and_Scientific"
do
    # your model path
    exp_name="output_dir/sft_text_Industrial_and_Scientific_qwen3-1.7B_bs1024"

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

    # Add PID to temp directory to avoid race conditions when running multiple evaluations
    temp_dir="./temp_text/${category}-${exp_name_clean}-$$"
    if [[ -d "$temp_dir" ]]; then
        echo "WARNING: Temp directory already exists, removing: $temp_dir"
        rm -rf "$temp_dir"
    fi
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"

    # Configure GPU list (can be overridden with CUDA_LIST environment variable)
    cuda_list=${CUDA_LIST:-"0,1,2,3"}
    echo "Splitting test data..."
    if ! python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "$cuda_list"; then
        echo "ERROR: split.py failed for category $category"
        continue
    fi

    # Verify split succeeded by checking for first split file
    first_gpu=$(echo "$cuda_list" | cut -d',' -f1)
    if [[ ! -f "$temp_dir/${first_gpu}.csv" ]]; then
        echo "ERROR: Data splitting failed - expected file $temp_dir/${first_gpu}.csv not found"
        echo "Check split.py output above for details"
        ls -la "$temp_dir" 2>/dev/null || echo "Temp directory doesn't exist"
        continue
    fi

    # Determine actual model path (check if final_checkpoint exists)
    if [[ -d "$exp_name/final_checkpoint" ]]; then
        model_path="$exp_name/final_checkpoint"
        echo "Using model: $model_path (found final_checkpoint)"
    elif [[ -d "$exp_name" ]]; then
        model_path="$exp_name"
        echo "Using model: $model_path (direct path)"
    else
        echo "ERROR: Model not found at $exp_name or $exp_name/final_checkpoint"
        continue
    fi

    cudalist=$(echo "$cuda_list" | tr ',' ' ')
    echo "Starting parallel evaluation (TEXT MODE)..."
    echo "GPUs: $cudalist"

    # Track PIDs for better process management
    pids=()
    gpu_pids=()

    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "[GPU $i] Starting text evaluation for category ${category}"
            echo "[GPU $i] Input: $temp_dir/${i}.csv"
            echo "[GPU $i] Output: $temp_dir/${i}.json"

            # Build the command with optional item_meta_path
            cmd="CUDA_VISIBLE_DEVICES=$i python -u src/evaluate_text.py \
                --base_model \"$model_path\" \
                --category ${category} \
                --test_data_path \"$temp_dir/${i}.csv\" \
                --result_json_data \"$temp_dir/${i}.json\" \
                --batch_size 4 \
                --num_beams 20 \
                --max_new_tokens 256 \
                --length_penalty 0.0"

            if [[ -f "$item_meta_file" ]]; then
                cmd="$cmd --item_meta_path \"$item_meta_file\""
            fi

            # Execute command in background with GPU prefix for logging
            eval "$cmd 2>&1 | sed \"s/^/[GPU $i] /\"" &

            pid=$!
            pids+=($pid)
            gpu_pids+=("$i:$pid")
            echo "[GPU $i] Process started with PID $pid"
        else
            echo "WARNING: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done

    if [[ ${#pids[@]} -eq 0 ]]; then
        echo "ERROR: No evaluation processes started!"
        continue
    fi

    echo ""
    echo "Waiting for ${#pids[@]} evaluation process(es) to complete..."
    echo "Active processes: ${gpu_pids[@]}"
    echo ""

    # Wait for all processes and track failures
    failed_gpus=()
    for idx in "${!pids[@]}"; do
        pid=${pids[$idx]}
        gpu_info=${gpu_pids[$idx]}
        gpu=${gpu_info%%:*}

        if wait $pid; then
            echo "✓ GPU $gpu (PID $pid) completed successfully"
        else
            exit_code=$?
            echo "✗ ERROR: GPU $gpu (PID $pid) failed with exit code $exit_code"
            failed_gpus+=($gpu)
        fi
    done

    echo ""
    if [[ ${#failed_gpus[@]} -gt 0 ]]; then
        echo "WARNING: ${#failed_gpus[@]} GPU(s) failed: ${failed_gpus[@]}"
        echo "Merge will continue with available results..."
    else
        echo "✓ All GPU processes completed successfully"
    fi
    echo ""

    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi

    output_dir="./results_text/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"

    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"

    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi

    # Similarity-based evaluation (default: enabled with 0.85 threshold)
    # Set USE_SIMILARITY=false for exact matching
    # Adjust SIMILARITY_THRESHOLD (0.0-1.0) as needed
    USE_SIMILARITY=${USE_SIMILARITY:-true}
    SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD:-0.85}

    echo "Calculating metrics with similarity matching..."
    python src/calc_text_similarity.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file" \
        --similarity_threshold ${SIMILARITY_THRESHOLD} \
        --use_similarity ${USE_SIMILARITY}

    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------"
done

echo "All categories processed!"
