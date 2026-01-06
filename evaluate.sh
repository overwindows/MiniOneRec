# Industrial_and_Scientific
# Office_Products
for category in "Industrial_and_Scientific"
do
    # your model path
    # exp_name="output_dir/sft_Industrial_and_Scientific_qwen3-1.7B_bs1024/final_checkpoint"
    exp_name="output_dir/rl_Industrial_and_Scientific_qwen3-1.7B_bs1024"
    # exp_name="output_dir/rl_Industrial_and_Scientific_qwen1.5b_bs1024/final_checkpoint"

    # exp_name="output_dir/sft_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/final_checkpoint"
    # exp_name="output_dir/rl_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/checkpoint-660"
    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    # Add PID to temp directory to avoid race conditions when running multiple evaluations
    temp_dir="./temp/${category}-${exp_name_clean}-$$"
    if [[ -d "$temp_dir" ]]; then
        echo "WARNING: Temp directory already exists, removing: $temp_dir"
        rm -rf "$temp_dir"
    fi
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    echo "Splitting test data..."
    if ! python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "4,5,6,7"; then
        echo "ERROR: split.py failed for category $category"
        continue
    fi

    if [[ ! -f "$temp_dir/4.csv" ]]; then
        echo "ERROR: Data splitting failed - expected file $temp_dir/4.csv not found"
        echo "Check split.py output above for details"
        ls -la "$temp_dir" 2>/dev/null || echo "Temp directory doesn't exist"
        continue
    fi

    cudalist="4 5 6 7"
    echo "Starting parallel evaluation (STANDARD MODE)..."
    echo "GPUs: $cudalist"

    # Track PIDs for better process management
    pids=()
    gpu_pids=()

    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "[GPU $i] Starting evaluation for category ${category}"
            echo "[GPU $i] Input: $temp_dir/${i}.csv"
            echo "[GPU $i] Output: $temp_dir/${i}.json"

            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$exp_name"/final_checkpoint \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 4 \
                --num_beams 50 \
                --max_new_tokens 256 \
                --temperature 1.0 \
                --guidance_scale 1.0 \
                --length_penalty 0.0 2>&1 | sed "s/^/[GPU $i] /" &

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
    
    output_dir="./results/${exp_name_clean}"
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
    
    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------" 
done

echo "All categories processed!"
