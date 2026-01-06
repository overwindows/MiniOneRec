#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
export WANDB_API_KEY=fd3aec2cadf8ee9a2b3c6f4ac8210f65d73d134b
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

python - <<'PY'
try:
    import trl  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "Missing dependency: trl. Install with: pip install trl"
    ) from exc
PY

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    PROCESS_NUM=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    if command -v nvidia-smi >/dev/null 2>&1; then
        PROCESS_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        PROCESS_NUM=1
    fi
fi
# Allow AML or local overrides without editing this file.
OUTPUT_ROOT="${OUTPUT_ROOT:-output_dir}"
# Path to the SFT-trained model (should match the output_dir from sft.sh)
# MODEL_PATH="${MODEL_PATH:-output_dir/sft_Industrial_and_Scientific_qwen3-4b-instruct-2507_bs1024/final_checkpoint}"
MODEL_PATH="${MODEL_PATH:-output_dir/sft_Industrial_and_Scientific_qwen3-1.7B_bs1024/final_checkpoint}"

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    MAIN_PORT=${MAIN_PORT:-$((20000 + RANDOM % 45000))}
    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes ${PROCESS_NUM} --main_process_port ${MAIN_PORT} \
                                    rl.py \
                        --model_path ${MODEL_PATH} \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type ranking \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir ${OUTPUT_ROOT}/rl_${category}_qwen3-1.7B_bs1024 \
                        --wandb_project MiniOneRec \
                        --wandb_run_name rl_${category}_qwen3-1.7B_bs1024 \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done
