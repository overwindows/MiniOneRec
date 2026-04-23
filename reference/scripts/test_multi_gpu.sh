#!/usr/bin/env bash
# Quick test to verify multi-GPU evaluation works
set -euo pipefail

echo "============================================"
echo "Multi-GPU Evaluation Test"
echo "============================================"
echo ""

# Check GPUs
echo "Checking GPUs..."
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "⚠️  Transformers not found"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')" 2>/dev/null || echo "⚠️  Accelerate not found - install with: pip install accelerate"
python -c "from lm_eval import evaluator; print('✓ lm-eval-harness installed')" 2>/dev/null || echo "⚠️  lm-eval-harness not found"
echo ""

echo "============================================"
echo "Running quick test (1% of MMLU only)"
echo "This should take ~5-10 minutes"
echo "============================================"
echo ""

# Test with very small limit
./eval_llm.sh Qwen/Qwen3-4B-Instruct-2507 mmlu llm_eval 0.01

echo ""
echo "============================================"
echo "Test completed!"
echo "============================================"
echo ""
echo "If successful, run full evaluation with:"
echo "  ./eval_llm.sh Qwen/Qwen3-4B-Instruct-2507"
echo ""
echo "Or use data parallel for maximum speed:"
echo "  python llm_eval_parallel.py --model_path Qwen/Qwen3-4B-Instruct-2507 --verbose"
