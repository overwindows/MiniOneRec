#!/usr/bin/env bash
set -euo pipefail

# Compare exact matching vs similarity-based matching for text-based SFT evaluation
# Usage: bash compare_matching_methods.sh [result_json] [item_path]

RESULT_JSON=${1:-"results_text/final_checkpoint/final_result_Industrial_and_Scientific.json"}
ITEM_PATH=${2:-"data/Amazon/info/Industrial_and_Scientific.txt"}

if [[ ! -f "$RESULT_JSON" ]]; then
    echo "Error: Result JSON not found: $RESULT_JSON"
    echo "Usage: bash compare_matching_methods.sh [result_json] [item_path]"
    exit 1
fi

if [[ ! -f "$ITEM_PATH" ]]; then
    echo "Error: Item file not found: $ITEM_PATH"
    echo "Usage: bash compare_matching_methods.sh [result_json] [item_path]"
    exit 1
fi

echo "============================================================"
echo "Comparing Matching Methods"
echo "Result JSON: $RESULT_JSON"
echo "Item Path: $ITEM_PATH"
echo "============================================================"
echo ""

# Run exact matching
echo "▶ Running EXACT MATCHING..."
echo "============================================================"
python src/calc_text_similarity.py \
    --path "$RESULT_JSON" \
    --item_path "$ITEM_PATH" \
    --use_similarity false
echo ""

# Run similarity matching with different thresholds
for threshold in 0.75 0.80 0.85 0.90 0.95; do
    echo ""
    echo "▶ Running SIMILARITY MATCHING (threshold=$threshold)..."
    echo "============================================================"
    python src/calc_text_similarity.py \
        --path "$RESULT_JSON" \
        --item_path "$ITEM_PATH" \
        --similarity_threshold $threshold \
        --use_similarity true
    echo ""
done

echo ""
echo "============================================================"
echo "Comparison Complete!"
echo "============================================================"
echo ""
echo "Analysis Tips:"
echo "  - Compare 'No matches' percentages across methods"
echo "  - Higher thresholds → stricter matching → closer to exact"
echo "  - Lower thresholds → more lenient → higher metrics"
echo "  - Recommended: Start with 0.85, adjust based on your needs"
echo ""
