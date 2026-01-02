#!/bin/bash
# Full Data Pipeline: Refine (English) -> Translate (Swahili)
# Usage: ./scripts/run_pipeline.sh [api_key] [limit]

set -e

API_KEY=$1
LIMIT=${2:-0}
STEP=${3:-"all"} # 'refine', 'translate', or 'all'

# Load from .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -n "$API_KEY" ]; then
    export OPENAI_API_KEY=$API_KEY
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not provided."
    echo "Usage: ./scripts/run_pipeline.sh [api_key] [limit] [step]"
    exit 1
fi

# Activate venv
source venv/bin/activate

echo "========================================================"
echo "🚀 STARTING AFRICHW DATA PIPELINE"
if [ "$LIMIT" -gt 0 ]; then
    echo "⚠️  LIMIT APPLIED: Processing first $LIMIT examples only"
fi
echo "========================================================"

# 1. Refine Responses (English -> Structured CHW Format)
if [ "$STEP" = "all" ] || [ "$STEP" = "refine" ]; then
    echo ""
    echo "🔹 STEP 1: Refining English Responses (Assessment/Action/Advice)..."
    echo "Input: data/llama_finetune/train.json"
    echo "Output: data/llama_finetune/train_refined.json"

    # We use the optimized train.json as input
    python scripts/refine_responses.py --workers 20 --model gpt-4o-mini --limit $LIMIT
    
    echo "✅ Refinement Complete."
fi

# 2. Translate to Swahili
if [ "$STEP" = "all" ] || [ "$STEP" = "translate" ]; then
    echo ""
    echo "🔹 STEP 2: Translating to Swahili..."
    echo "Input: data/llama_finetune/train_refined.json"
    echo "Output: data/translated_dataset/train_swahili.json"
    
    # Check if refined data exists, else use raw train.json
    INPUT_DATA="data/llama_finetune/train_refined.json"
    if [ ! -f "$INPUT_DATA" ]; then
        echo "⚠️  Refined data not found, using optimized train.json"
        INPUT_DATA="data/llama_finetune/train.json"
    fi

    python scripts/translate_dataset.py \
        --input "$INPUT_DATA" \
        --output_dir data/translated_dataset \
        --workers 20 \
        --model gpt-4o-mini \
        --limit $LIMIT
    
    echo "✅ Translation Complete."
fi

echo ""
echo "========================================================"
echo "🎉 PIPELINE FINISHED"
echo "files:"
echo "  - English (Refined): data/llama_finetune/train_refined.json"
echo "  - Swahili: data/translated_dataset/train_swahili.json"
echo "========================================================"
