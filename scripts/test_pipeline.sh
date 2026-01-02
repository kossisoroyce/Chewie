#!/bin/bash
set -e

# Activate venv
source venv/bin/activate

export OPENAI_API_KEY="OPENAI_API_KEY_PLACEHOLDER-wZ6rV3L2dX8wDAKJuTlMZZn9zTvAJ0kPiBRM4C92DjIFQ5fgbV1aFmGQO9GgBrU8nNVTO-d9E6T3BlbkFJ-Z9crP1ctpHAZgRn6bCPM20Ff-FGI92VPOexEzZaJw-97HUP_MUJvpRslwLQXumtAt2Mo1qHMA"

echo "🧪 TESTING PIPELINE (5 items)"

# 1. Refine
python scripts/refine_responses.py --limit 5 --workers 5

# 2. Translate
python scripts/translate_dataset.py \
    --input data/llama_finetune/train_refined.json \
    --output_dir data/translated_dataset \
    --limit 5 \
    --workers 5

echo "✅ Test Complete"
