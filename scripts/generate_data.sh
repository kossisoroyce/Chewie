#!/bin/bash
# Generate training data for AfriMed

set -e

echo "🏥 AfriMed - Data Generation Pipeline"
echo "======================================"

# Activate virtual environment
source venv/bin/activate

# Step 1: Scrape WHO guidelines
echo ""
echo "Step 1: Scraping WHO maternal health guidelines..."
python -m src.data_collection.who_scraper
echo "✓ WHO guidelines scraped"

# Step 2: Generate synthetic training data
echo ""
echo "Step 2: Generating synthetic training data..."
python -m src.data_generation.synthetic_generator \
    --output data/synthetic \
    --num-examples 2000 \
    --include-swahili \
    --swahili-ratio 0.3
echo "✓ Synthetic data generated"

# Step 3: Combine and prepare for fine-tuning
echo ""
echo "Step 3: Preparing data for fine-tuning..."

# Create processed directory
mkdir -p data/processed

# Combine all data sources
cat data/synthetic/training_data.jsonl > data/processed/training_data.jsonl
echo "✓ Training data prepared"

# Count examples
train_count=$(wc -l < data/processed/training_data.jsonl)
echo ""
echo "======================================"
echo "✅ Data generation complete!"
echo "   Training examples: $train_count"
echo ""
echo "Next: Run fine-tuning with:"
echo "   python -m src.fine_tuning.train --config configs/finetune_config.yaml"
