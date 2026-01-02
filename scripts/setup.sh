#!/bin/bash
# AfriMed CHW Assistant - Setup Script

set -e

echo "🏥 AfriMed CHW Assistant - Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed,synthetic}
mkdir -p data/raw/who
mkdir -p logs
mkdir -p models
echo "✓ Directories created"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your credentials"
fi

# Check GCP authentication
echo "Checking GCP authentication..."
if gcloud auth application-default print-access-token &>/dev/null; then
    echo "✓ GCP authentication configured"
else
    echo "⚠️  GCP not authenticated. Run: gcloud auth application-default login"
fi

echo ""
echo "================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your GCP project ID and API keys"
echo "  2. Run: source venv/bin/activate"
echo "  3. Generate training data: python -m src.data_generation.synthetic_generator"
echo "  4. Scrape WHO guidelines: python -m src.data_collection.who_scraper"
echo "  5. Fine-tune model: python -m src.fine_tuning.train"
echo "  6. Start API: uvicorn src.api.main:app --reload"
