#!/usr/bin/env python3
"""
Fine-tune Gemini Flash using Google AI Studio or Vertex AI.

Prerequisites:
1. Google Cloud Project with Vertex AI enabled OR Google AI Studio API key
2. Set environment variables:
   - GOOGLE_API_KEY (for AI Studio)
   - OR GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS (for Vertex AI)

Usage:
    python finetune_gemini.py --method aistudio  # Using Google AI Studio
    python finetune_gemini.py --method vertex    # Using Vertex AI
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
GEMINI_DATA_DIR = DATA_DIR / "gemini_finetune"
OUTPUT_DIR = Path(__file__).parent.parent / "models"

# Fine-tuning configuration
CONFIG = {
    "base_model": "gemini-1.5-flash-002",  # or gemini-2.0-flash when available for tuning
    "display_name": f"africhw-medical-{datetime.now().strftime('%Y%m%d')}",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate_multiplier": 1.0,
}


def finetune_with_aistudio():
    """Fine-tune using Google AI Studio API."""
    import google.generativeai as genai
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("   Get your API key from: https://aistudio.google.com/app/apikey")
        return None
    
    genai.configure(api_key=api_key)
    
    print("📤 Uploading training data...")
    train_file = GEMINI_DATA_DIR / "train.jsonl"
    
    # Upload training file
    training_data = genai.upload_file(
        path=str(train_file),
        display_name="africhw-medical-train"
    )
    print(f"   Uploaded: {training_data.name}")
    
    # Check for validation file
    val_file = GEMINI_DATA_DIR / "validation.jsonl"
    validation_data = None
    if val_file.exists():
        validation_data = genai.upload_file(
            path=str(val_file),
            display_name="africhw-medical-val"
        )
        print(f"   Uploaded validation: {validation_data.name}")
    
    print("\n🚀 Starting fine-tuning job...")
    print(f"   Base model: {CONFIG['base_model']}")
    print(f"   Display name: {CONFIG['display_name']}")
    print(f"   Epochs: {CONFIG['epochs']}")
    
    # Create tuning job
    operation = genai.create_tuned_model(
        source_model=CONFIG["base_model"],
        training_data=training_data,
        display_name=CONFIG["display_name"],
        epoch_count=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate_multiplier=CONFIG["learning_rate_multiplier"],
    )
    
    print("\n⏳ Fine-tuning in progress...")
    print("   This may take several hours. You can check status with:")
    print(f"   python -c \"import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); print(genai.get_tuned_model('{CONFIG['display_name']}'))\"")
    
    # Wait for completion (this blocks)
    try:
        result = operation.result()
        print(f"\n✅ Fine-tuning complete!")
        print(f"   Model name: {result.name}")
        
        # Save model info
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_info = {
            "model_name": result.name,
            "display_name": CONFIG["display_name"],
            "base_model": CONFIG["base_model"],
            "created": datetime.now().isoformat(),
            "config": CONFIG,
        }
        with open(OUTPUT_DIR / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        return None


def finetune_with_vertex():
    """Fine-tune using Vertex AI."""
    from google.cloud import aiplatform
    from vertexai.tuning import sft
    
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project:
        print("❌ Error: GOOGLE_CLOUD_PROJECT environment variable not set")
        return None
    
    print(f"📍 Using project: {project}, location: {location}")
    
    aiplatform.init(project=project, location=location)
    
    # Upload training data to GCS first
    print("\n📤 Note: For Vertex AI, upload your training data to GCS first:")
    print(f"   gsutil cp {GEMINI_DATA_DIR}/train.jsonl gs://YOUR_BUCKET/africhw/train.jsonl")
    print(f"   gsutil cp {GEMINI_DATA_DIR}/validation.jsonl gs://YOUR_BUCKET/africhw/validation.jsonl")
    
    gcs_train_path = input("\nEnter GCS path to training data (gs://...): ").strip()
    if not gcs_train_path:
        print("❌ GCS path required for Vertex AI")
        return None
    
    gcs_val_path = input("Enter GCS path to validation data (optional, press Enter to skip): ").strip()
    
    print("\n🚀 Starting Vertex AI fine-tuning job...")
    
    # Create supervised fine-tuning job
    sft_tuning_job = sft.train(
        source_model=CONFIG["base_model"],
        train_dataset=gcs_train_path,
        validation_dataset=gcs_val_path if gcs_val_path else None,
        epochs=CONFIG["epochs"],
        adapter_size=4,  # LoRA adapter size
        learning_rate_multiplier=CONFIG["learning_rate_multiplier"],
        tuned_model_display_name=CONFIG["display_name"],
    )
    
    print(f"\n⏳ Fine-tuning job started: {sft_tuning_job.resource_name}")
    print("   Monitor in Google Cloud Console or wait here for completion...")
    
    # This blocks until complete
    sft_tuning_job.wait()
    
    print(f"\n✅ Fine-tuning complete!")
    print(f"   Tuned model endpoint: {sft_tuning_job.tuned_model_endpoint_name}")
    
    # Save model info
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_info = {
        "endpoint": sft_tuning_job.tuned_model_endpoint_name,
        "display_name": CONFIG["display_name"],
        "base_model": CONFIG["base_model"],
        "created": datetime.now().isoformat(),
        "config": CONFIG,
    }
    with open(OUTPUT_DIR / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    return sft_tuning_job


def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check data files
    train_file = GEMINI_DATA_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}")
        print("   Run: python scripts/prepare_gemini_dataset.py")
        return False
    
    # Count examples
    with open(train_file) as f:
        num_examples = sum(1 for _ in f)
    print(f"✅ Training data found: {num_examples:,} examples")
    
    # Check for API key or GCP credentials
    has_aistudio = bool(os.environ.get("GOOGLE_API_KEY"))
    has_vertex = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    
    if has_aistudio:
        print("✅ Google AI Studio API key found")
    if has_vertex:
        print("✅ Google Cloud Project configured")
    
    if not has_aistudio and not has_vertex:
        print("❌ No API credentials found")
        print("   Set GOOGLE_API_KEY for AI Studio")
        print("   Or set GOOGLE_CLOUD_PROJECT for Vertex AI")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemini for CHW assistant")
    parser.add_argument(
        "--method", 
        choices=["aistudio", "vertex", "auto"],
        default="auto",
        help="Fine-tuning method (default: auto-detect)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites, don't start fine-tuning"
    )
    args = parser.parse_args()
    
    print("="*60)
    print("GEMINI FINE-TUNING FOR AFRICHW-MEDICAL")
    print("="*60)
    
    if not check_prerequisites():
        return
    
    if args.check_only:
        print("\n✅ Prerequisites check passed!")
        return
    
    # Determine method
    method = args.method
    if method == "auto":
        if os.environ.get("GOOGLE_API_KEY"):
            method = "aistudio"
        elif os.environ.get("GOOGLE_CLOUD_PROJECT"):
            method = "vertex"
        else:
            print("❌ Could not auto-detect method")
            return
    
    print(f"\n🎯 Using method: {method}")
    
    if method == "aistudio":
        finetune_with_aistudio()
    else:
        finetune_with_vertex()


if __name__ == "__main__":
    main()
