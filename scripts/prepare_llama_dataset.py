#!/usr/bin/env python3
"""
Prepare the AfriCHW-Medical dataset for Llama 3.2 fine-tuning.

Llama 3.2 uses the chat template format:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{assistant_response}<|eot_id|>
"""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "africhw_medical_v1_clean"
OUTPUT_DIR = DATA_DIR / "llama_finetune"

# Fine-tuning limits for A100 GPU
MAX_EXAMPLES = 100000  # Full dataset for A100
MAX_LENGTH = 4096      # Longer context for A100


def format_llama_chat(row: dict) -> str:
    """Format as Llama 3.2 chat template."""
    conversation = row["conversation"]
    system_prompt = row["system_prompt"]
    
    # Build the formatted text
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    
    for turn in conversation:
        role = turn["role"]
        content = turn["content"]
        
        if role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
        else:
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    
    text += "<|end_of_text|>"
    return text


def format_alpaca_style(row: dict) -> dict:
    """Format as Alpaca-style for easier fine-tuning."""
    conversation = row["conversation"]
    system_prompt = row["system_prompt"]
    
    # Extract user input and assistant output
    user_text = ""
    assistant_text = ""
    
    for turn in conversation:
        if turn["role"] == "user":
            user_text = turn["content"]
        else:
            assistant_text = turn["content"]
    
    return {
        "instruction": system_prompt,
        "input": user_text,
        "output": assistant_text,
    }


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for quality examples."""
    print("  Filtering for quality...")
    initial = len(df)
    
    def check_output_length(x):
        try:
            if x is None or len(x) == 0:
                return False
            last_turn = x[-1] if isinstance(x, list) else list(x)[-1]
            content = last_turn.get("content", "")
            # Output between 50 and 2000 chars (good for training)
            return 50 < len(content) < 2000
        except:
            return False
    
    def check_total_length(x):
        try:
            return len(str(x)) < 8000  # Reasonable total length
        except:
            return False
    
    df = df[df["conversation"].apply(check_output_length)]
    df = df[df["conversation"].apply(check_total_length)]
    
    print(f"  Filtered: {initial:,} -> {len(df):,}")
    return df


def prioritize_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Prioritize high-value examples."""
    print("  Prioritizing examples...")
    
    priority_map = {
        "afrimedqa": 1,
        "chw_synthetic": 1,
        "symptom_diagnosis": 2,
        "medqa_usmle": 2,
        "diseases_symptoms": 2,
        "chatdoctor": 3,
        "medical_chatbot": 4,
        "medinstruct": 4,
    }
    
    df["priority"] = df["source"].map(priority_map).fillna(10)
    df = df.sort_values("priority")
    
    return df


def main():
    print("="*60)
    print("PREPARING DATASET FOR LLAMA 3.2 FINE-TUNING")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load clean dataset
    print("\n📂 Loading clean dataset...")
    train_df = pd.read_parquet(INPUT_DIR / "train.parquet")
    val_df = pd.read_parquet(INPUT_DIR / "validation.parquet")
    print(f"  Train: {len(train_df):,}, Validation: {len(val_df):,}")
    
    # Filter and prioritize
    train_df = filter_quality(train_df)
    train_df = prioritize_examples(train_df)
    val_df = filter_quality(val_df)
    
    # Take subset for fine-tuning
    train_subset = train_df.head(MAX_EXAMPLES)
    val_subset = val_df.head(2000)
    
    print(f"\n📊 Using {len(train_subset):,} training examples")
    print(f"   Using {len(val_subset):,} validation examples")
    
    # Show source distribution
    print("\n   Source distribution:")
    for source, count in train_subset["source"].value_counts().head(8).items():
        print(f"     {source}: {count:,}")
    
    # Convert to Alpaca format (works best with most fine-tuning libraries)
    print("\n🔄 Converting to Alpaca format...")
    
    train_records = []
    for _, row in tqdm(train_subset.iterrows(), total=len(train_subset), desc="  Train"):
        record = format_alpaca_style(row.to_dict())
        if record["input"] and record["output"]:
            train_records.append(record)
    
    val_records = []
    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc="  Val"):
        record = format_alpaca_style(row.to_dict())
        if record["input"] and record["output"]:
            val_records.append(record)
    
    # Save JSONL files
    print("\n💾 Saving files...")
    
    train_file = OUTPUT_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")
    
    val_file = OUTPUT_DIR / "validation.jsonl"
    with open(val_file, "w") as f:
        for record in val_records:
            f.write(json.dumps(record) + "\n")
    
    # Also save as single JSON for HuggingFace datasets
    train_json = OUTPUT_DIR / "train.json"
    with open(train_json, "w") as f:
        json.dump(train_records, f)
    
    val_json = OUTPUT_DIR / "validation.json"
    with open(val_json, "w") as f:
        json.dump(val_records, f)
    
    # Save metadata
    metadata = {
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "format": "alpaca",
        "target_model": "llama-3.2-3b",
        "max_examples": MAX_EXAMPLES,
        "source_distribution": train_subset["source"].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("LLAMA FINE-TUNING DATA READY")
    print("="*60)
    print(f"\n📁 Output: {OUTPUT_DIR}")
    print(f"\n📄 Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1024*1024:
            print(f"   {f.name}: {size/1024/1024:.1f} MB")
        else:
            print(f"   {f.name}: {size/1024:.1f} KB")
    
    print(f"\n📊 Dataset stats:")
    print(f"   Training examples: {len(train_records):,}")
    print(f"   Validation examples: {len(val_records):,}")
    
    print("\n✅ Ready for Llama 3.2 fine-tuning on Colab!")
    print("   Upload train.json to Colab and run the notebook")


if __name__ == "__main__":
    main()
