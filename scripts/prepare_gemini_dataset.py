#!/usr/bin/env python3
"""
Prepare the AfriCHW-Medical dataset for Gemini fine-tuning.

Gemini fine-tuning expects JSONL with this format:
{
  "contents": [
    {"role": "user", "parts": [{"text": "..."}]},
    {"role": "model", "parts": [{"text": "..."}]}
  ]
}

Or for supervised fine-tuning:
{
  "text_input": "...",
  "output": "..."
}
"""

import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "africhw_medical_v1_clean"
OUTPUT_DIR = DATA_DIR / "gemini_finetune"

# Fine-tuning limits
MAX_EXAMPLES = 50000  # Start with a subset for faster iteration
MAX_INPUT_CHARS = 8000  # Gemini context limit consideration
MAX_OUTPUT_CHARS = 4000


def convert_to_gemini_chat_format(row: dict) -> dict:
    """Convert to Gemini chat format with system instruction."""
    conversation = row["conversation"]
    system_prompt = row["system_prompt"]
    
    contents = []
    
    for turn in conversation:
        role = "user" if turn["role"] == "user" else "model"
        text = turn["content"]
        
        # Add system prompt to first user message
        if role == "user" and not contents:
            text = f"{system_prompt}\n\n{text}"
        
        contents.append({
            "role": role,
            "parts": [{"text": text}]
        })
    
    return {"contents": contents}


def convert_to_gemini_sft_format(row: dict) -> dict:
    """Convert to simple supervised fine-tuning format."""
    conversation = row["conversation"]
    system_prompt = row["system_prompt"]
    
    # Combine system prompt with user input
    user_text = ""
    assistant_text = ""
    
    for turn in conversation:
        if turn["role"] == "user":
            user_text = turn["content"]
        else:
            assistant_text = turn["content"]
    
    # Prepend system prompt
    text_input = f"{system_prompt}\n\n{user_text}"
    
    return {
        "text_input": text_input[:MAX_INPUT_CHARS],
        "output": assistant_text[:MAX_OUTPUT_CHARS]
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
            return len(last_turn.get("content", "")) > 50
        except:
            return False
    
    def check_total_length(x):
        try:
            return len(str(x)) < 15000
        except:
            return False
    
    # Remove very short outputs
    df = df[df["conversation"].apply(check_output_length)]
    
    # Remove very long examples (may cause issues)
    df = df[df["conversation"].apply(check_total_length)]
    
    print(f"  Filtered: {initial} -> {len(df)} ({initial - len(df)} removed)")
    return df


def prioritize_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Prioritize high-value examples for fine-tuning."""
    print("  Prioritizing examples...")
    
    # Priority order for sources
    priority_map = {
        "afrimedqa": 1,        # African-specific - highest priority
        "chw_synthetic": 1,    # CHW examples - highest priority
        "medqa_usmle": 2,      # Medical reasoning
        "symptom_diagnosis": 2, # Diagnosis skills
        "diseases_symptoms": 2,
        "chatdoctor": 3,       # Consultations
        "medical_chatbot": 4,  # General
        "medinstruct": 4,
        "wikidoc": 5,
        "wikidoc_patient": 5,
        "medmcqa": 6,          # MCQs - lower priority
    }
    
    df["priority"] = df["source"].map(priority_map).fillna(10)
    df = df.sort_values("priority")
    
    return df


def main():
    print("="*60)
    print("PREPARING DATASET FOR GEMINI FINE-TUNING")
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
    val_subset = val_df.head(5000)
    
    print(f"\n📊 Using {len(train_subset):,} training examples")
    print(f"   Using {len(val_subset):,} validation examples")
    
    # Show source distribution in subset
    print("\n   Source distribution in training subset:")
    for source, count in train_subset["source"].value_counts().head(10).items():
        print(f"     {source}: {count:,}")
    
    # Convert to Gemini SFT format
    print("\n🔄 Converting to Gemini format...")
    
    train_records = []
    for _, row in tqdm(train_subset.iterrows(), total=len(train_subset), desc="  Train"):
        record = convert_to_gemini_sft_format(row.to_dict())
        if record["text_input"] and record["output"]:
            train_records.append(record)
    
    val_records = []
    for _, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc="  Val"):
        record = convert_to_gemini_sft_format(row.to_dict())
        if record["text_input"] and record["output"]:
            val_records.append(record)
    
    # Save JSONL files
    print("\n💾 Saving Gemini format files...")
    
    train_file = OUTPUT_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for record in train_records:
            f.write(json.dumps(record) + "\n")
    
    val_file = OUTPUT_DIR / "validation.jsonl"
    with open(val_file, "w") as f:
        for record in val_records:
            f.write(json.dumps(record) + "\n")
    
    # Also save chat format version
    print("  Saving chat format version...")
    train_chat_records = []
    for _, row in tqdm(train_subset.head(10000).iterrows(), total=min(10000, len(train_subset)), desc="  Chat"):
        record = convert_to_gemini_chat_format(row.to_dict())
        train_chat_records.append(record)
    
    chat_file = OUTPUT_DIR / "train_chat.jsonl"
    with open(chat_file, "w") as f:
        for record in train_chat_records:
            f.write(json.dumps(record) + "\n")
    
    # Save metadata
    metadata = {
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "format": "gemini_sft",
        "max_input_chars": MAX_INPUT_CHARS,
        "max_output_chars": MAX_OUTPUT_CHARS,
        "source_distribution": train_subset["source"].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("GEMINI FINE-TUNING DATA READY")
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
    
    print("\n✅ Ready for Gemini fine-tuning!")
    print("   Next: Run finetune_gemini.py with your API credentials")


if __name__ == "__main__":
    main()
