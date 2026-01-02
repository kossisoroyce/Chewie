#!/usr/bin/env python3
"""
Finalize the pilot dataset by combining:
1. Refined English examples (High Quality)
2. Translated Swahili examples
"""

import json
import random
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    refined_path = data_dir / "llama_finetune" / "train_refined_final.json"
    swahili_path = data_dir / "translated_dataset" / "train_swahili.json"
    output_dir = data_dir / "final_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_records = []
    
    # 1. Load Refined English
    if refined_path.exists():
        print(f"Loading refined English data from {refined_path}...")
        with open(refined_path) as f:
            refined_data = json.load(f)
            
        print(f"  Found {len(refined_data)} records")
        for item in refined_data:
            # Ensure standard fields
            record = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "source": item.get("source", "unknown"),
                "language": "en",
                "quality": "refined"
            }
            final_records.append(record)
    else:
        print(f"⚠️ Warning: Refined file not found at {refined_path}")

    # 2. Load Translated Swahili
    if swahili_path.exists():
        print(f"Loading translated Swahili data from {swahili_path}...")
        with open(swahili_path) as f:
            swahili_data = json.load(f)
            
        print(f"  Found {len(swahili_data)} records")
        for item in swahili_data:
            record = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
                "source": item.get("source", "unknown"),
                "language": "sw",
                "quality": "translated"
            }
            final_records.append(record)
    else:
        print(f"⚠️ Warning: Swahili file not found at {swahili_path}")

    # 3. Shuffle and Save
    print(f"Total records: {len(final_records)}")
    random.shuffle(final_records)
    
    output_path = output_dir / "train_pilot.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Saved final pilot dataset to {output_path}")
    
    # Create a small validation split (optional, e.g., 5%)
    val_size = int(len(final_records) * 0.05)
    train_data = final_records[val_size:]
    val_data = final_records[:val_size]
    
    print(f"Split: {len(train_data)} Train, {len(val_data)} Validation")
    
    with open(output_dir / "train.json", "w", encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(output_dir / "val.json", "w", encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
