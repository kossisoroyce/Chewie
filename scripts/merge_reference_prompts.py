import json
import os
from pathlib import Path

# Paths
DATA_DIR = Path("/home/kossiso/CascadeProjects/afrimed-chw/data/llama_finetune")
REFERENCE_REFINED = DATA_DIR / "reference_prompts_refined.json"
TRAIN_PATCHED = DATA_DIR / "train_refined_patched.json"
OUTPUT_FILE = DATA_DIR / "train_refined_final.json"

def main():
    print(f"Merging {REFERENCE_REFINED.name} and {TRAIN_PATCHED.name}...")
    
    if not REFERENCE_REFINED.exists():
        print(f"Error: {REFERENCE_REFINED} not found.")
        return
    if not TRAIN_PATCHED.exists():
        print(f"Error: {TRAIN_PATCHED} not found.")
        return
        
    with open(REFERENCE_REFINED, 'r') as f:
        ref_data = json.load(f)
        
    with open(TRAIN_PATCHED, 'r') as f:
        train_data = json.load(f)
        
    combined_data = train_data + ref_data
    
    print(f"Loaded {len(ref_data)} reference prompts.")
    print(f"Loaded {len(train_data)} existing patched training examples.")
    print(f"Total: {len(combined_data)} records.")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Successfully merged into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
