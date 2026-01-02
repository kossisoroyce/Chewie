import json
from pathlib import Path

# Paths
DATA_DIR = Path("/home/kossiso/CascadeProjects/afrimed-chw/data/llama_finetune")
REFERENCE_REFINED = DATA_DIR / "reference_prompts_refined.json"
TRAIN_BILINGUAL = DATA_DIR / "train_bilingual.json"

def main():
    print(f"Merging {REFERENCE_REFINED.name} into {TRAIN_BILINGUAL.name}...")
    
    if not REFERENCE_REFINED.exists():
        print(f"Error: {REFERENCE_REFINED} not found.")
        return
    if not TRAIN_BILINGUAL.exists():
        print(f"Error: {TRAIN_BILINGUAL} not found.")
        return
        
    with open(REFERENCE_REFINED, 'r') as f:
        ref_data = json.load(f)
        
    with open(TRAIN_BILINGUAL, 'r') as f:
        bilingual_data = json.load(f)
        
    # Add language field to reference data if missing
    for item in ref_data:
        if "language" not in item:
            item["language"] = "en"
            
    combined_data = bilingual_data + ref_data
    
    print(f"Loaded {len(ref_data)} English reference prompts.")
    print(f"Loaded {len(bilingual_data)} existing bilingual examples.")
    print(f"Total after merge: {len(combined_data)} records.")
    
    with open(TRAIN_BILINGUAL, 'w') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ Successfully merged and updated {TRAIN_BILINGUAL}")

if __name__ == "__main__":
    main()
