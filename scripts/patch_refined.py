#!/usr/bin/env python3
"""
Quick patch for train_refined.json to replace PII placeholders with neutral terms.
"""

import json
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train_refined.json"
OUTPUT_FILE = DATA_DIR / "llama_finetune" / "train_refined_patched.json"

# Replacement map
REPLACEMENTS = {
    "[DOCTOR]": "the specialist",
    "[NAME]": "the patient",
}

def main():
    print("🔧 Patching train_refined.json...")
    
    # Load
    with open(INPUT_FILE) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records")
    
    # Patch
    patched = []
    for item in data:
        new_item = item.copy()
        for old, new in REPLACEMENTS.items():
            new_item["output"] = new_item["output"].replace(old, new)
        patched.append(new_item)
    
    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(patched, f, indent=2)
    
    print(f"✅ Patched dataset saved to {OUTPUT_FILE}")
    
    # Verify
    remaining = sum(1 for item in patched if any(tok in item["output"] for tok in REPLACEMENTS))
    print(f"Remaining placeholders: {remaining}")

if __name__ == "__main__":
    main()
