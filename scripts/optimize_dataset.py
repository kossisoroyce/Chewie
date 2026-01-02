#!/usr/bin/env python3
"""
Optimize the dataset for a precise CHW model.
1. Filters out non-CHW content (advanced hospital procedures, complex specialist diagnostics).
2. Standardizes the system prompt to a strong CHW persona.
3. Deduplicates and cleans.
"""

import json
import re
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train.json"
OUTPUT_FILE = DATA_DIR / "llama_finetune" / "train_optimized.json"

# High-quality CHW System Prompt
CHW_PROMPT = (
    "You are a Community Health Worker (CHW) assistant. "
    "Your goal is to provide accurate, accessible, and practical health advice relevant to resource-limited settings. "
    "Always prioritize patient safety, recognize danger signs that require urgent referral to a facility, "
    "and explain concepts simply. Do not provide advice outside your scope of practice."
)

# Keywords that suggest hospital-level care NOT relevant for CHW action
# We filter if these appear in the *Instruction* or *Output* in a way that implies the user is a specialist.
# However, patients asking about them is fine. We mostly want to avoid the model *acting* like a surgeon.
EXCLUDE_KEYWORDS = [
    "perform surgery", "start chemotherapy", "administer general anesthesia",
    "intubate", "order mri", "order ct scan", "biopsy result",
    "surgical resection", "laparoscopy"
]

# Topics to upsample (critical for CHW)
UPSAMPLE_TOPICS = {
    "malaria": 4,      # 4x duplication
    "tuberculosis": 2,
    "hiv": 2,
    "cholera": 4,
    "maternal": 2,
    "pregnant": 2,
    "newborn": 3,
    "infant": 2,
    "diarrhea": 2
}

def clean_text(text):
    # Remove excessive whitespace
    return re.sub(r'\s+', ' ', text).strip()

def get_upsample_factor(item):
    text = (item['input'] + " " + item['output']).lower()
    max_factor = 1
    for topic, factor in UPSAMPLE_TOPICS.items():
        if topic in text:
            max_factor = max(max_factor, factor)
    return max_factor

def is_chw_appropriate(item):
    text = (item['input'] + " " + item['output']).lower()
    
    # Filter out very short or empty content
    if len(item['output']) < 20:
        return False
        
    # Filter out "I am an AI" boilerplate if possible (though less common in this dataset)
    if "as an ai" in item['output'].lower():
        return False

    # Filter out specialist procedures where the model acts as the specialist
    # This is a heuristic; might lose some valid explanations but safer for a "precise" model
    for kw in EXCLUDE_KEYWORDS:
        if kw in item['output'].lower():
            return False
            
    return True

def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"Original size: {len(data)}")
    
    optimized_data = []
    seen_inputs = set()
    
    for item in tqdm(data):
        # 1. Clean text
        item['input'] = clean_text(item['input'])
        item['output'] = clean_text(item['output'])
        
        # 2. Deduplicate based on input
        if item['input'] in seen_inputs:
            continue
        seen_inputs.add(item['input'])
        
        # 3. Filter appropriateness
        if not is_chw_appropriate(item):
            continue
            
        # 4. Standardize System Prompt
        # We replace the varied prompts with our strong CHW prompt
        # This aligns the model to a single persona
        item['instruction'] = CHW_PROMPT
        
        # 5. Upsample critical topics
        factor = get_upsample_factor(item)
        for _ in range(factor):
            optimized_data.append(item.copy())
    
    print(f"Optimized size: {len(optimized_data)}")
    print(f"Removed: {len(data) - len(optimized_data)} items")
    
    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(optimized_data, f, indent=2)
        
    # Also save metadata
    meta = {
        "original_count": len(data),
        "optimized_count": len(optimized_data),
        "system_prompt": CHW_PROMPT
    }
    with open(DATA_DIR / "llama_finetune" / "optimization_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved optimized dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
