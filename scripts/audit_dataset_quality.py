#!/usr/bin/env python3
"""
Deep dive analysis of the curated dataset quality and content.
Goal: Identify what to keep, what to drop, and how to rewrite for CHW precision.
"""

import json
from pathlib import Path
import random

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train.json"

def analyze_quality():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    # Group by source
    by_source = {}
    for item in data:
        # We lost the source column in the alpaca format json, 
        # but we can try to infer or just look at general content.
        # Wait, the previous script `prepare_llama_dataset.py` dropped the source field in the json output
        # but we can look at the `system_prompt` (instruction) to guess or just analyze randomly.
        # Actually, let's load the pandas parquet file for the curated set if possible?
        # No, the previous step just saved the json. 
        # However, the `prepare_llama_dataset.py` script saved metadata with source counts, 
        # so we know the composition.
        # Let's just sample random items to gauge "voice" and "content".
        pass

    # Random sampling of 20 items to inspect "CHW relevance"
    print("\n=== SAMPLING FOR QUALITY CHECK ===")
    
    samples = random.sample(data, 10)
    for i, item in enumerate(samples):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {item['instruction']}")
        print(f"Input: {item['input'][:200]}..." if len(item['input']) > 200 else f"Input: {item['input']}")
        print(f"Output: {item['output'][:300]}..." if len(item['output']) > 300 else f"Output: {item['output']}")
        
        # Simple heuristic check for CHW relevance
        # Non-CHW indicators: "surgery", "CT scan", "MRI", "prescription", "consult your doctor" (vague)
        is_hospital = any(x in item['output'].lower() for x in ['surgery', 'mri', 'ct scan', 'biopsy'])
        print(f"Potential Hospital-Level content? {'YES' if is_hospital else 'No'}")

if __name__ == "__main__":
    analyze_quality()
