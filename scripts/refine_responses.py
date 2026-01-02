#!/usr/bin/env python3
"""
Synthetic Refinement Pipeline for AfriCHW-Medical.
Uses a teacher model (GPT-4o/Gemini) to rewrite dataset outputs into a strict CHW format.

Format:
1. **Assessment:** Brief summary of symptoms/situation.
2. **Action:** Clear steps (Referral vs Home Care).
3. **Explanation:** Simple, non-jargon explanation for the patient.

Usage:
    python scripts/refine_responses.py --limit 1000 --model gpt-4o-mini
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    OpenAI = None

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train.json"
OUTPUT_FILE = DATA_DIR / "llama_finetune" / "train_refined.json"

TEACHER_PROMPT = """You are an expert Community Health Worker trainer.
Rewrite the following medical response to fit a standardized CHW protocol.
The new response MUST:
1. Be structured, clear, and actionable.
2. Use simple language suitable for a layperson.
3. Explicitly state if a referral to a clinic is needed (Danger Signs).
4. Remove any "I am an AI" fluff.
5. Keep the medical accuracy of the original response but improve the delivery.

Structure:
**Assessment:** [Brief understanding of the issue]
**Action:** [Step-by-step guidance: Refer to Clinic OR Home Care instructions]
**Advice:** [Preventative measures or education]

Input Response:
"""

def refine_batch(client, batch, model="gpt-4o-mini"):
    refined_batch = []
    for item in batch:
        try:
            prompt = f"{TEACHER_PROMPT}\n{item['output']}\n\nUser Question: {item['input']}"
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            refined_output = response.choices[0].message.content.strip()
            
            # Update the item
            new_item = item.copy()
            new_item['output'] = refined_output
            new_item['is_refined'] = True
            refined_batch.append(new_item)
            
        except Exception as e:
            print(f"Error refining item: {e}")
            refined_batch.append(item) # Keep original on error
            
    return refined_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Number of examples to refine (0 = all)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    
    if not OpenAI:
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Refining first {args.limit} examples...")
        
    refined_data = []
    
    # Check for existing work
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE, 'r') as f:
                refined_data = json.load(f)
            print(f"Resuming from {len(refined_data)} examples")
            data = data[len(refined_data):]
        except:
            pass

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Process in batches of 1 for simplicity with progress bar
        futures = [executor.submit(refine_batch, client, [item], args.model) for item in data]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            refined_data.extend(result)
            
            if len(refined_data) % 50 == 0:
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(refined_data, f, indent=2)
                    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(refined_data, f, indent=2)
        
    print(f"✅ Refined dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
