#!/usr/bin/env python3
"""
Translation pipeline for AfriCHW-Medical dataset using LLMs (GPT-4o-mini/Gemini).
Features:
- Glossary enforcement for medical consistency
- Cost-effective batch processing
- Progress tracking and resumption
- Parallel processing
"""

import os
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Imports for API (assumes OpenAI compatible SDK)
try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    OpenAI = None

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train.json"
OUTPUT_DIR = DATA_DIR / "translated_dataset"
GLOSSARY_FILE = DATA_DIR / "translation_analysis" / "swahili_medical_glossary.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompts
SYSTEM_PROMPT = """You are an expert English-to-Swahili medical translator. 
Translate the following medical conversation while strictly adhering to these rules:
1. Use standard Swahili used in East African medical settings.
2. For medical terms, PREFER the provided glossary terms if applicable.
3. Maintain the professional yet accessible tone of a health worker.
4. Preserve all formatting, special tokens, and JSON structure.
5. Do NOT translate system tokens (like <|begin_of_text|>).

Input format: JSON object with 'instruction', 'input', 'output'.
Output format: JSON object with 'instruction', 'input', 'output' (all values translated to Swahili)."""

def load_glossary():
    if GLOSSARY_FILE.exists():
        with open(GLOSSARY_FILE, 'r') as f:
            return json.load(f)
    return {}

def translate_batch(client, batch, glossary_text, model="gpt-4o-mini"):
    """Translate a batch of examples."""
    translations = []
    
    # We process one by one in the batch for higher quality control, 
    # or we could combine them into one prompt for speed (but risk alignment issues).
    # With GPT-4o-mini being so cheap, individual requests or small groups are fine.
    
    for item in batch:
        retries = 3
        delay = 5
        success = False
        
        while retries > 0 and not success:
            try:
                # Construct the prompt
                user_content = f"""
Glossary Context:
{glossary_text}

Translate this JSON object to Swahili:
{json.dumps(item, ensure_ascii=False)}
"""
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                translated_item = json.loads(response.choices[0].message.content)
                
                # Preserve metadata if any
                if 'source' in item:
                    translated_item['source'] = item['source']
                if 'id' in item:
                    translated_item['id'] = item['id']
                    
                translations.append(translated_item)
                success = True
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    print(f"Rate limit hit. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries -= 1
                else:
                    print(f"Error translating item: {e}")
                    # Return original on error to preserve data count, can filter later
                    item['translation_error'] = str(e)
                    translations.append(item)
                    break # Don't retry non-rate-limit errors
        
        if not success and retries == 0:
            print("Max retries reached for item.")
            item['translation_error'] = "Rate limit max retries exceeded"
            translations.append(item)
            
    return translations

def main():
    parser = argparse.ArgumentParser(description="Translate dataset to Swahili")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 for all)")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--input", type=str, default=str(INPUT_FILE), help="Input JSON file path")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--retry", action="store_true", help="Retry items with translation errors")
    args = parser.parse_args()

    if not OpenAI:
        return

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_swahili.json"

    # RETRY MODE
    if args.retry:
        if not output_file.exists():
            print(f"Error: Output file {output_file} does not exist. Cannot retry.")
            return
            
        print(f"Loading existing translations from {output_file} for retry...")
        with open(output_file, 'r') as f:
            translated_data = json.load(f)
            
        # Identify failed items
        failed_indices = [i for i, item in enumerate(translated_data) if 'translation_error' in item]
        print(f"Found {len(failed_indices)} items with errors to retry.")
        
        if not failed_indices:
            print("No errors found.")
            return
            
        # Load glossary once
        glossary = load_glossary()
        
        # Process retries
        print(f"Retrying {len(failed_indices)} items with {args.workers} workers...")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {}
            for idx in failed_indices:
                item = translated_data[idx]
                # Clean up previous error
                if 'translation_error' in item:
                    del item['translation_error']
                
                # Use input text from item if available, or try to reconstruct/use existing fields
                # Note: The item in translated_data might be the original English one if it failed completely
                # and we appended it in the except block.
                
                # Dynamic glossary logic (duplicated from main loop, could be refactored)
                text_content = (item.get('input', '') + ' ' + item.get('output', '')).lower()
                relevant_terms = {}
                for term, trans in glossary.items():
                    if term in text_content:
                        relevant_terms[term] = trans
                if len(relevant_terms) > 50:
                    relevant_terms = dict(list(relevant_terms.items())[:50])
                glossary_text = "\n".join([f"{k} -> {v}" for k,v in relevant_terms.items()]) or "No specific glossary terms found."

                future = executor.submit(translate_batch, client, [item], glossary_text, args.model)
                future_to_index[future] = idx
            
            for future in tqdm(as_completed(future_to_index), total=len(failed_indices)):
                idx = future_to_index[future]
                try:
                    results = future.result()
                    if results and len(results) > 0:
                        translated_data[idx] = results[0]
                except Exception as e:
                    print(f"Retry failed for index {idx}: {e}")
                    
                # Periodic save during retry
                if len(future_to_index) % 50 == 0:
                     with open(output_file, 'w') as f:
                        json.dump(translated_data, f, indent=2, ensure_ascii=False)

        # Final save after retry
        with open(output_file, 'w') as f:
            json.dump(translated_data, f, indent=2, ensure_ascii=False)
            
        print("Retry complete.")
        return

    # NORMAL MODE (New Translation or Resume Append)
    # Load data
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limiting to {args.limit} examples")

    # Load glossary
    glossary = load_glossary()
    
    print(f"Starting translation of {len(data)} examples...")
    print(f"Model: {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Glossary terms: {len(glossary)}")
    
    translated_data = []
    
    # Resume capability
    if output_file.exists():
        print("Found existing partial translation. Loading...")
        with open(output_file, 'r') as f:
            try:
                translated_data = json.load(f)
                print(f"Resuming from {len(translated_data)} examples")
                data = data[len(translated_data):]
            except json.JSONDecodeError:
                print("Could not read existing file. Starting over.")

    # Process in chunks
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for item in data:
            # Dynamic glossary extraction
            # Find which glossary terms appear in this specific example
            text_content = (item.get('input', '') + ' ' + item.get('output', '')).lower()
            relevant_terms = {}
            for term, trans in glossary.items():
                if term in text_content:
                    relevant_terms[term] = trans
            
            # If too many, cap at 50 to save context window
            if len(relevant_terms) > 50:
                # Prioritize longer terms (likely more specific) or just take first 50
                relevant_terms = dict(list(relevant_terms.items())[:50])
            
            glossary_text = "\n".join([f"{k} -> {v}" for k,v in relevant_terms.items()])
            if not glossary_text:
                glossary_text = "No specific glossary terms found."

            futures.append(executor.submit(translate_batch, client, [item], glossary_text, args.model))
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            translated_data.extend(result)
            
            # Periodic save (every 100)
            if len(translated_data) % 100 == 0:
                with open(output_file, 'w') as f:
                    json.dump(translated_data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(output_file, 'w') as f:
        json.dump(translated_data, f, indent=2, ensure_ascii=False)
        
    # Also save as JSONL
    with open(OUTPUT_DIR / "train_swahili.jsonl", 'w') as f:
        for item in translated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Translation complete. Saved to {output_file}")

if __name__ == "__main__":
    main()
