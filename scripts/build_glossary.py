#!/usr/bin/env python3
"""
Build a Swahili medical glossary using an LLM (GPT-4o-mini or Gemini).
This script takes the extracted terms and generates standardized Swahili translations.
"""

import os
import json
import time
from pathlib import Path
from tqdm import tqdm

# You would install openai or google-generativeai
# pip install openai

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    OpenAI = None

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_TERMS = DATA_DIR / "translation_analysis" / "glossary_candidates.txt"
OUTPUT_GLOSSARY = DATA_DIR / "translation_analysis" / "swahili_medical_glossary.json"

SYSTEM_PROMPT = """You are a strictly accurate medical translator for English to Swahili.
Your goal is to provide the most common, medically accurate Swahili translation for the given English medical term.
If a direct Swahili term doesn't exist, provide the commonly used loan word or descriptive phrase used in East African medical settings.
Format response as JSON: {"english": "term", "swahili": "translation", "notes": "optional context"}"""

def build_glossary():
    if not INPUT_TERMS.exists():
        print(f"Error: {INPUT_TERMS} not found. Run analysis script first.")
        return

    # Load terms
    with open(INPUT_TERMS, 'r') as f:
        terms = [line.strip() for line in f if line.strip()]  # Process ALL terms
    
    print(f"Generating glossary for {len(terms)} terms...")
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if OpenAI else None
    
    if not client:
        print("⚠️ OpenAI client not available. Creating template only.")
        # Create a dummy template for manual filling if API not present
        glossary = {}
        for term in terms[:10]:
            glossary[term] = "TODO_TRANSLATE"
        
        with open(OUTPUT_GLOSSARY, 'w') as f:
            json.dump(glossary, f, indent=2)
        print(f"Created template at {OUTPUT_GLOSSARY}")
        return

    glossary = {}
    
    # Process in batches to save time/calls if needed, but term-by-term is safer for quality
    for term in tqdm(terms):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Translate: {term}"}
                ],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            glossary[term] = result['swahili']
            
        except Exception as e:
            print(f"Error translating {term}: {e}")
            glossary[term] = "ERROR"
            time.sleep(1)

    # Save
    with open(OUTPUT_GLOSSARY, 'w') as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Glossary saved to {OUTPUT_GLOSSARY}")

if __name__ == "__main__":
    build_glossary()
