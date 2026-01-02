#!/usr/bin/env python3
"""
Analyze the curated dataset for translation needs and cost estimation.
"""

import json
import re
from pathlib import Path
from collections import Counter
import string

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "llama_finetune" / "train.json"
OUTPUT_DIR = DATA_DIR / "translation_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_stop_words():
    # Basic English stop words
    return set([
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
        'patient', 'doctor', 'help', 'please', 'thanks', 'hello', 'hi', 'thank', 'question', 'answer' # context specific
    ])

def analyze_dataset():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    total_examples = len(data)
    total_chars = 0
    total_words = 0
    
    # Frequency analysis
    word_freq = Counter()
    stop_words = get_stop_words()
    
    # Store just the unique terms to check for medical glossary candidates
    # Simple regex for words
    word_pattern = re.compile(r'\b[a-zA-Z]{3,}\b')
    
    print("Analyzing text...")
    for item in data:
        # Combine all text fields
        text = f"{item['instruction']} {item['input']} {item['output']}"
        total_chars += len(text)
        
        words = word_pattern.findall(text.lower())
        total_words += len(words)
        
        # Filter stop words and count
        filtered_words = [w for w in words if w not in stop_words]
        word_freq.update(filtered_words)
            
    print("\n=== DATASET STATISTICS ===")
    print(f"Total Examples: {total_examples:,}")
    print(f"Total Characters: {total_chars:,}")
    print(f"Total Words (approx): {total_words:,}")
    print(f"Avg Characters per Example: {total_chars / total_examples:.1f}")
    
    print("\n=== COST ESTIMATION (USD) ===")
    # Pricing estimates (approximate as of late 2024)
    google_trans_price = 20.0 # per million chars
    deepl_price = 25.0 # per million chars
    gpt4o_mini_price_input = 0.15 # per 1M tokens (approx 4M chars) -> very cheap for input
    gpt4o_mini_price_output = 0.60 # per 1M tokens
    
    # GPT-4o-mini estimate
    # Input tokens approx char/4
    # Output tokens approx char/4 (assuming 1:1 translation length)
    input_tokens = total_chars / 4
    output_tokens = total_chars / 4
    gpt4o_mini_cost = (input_tokens / 1_000_000 * gpt4o_mini_price_input) + (output_tokens / 1_000_000 * gpt4o_mini_price_output)
    
    print(f"Google Translate API (~$20/1M chars): ${total_chars / 1_000_000 * google_trans_price:.2f}")
    print(f"DeepL API (~$25/1M chars): ${total_chars / 1_000_000 * deepl_price:.2f}")
    print(f"GPT-4o-mini (estimated): ${gpt4o_mini_cost:.2f}")
    print("Note: LLM translation allows for glossary enforcement and context awareness.")

    print("\n=== POTENTIAL MEDICAL TERMS (Top 50 non-stop words) ===")
    # This is a naive list, a real medical NER would be better, but this gives a good overview
    common_words = word_freq.most_common(50)
    for word, count in common_words:
        print(f"{word}: {count:,}")
        
    # Save word list for glossary review
    glossary_candidates = [word for word, count in word_freq.most_common(2000)]
    with open(OUTPUT_DIR / "glossary_candidates.txt", "w") as f:
        f.write("\n".join(glossary_candidates))
    print(f"\nSaved top 2000 candidate terms to {OUTPUT_DIR}/glossary_candidates.txt")

if __name__ == "__main__":
    analyze_dataset()
