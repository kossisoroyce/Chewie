#!/usr/bin/env python3
"""
Remove Personally Identifiable Information (PII) from the medical dataset.

Targets:
- Doctor names (Dr. X, Dr X)
- Signatures like "Regards, Dr. X"
- Email-like patterns
- Phone numbers
- Patient names when clearly identifiable
- Healthcare platform references
"""

import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_DIR = DATA_DIR / "africhw_medical_v1"
OUTPUT_DIR = DATA_DIR / "africhw_medical_v1_clean"

# Doctor name and signature patterns
DOCTOR_PATTERNS = [
    # Remove entire signature lines at end of text (with or without space)
    r'[Rr]egards[,.\s]*.{0,100}$',
    r'[Tt]ake [Cc]are[,.\s]*.{0,80}$',
    r'[Bb]est [Ww]ishes[,.\s]*.{0,80}$',
    r'[Hh]ope this helps[,.\s]*.{0,80}$',
    r'[Ww]ish you good health[,.\s]*.{0,80}$',
    r'[Gg]et well soon[,.\s]*.{0,80}$',
    # Dr. followed by name (strict to avoid "hydration" -> "hy[DOCTOR]")
    # Matches: Dr. Smith, Dr Smith, Dr. A. B. Jones
    r'\bDr(?:[\.\s])\s*(?:[A-Z](?:[a-z]+|\.)(?:\s+[A-Z](?:[a-z]+|\.)){0,4})(?:\s*[,;]\s*[^,;\n]{0,60})?',
]

# Platform/website patterns to remove
PLATFORM_PATTERNS = [
    r'\b(?:Health\s*Care\s*Magic|HCM|Healthcare\s*Magic|Chat\s*Doctor|ChatDoctor)\b',
    r'(?:www\.|https?://)[^\s]+',
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
]

# Phone number patterns
PHONE_PATTERNS = [
    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',
]

# Names to anonymize (common patterns in medical chat)
NAME_PATTERNS = [
    # "Dear X" or "Hello X" at start
    r'\b(?:Dear|Hello|Hi)\s+[A-Z][a-z]+\b',
    # "I am Dr. X" or "This is Dr. X"
    r'\b(?:I am|This is|I\'m)\s+(?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
]

# Replacement tokens
REPLACEMENTS = {
    "doctor": "[DOCTOR]",
    "patient": "[PATIENT]",
    "platform": "",
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "name": "[NAME]",
}


def remove_doctor_names(text: str) -> str:
    """Remove doctor names and signatures."""
    for pattern in DOCTOR_PATTERNS:
        text = re.sub(pattern, REPLACEMENTS["doctor"], text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def remove_platforms(text: str) -> str:
    """Remove platform references and URLs."""
    for pattern in PLATFORM_PATTERNS:
        text = re.sub(pattern, REPLACEMENTS["platform"], text, flags=re.IGNORECASE)
    return text


def remove_phones(text: str) -> str:
    """Remove phone numbers."""
    for pattern in PHONE_PATTERNS:
        text = re.sub(pattern, REPLACEMENTS["phone"], text)
    return text


def remove_names(text: str) -> str:
    """Remove common name patterns."""
    for pattern in NAME_PATTERNS:
        text = re.sub(pattern, lambda m: m.group(0).split()[0] + " " + REPLACEMENTS["name"], text)
    return text


def clean_whitespace(text: str) -> str:
    """Clean up extra whitespace left by removals."""
    # Multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Trailing/leading whitespace
    text = text.strip()
    return text


def clean_text(text: str) -> str:
    """Apply all PII removal steps."""
    if not isinstance(text, str):
        return text
    
    text = remove_doctor_names(text)
    text = remove_platforms(text)
    text = remove_phones(text)
    text = remove_names(text)
    text = clean_whitespace(text)
    
    return text


def clean_conversation(conversation: list) -> list:
    """Clean all turns in a conversation."""
    cleaned = []
    for turn in conversation:
        cleaned.append({
            "role": turn["role"],
            "content": clean_text(turn["content"])
        })
    return cleaned


def clean_dataset(input_file: Path, output_file: Path):
    """Clean a single dataset file."""
    print(f"  Loading {input_file.name}...")
    df = pd.read_parquet(input_file)
    
    print(f"  Cleaning {len(df):,} records...")
    
    # Clean conversations
    tqdm.pandas(desc="  Cleaning")
    df["conversation"] = df["conversation"].progress_apply(clean_conversation)
    
    # Clean system prompts if needed
    df["system_prompt"] = df["system_prompt"].apply(clean_text)
    
    # Save
    df.to_parquet(output_file, index=False)
    print(f"  Saved to {output_file.name}")
    
    return len(df)


def main():
    print("="*60)
    print("PII REMOVAL FROM AFRICHW-MEDICAL DATASET")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ["train", "validation", "test"]
    total = 0
    
    for split in splits:
        input_file = INPUT_DIR / f"{split}.parquet"
        output_file = OUTPUT_DIR / f"{split}.parquet"
        
        if input_file.exists():
            print(f"\n📄 Processing {split}...")
            count = clean_dataset(input_file, output_file)
            total += count
    
    # Copy metadata and README
    import shutil
    for file in ["metadata.json", "README.md"]:
        src = INPUT_DIR / file
        if src.exists():
            shutil.copy(src, OUTPUT_DIR / file)
    
    # Create sample
    print("\n📄 Creating clean sample...")
    df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    df.head(1000).to_json(OUTPUT_DIR / "train_sample.jsonl", orient="records", lines=True)
    
    # Show examples of cleaning
    print("\n" + "="*60)
    print("CLEANING EXAMPLES")
    print("="*60)
    
    test_strings = [
        "Thanks for your question. Regards, Dr. Parth Goswami, General & Family Physician",
        "Hello and welcome to HealthCareMagic. I have reviewed your query.",
        "Contact me at doctor@hospital.com or call 555-123-4567",
        "I am Dr. Smith and I recommend...",
    ]
    
    for s in test_strings:
        cleaned = clean_text(s)
        print(f"\nBefore: {s}")
        print(f"After:  {cleaned}")
    
    print(f"\n✅ Cleaned {total:,} records")
    print(f"📁 Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
