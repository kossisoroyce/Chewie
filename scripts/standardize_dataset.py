#!/usr/bin/env python3
"""
Standardize the combined medical dataset into a clean, independent dataset
for CHW (Community Health Worker) assistant fine-tuning.

Output: AfriCHW-Medical dataset
"""

import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_DIR / "processed" / "master_dataset.parquet"
OUTPUT_DIR = DATA_DIR / "africhw_medical_v1"

# Dataset metadata
DATASET_NAME = "AfriCHW-Medical"
DATASET_VERSION = "1.0.0"
DATASET_DESCRIPTION = """
AfriCHW-Medical is a curated instruction-tuning dataset designed for fine-tuning 
LLMs to serve as Community Health Worker (CHW) assistants in African healthcare contexts.

The dataset combines multiple medical QA sources with African-specific medical content,
standardized into a consistent format suitable for instruction fine-tuning.
"""

# Standardized schema
SCHEMA = {
    "id": "Unique identifier for each record",
    "conversation": "List of conversation turns [{role, content}]",
    "system_prompt": "System prompt for the conversation",
    "category": "Primary category of the content",
    "subcategory": "More specific categorization",
    "source": "Original dataset source",
    "language": "Language of the content",
    "difficulty": "Estimated difficulty level",
    "tags": "List of relevant tags",
}

# Category mapping
CATEGORY_MAP = {
    "patient_consultation": ("clinical", "consultation"),
    "medical_mcq": ("education", "mcq"),
    "medical_instruction": ("clinical", "instruction"),
    "medical_qa": ("education", "qa"),
    "african_medical_qa": ("clinical", "african_context"),
    "diagnosis": ("clinical", "diagnosis"),
    "chw_triage": ("chw", "triage"),
    "chw_prevention": ("chw", "prevention"),
}

# System prompts by category
SYSTEM_PROMPTS = {
    "clinical": "You are a knowledgeable medical assistant helping healthcare workers and patients. Provide accurate, helpful medical information while being mindful of when to recommend professional medical consultation.",
    "education": "You are a medical education assistant helping healthcare students and professionals learn medical concepts. Provide clear, accurate explanations.",
    "chw": "You are a Community Health Worker (CHW) assistant supporting frontline healthcare workers in resource-limited settings. Provide practical, actionable guidance appropriate for community health contexts in Africa.",
}


def generate_id(text: str) -> str:
    """Generate a unique ID from content."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove null bytes
    text = text.replace('\x00', '')
    return text


def estimate_difficulty(text: str, category: str) -> str:
    """Estimate difficulty level based on content."""
    text_lower = text.lower()
    
    # Medical terminology indicators
    advanced_terms = ['pathophysiology', 'differential diagnosis', 'contraindication',
                      'pharmacokinetics', 'etiology', 'prognosis', 'histopathology']
    intermediate_terms = ['diagnosis', 'treatment', 'symptoms', 'medication', 
                          'prescription', 'examination']
    
    advanced_count = sum(1 for term in advanced_terms if term in text_lower)
    intermediate_count = sum(1 for term in intermediate_terms if term in text_lower)
    
    if advanced_count >= 2 or category == "mcq":
        return "advanced"
    elif intermediate_count >= 2:
        return "intermediate"
    else:
        return "basic"


def extract_tags(text: str, category: str, source: str) -> List[str]:
    """Extract relevant tags from content."""
    tags = [source]
    text_lower = text.lower()
    
    # Medical topic tags
    topic_keywords = {
        "malaria": ["malaria", "plasmodium"],
        "tuberculosis": ["tuberculosis", "tb", "mycobacterium"],
        "hiv_aids": ["hiv", "aids", "antiretroviral"],
        "maternal_health": ["pregnancy", "prenatal", "maternal", "obstetric", "childbirth"],
        "pediatrics": ["child", "pediatric", "infant", "neonatal"],
        "cardiovascular": ["heart", "cardiac", "hypertension", "blood pressure"],
        "respiratory": ["lung", "respiratory", "breathing", "pneumonia"],
        "infectious_disease": ["infection", "bacteria", "virus", "parasite"],
        "nutrition": ["nutrition", "malnutrition", "diet", "vitamin"],
        "mental_health": ["mental", "depression", "anxiety", "psychiatric"],
    }
    
    for tag, keywords in topic_keywords.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(tag)
    
    # Context tags
    if "africa" in text_lower or source in ["afrimedqa", "chw_synthetic"]:
        tags.append("african_context")
    
    return list(set(tags))


def convert_to_conversation(row: Dict) -> List[Dict]:
    """Convert instruction/input/output format to conversation format."""
    conversation = []
    
    # User turn
    user_content = row.get("instruction", "")
    input_text = row.get("input", "")
    
    if input_text:
        if user_content:
            user_content = f"{user_content}\n\n{input_text}"
        else:
            user_content = input_text
    
    conversation.append({
        "role": "user",
        "content": clean_text(user_content)
    })
    
    # Assistant turn
    output = row.get("output", "")
    conversation.append({
        "role": "assistant", 
        "content": clean_text(output)
    })
    
    return conversation


def standardize_record(row: Dict) -> Optional[Dict]:
    """Convert a raw record to standardized format."""
    # Get original values
    instruction = str(row.get("instruction", ""))
    input_text = str(row.get("input", ""))
    output = str(row.get("output", ""))
    source = str(row.get("source", "unknown"))
    orig_category = str(row.get("category", "general"))
    
    # Skip empty records
    if not output or len(output.strip()) < 10:
        return None
    
    # Clean texts
    instruction = clean_text(instruction)
    input_text = clean_text(input_text)
    output = clean_text(output)
    
    # Generate ID
    content_hash = f"{instruction}{input_text}{output}"
    record_id = generate_id(content_hash)
    
    # Map category
    category, subcategory = CATEGORY_MAP.get(orig_category, ("general", "other"))
    
    # Get system prompt
    system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["clinical"])
    
    # Build conversation
    conversation = convert_to_conversation({
        "instruction": instruction,
        "input": input_text,
        "output": output
    })
    
    # Extract metadata
    full_text = f"{instruction} {input_text} {output}"
    difficulty = estimate_difficulty(full_text, subcategory)
    tags = extract_tags(full_text, subcategory, source)
    
    return {
        "id": record_id,
        "conversation": conversation,
        "system_prompt": system_prompt,
        "category": category,
        "subcategory": subcategory,
        "source": source,
        "language": "en",
        "difficulty": difficulty,
        "tags": tags,
    }


def create_splits(df: pd.DataFrame, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """Create train/validation/test splits with stratification by source."""
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=df["source"]
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=42,
        stratify=temp_df["source"]
    )
    
    return train_df, val_df, test_df


def create_dataset_card() -> str:
    """Create a dataset card in markdown format."""
    return f"""---
license: cc-by-sa-4.0
language:
- en
tags:
- medical
- healthcare
- african
- community-health
- instruction-tuning
- chw
size_categories:
- 100K<n<1M
task_categories:
- text-generation
- question-answering
---

# {DATASET_NAME}

## Dataset Description

{DATASET_DESCRIPTION}

### Dataset Summary

- **Version:** {DATASET_VERSION}
- **Created:** {datetime.now().strftime("%Y-%m-%d")}
- **License:** CC-BY-SA-4.0
- **Language:** English

### Intended Use

This dataset is designed for fine-tuning large language models to serve as:
- Community Health Worker (CHW) assistants
- Medical education tools
- Clinical decision support systems
- Patient health information chatbots

**Important:** This dataset is for research and educational purposes. Any medical AI system should be validated by healthcare professionals before clinical use.

### Schema

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `conversation` | list | List of turns with `role` and `content` |
| `system_prompt` | string | Suggested system prompt |
| `category` | string | Primary category (clinical/education/chw) |
| `subcategory` | string | Specific subcategory |
| `source` | string | Original dataset source |
| `language` | string | Language code |
| `difficulty` | string | Estimated difficulty (basic/intermediate/advanced) |
| `tags` | list | Relevant topic tags |

### Data Splits

| Split | Records | Purpose |
|-------|---------|---------|
| train | ~90% | Model training |
| validation | ~5% | Hyperparameter tuning |
| test | ~5% | Final evaluation |

### Sources

This dataset combines and standardizes data from:
- AfriMed-QA (African Medical QA)
- MedQA-USMLE
- ChatDoctor
- MedMCQA
- MedInstruct
- WikiDoc Medical
- Symptom-Diagnosis datasets
- CHW-specific synthetic examples

### Citation

```bibtex
@dataset{{africhw_medical_2024,
  title = {{{DATASET_NAME}: A Medical Instruction Dataset for African Healthcare}},
  year = {{2024}},
  version = {{{DATASET_VERSION}}}
}}
```

### Limitations

- Primarily English language
- Medical information should be verified by professionals
- May contain biases from source datasets
- Not a substitute for professional medical advice
"""


def main():
    """Main standardization pipeline."""
    print("="*60)
    print(f"CREATING STANDARDIZED DATASET: {DATASET_NAME} v{DATASET_VERSION}")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    print("\n📂 Loading combined dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Loaded {len(df):,} records")
    
    # Standardize records
    print("\n🔄 Standardizing records...")
    standardized = []
    skipped = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Processing"):
        record = standardize_record(row.to_dict())
        if record:
            standardized.append(record)
        else:
            skipped += 1
    
    print(f"   Standardized: {len(standardized):,}")
    print(f"   Skipped (empty/invalid): {skipped:,}")
    
    # Create DataFrame
    std_df = pd.DataFrame(standardized)
    
    # Remove duplicates by ID
    initial = len(std_df)
    std_df = std_df.drop_duplicates(subset=["id"])
    print(f"   Duplicates removed: {initial - len(std_df):,}")
    print(f"   Final count: {len(std_df):,}")
    
    # Create splits
    print("\n📊 Creating train/val/test splits...")
    try:
        train_df, val_df, test_df = create_splits(std_df)
    except Exception as e:
        print(f"   Stratified split failed ({e}), using random split...")
        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(std_df, test_size=0.1, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"   Train: {len(train_df):,}")
    print(f"   Validation: {len(val_df):,}")
    print(f"   Test: {len(test_df):,}")
    
    # Save splits
    print("\n💾 Saving dataset...")
    
    # Parquet format (efficient)
    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "validation.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    
    # JSONL format (for small sample)
    train_df.head(1000).to_json(OUTPUT_DIR / "train_sample.jsonl", orient="records", lines=True)
    
    # Save dataset card
    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(create_dataset_card())
    
    # Save metadata
    metadata = {
        "name": DATASET_NAME,
        "version": DATASET_VERSION,
        "created": datetime.now().isoformat(),
        "splits": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "total_records": len(std_df),
        "schema": SCHEMA,
        "categories": std_df["category"].value_counts().to_dict(),
        "sources": std_df["source"].value_counts().to_dict(),
        "difficulties": std_df["difficulty"].value_counts().to_dict(),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("DATASET CREATED SUCCESSFULLY")
    print("="*60)
    print(f"\n📁 Location: {OUTPUT_DIR}")
    print(f"\n📊 Statistics:")
    print(f"   Total records: {len(std_df):,}")
    print(f"\n   By category:")
    for cat, count in std_df["category"].value_counts().items():
        print(f"      {cat}: {count:,}")
    print(f"\n   By difficulty:")
    for diff, count in std_df["difficulty"].value_counts().items():
        print(f"      {diff}: {count:,}")
    print(f"\n📄 Files created:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/1024/1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"      {f.name}: {size_str}")


if __name__ == "__main__":
    main()
