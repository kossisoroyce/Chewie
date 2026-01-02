#!/usr/bin/env python3
"""Merge refined English and patched Swahili datasets into a single bilingual split."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
EN_FILE = DATA_DIR / "llama_finetune" / "train_refined_patched.json"
SW_FILE = DATA_DIR / "translated_dataset" / "train_swahili_patched.json"
OUT_DIR = DATA_DIR / "llama_finetune"
TRAIN_OUT = OUT_DIR / "train_bilingual.json"
VAL_OUT = OUT_DIR / "validation_bilingual.json"
VAL_RATIO = 0.05
SEED = 42


def load_records(path, language):
    with open(path) as f:
        data = json.load(f)
    for item in data:
        item = item.copy()
        item["language"] = language
        yield item


def main():
    print("📚 Loading English dataset...")
    english = list(load_records(EN_FILE, "en"))
    print(f"  English records: {len(english):,}")

    print("📚 Loading Swahili dataset...")
    swahili = list(load_records(SW_FILE, "sw"))
    print(f"  Swahili records: {len(swahili):,}")

    combined = english + swahili
    total = len(combined)
    print(f"🔄 Combined total: {total:,}")

    random.Random(SEED).shuffle(combined)

    val_size = max(1, int(total * VAL_RATIO))
    val_records = combined[:val_size]
    train_records = combined[val_size:]

    print(f"🧪 Validation: {len(val_records):,} ({VAL_RATIO*100:.1f}% of total)")
    print(f"🚆 Training: {len(train_records):,}")

    TRAIN_OUT.write_text(json.dumps(train_records, ensure_ascii=False, indent=2))
    VAL_OUT.write_text(json.dumps(val_records, ensure_ascii=False, indent=2))

    english_share = sum(1 for r in combined if r.get("language") == "en")
    swahili_share = total - english_share

    summary = {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "english_total": english_share,
        "swahili_total": swahili_share,
        "english_train": sum(1 for r in train_records if r.get("language") == "en"),
        "swahili_train": sum(1 for r in train_records if r.get("language") == "sw"),
        "english_val": sum(1 for r in val_records if r.get("language") == "en"),
        "swahili_val": sum(1 for r in val_records if r.get("language") == "sw"),
    }

    meta_path = OUT_DIR / "bilingual_metadata.json"
    meta_path.write_text(json.dumps(summary, indent=2))

    print("✅ Saved bilingual datasets:")
    print(f"   Train -> {TRAIN_OUT}")
    print(f"   Val   -> {VAL_OUT}")
    print(f"📄 Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
