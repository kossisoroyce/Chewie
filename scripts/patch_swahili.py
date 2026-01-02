#!/usr/bin/env python3
"""Patch the Swahili translation dataset by replacing leftover PII placeholders."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "translated_dataset"
INPUT_FILE = DATA_DIR / "train_swahili.json"
OUTPUT_FILE = DATA_DIR / "train_swahili_patched.json"

REPLACEMENTS = {
    "[DOCTOR]": "mtaalamu wa afya",
    "[NAME]": "mgonjwa",
    "[PATIENT]": "mgonjwa",
    "[PHONE]": "nambari ya simu",
    "[EMAIL]": "anuani ya barua pepe",
}


def main():
    print("🔧 Patching Swahili dataset...")

    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records")

    patched = []
    placeholder_hits = 0
    for item in data:
        new_item = item.copy()
        out = new_item.get("output", "")
        for old, new in REPLACEMENTS.items():
            if old in out:
                placeholder_hits += out.count(old)
                out = out.replace(old, new)
        new_item["output"] = out
        patched.append(new_item)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(patched, f, ensure_ascii=False, indent=2)

    remaining = sum(1 for item in patched if any(tok in item.get("output", "") for tok in REPLACEMENTS))

    print(f"Patched placeholders: {placeholder_hits}")
    print(f"Remaining placeholders: {remaining}")
    print(f"✅ Saved patched dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
