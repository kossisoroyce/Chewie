#!/usr/bin/env python3
"""
Extract readable text from reference PDFs to support prompt generation.
"""

import argparse
import sys
from pathlib import Path

from pdfminer.high_level import extract_text


def extract_pdf_text(pdf_path: Path, out_path: Path) -> int:
    """Extract text from a single PDF and save to out_path. Returns character count."""
    try:
        text = extract_text(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to extract {pdf_path.name}: {exc}") from exc

    out_path.write_text(text, encoding="utf-8")
    return len(text)


def main():
    parser = argparse.ArgumentParser(description="Extract text from reference PDFs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs",
        help="Directory containing source PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs" / "extracted_text",
        help="Directory where extracted .txt files will be written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .txt files.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(path for path in input_dir.glob("*.pdf") if path.stat().st_size > 0)

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        sys.exit(1)

    successes = 0
    failures = []

    for pdf_file in pdf_files:
        out_file = output_dir / (pdf_file.stem + ".txt")
        if out_file.exists() and not args.force:
            print(f"↺ Skipping {pdf_file.name} (already extracted)")
            continue

        try:
            chars = extract_pdf_text(pdf_file, out_file)
        except RuntimeError as err:
            failures.append(str(err))
            continue

        successes += 1
        print(f"✅ {pdf_file.name} -> {out_file.name} ({chars:,} chars)")

    print(f"\nDone. Extracted {successes} file(s).")
    if failures:
        print("⚠️ Failures:")
        for msg in failures:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
