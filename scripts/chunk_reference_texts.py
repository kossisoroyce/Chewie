#!/usr/bin/env python3
"""
Chunk extracted reference texts into manageable segments with metadata.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List


def normalize_source_name(stem: str) -> str:
    """Generate a compact source identifier from filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return cleaned.lower()


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (double newline)."""
    # Remove repeated form-feed markers, trim whitespace
    text = text.replace("\f", "\n").strip()
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [para.strip() for para in raw_paragraphs if para.strip()]


def chunk_paragraphs(paragraphs: Iterable[str], chunk_size: int, overlap: int) -> List[str]:
    """Group paragraphs into chunks ~chunk_size characters with optional overlap."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if current and current_len + len(para) + 1 > chunk_size:
            chunks.append("\n\n".join(current))
            if overlap and len(current) > 1:
                overlap_items = current[-overlap:]
            else:
                overlap_items = []
            current = list(overlap_items)
            current_len = sum(len(p) + 2 for p in current)

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_file(txt_path: Path, chunk_size: int, overlap: int) -> List[dict]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = split_paragraphs(text)
    raw_chunks = chunk_paragraphs(paragraphs, chunk_size, overlap)
    source_id = normalize_source_name(txt_path.stem)

    chunk_entries = []
    for idx, chunk_text in enumerate(raw_chunks, start=1):
        entry = {
            "source": source_id,
            "source_file": txt_path.name,
            "chunk_id": idx,
            "text": chunk_text.strip(),
            "char_len": len(chunk_text),
        }
        chunk_entries.append(entry)

    return chunk_entries


def main():
    parser = argparse.ArgumentParser(description="Chunk extracted reference texts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs" / "extracted_text",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs" / "reference_chunks.jsonl",
    )
    parser.add_argument("--chunk-size", type=int, default=1500, help="Approximate chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=1, help="Paragraph overlap between consecutive chunks.")
    parser.add_argument("--limit", type=int, help="Limit number of files (for debugging).")
    args = parser.parse_args()

    txt_files = sorted(p for p in args.input_dir.glob("*.txt") if p.stat().st_size > 0)
    if args.limit:
        txt_files = txt_files[: args.limit]

    total_chunks = 0
    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for txt_file in txt_files:
            chunks = chunk_file(txt_file, args.chunk_size, args.overlap)
            for entry in chunks:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"📚 {txt_file.name}: {len(chunks)} chunk(s)")

    print(f"\n✅ Wrote {total_chunks} chunks to {output_path}")


if __name__ == "__main__":
    main()
