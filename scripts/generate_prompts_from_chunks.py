#!/usr/bin/env python3
"""
Generate CHW instruction/output samples from reference chunks using OpenAI.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Set, Tuple

from openai import OpenAI, OpenAIError

SYSTEM_PROMPT = """You are an expert curriculum designer for African Community Health Workers (CHWs).
Using the provided reference text, produce a single high-quality training example in JSON with:
- instruction: CHW-focused task framing (English, concise).
- input: Optional scenario details. Use "" if instruction already contains all context.
- output: Structured response with Assessment → Action → Advice sections, referencing the guidance.
- topic: short tag (e.g., PMTCT, Malaria, Nutrition).
- safety_notes: flag any referral/danger sign guidance mentioned.
Constraints:
- Stick to evidence directly in the provided chunk; do not invent dosages or policies not present.
- Highlight referral criteria/danger signs explicitly when present.
- Keep output under 280 words.
- Cite the source identifier in parentheses when relevant (e.g., “[Source: who_imci_chart]”).
Return ONLY valid JSON."""


def load_chunks(chunks_path: Path) -> Iterable[dict]:
    with chunks_path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_existing_records(output_path: Path) -> Set[Tuple[str, int]]:
    if not output_path.exists():
        return set()
    seen: Set[Tuple[str, int]] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (record.get("source"), record.get("chunk_id"))
            if None not in key:
                seen.add(key)
    return seen


def sample_chunks(chunks: Iterable[dict], per_source: int) -> List[dict]:
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk["source"], []).append(chunk)

    selected: List[dict] = []
    for source, items in grouped.items():
        random.shuffle(items)
        take = items[:per_source] if per_source else items
        selected.extend(take)

    return selected


def call_openai(client: OpenAI, chunk: dict, retries: int = 3, delay: float = 2.0) -> dict | None:
    prompt = chunk["text"]
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Source: {chunk['source']} (chunk {chunk['chunk_id']})\n\n{prompt}",
                    },
                ],
                temperature=0.2,
                max_output_tokens=600,
            )
            raw = response.output[0].content[0].text
            # Try to extract JSON from the raw response in case of extra text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                json_text = raw[start:end]
                return json.loads(json_text)
            else:
                raise ValueError("No JSON object found in response")
        except (OpenAIError, json.JSONDecodeError) as err:
            if attempt == retries:
                print(f"⚠️ Failed chunk {chunk['source']}#{chunk['chunk_id']}: {err}")
                return None
            time.sleep(delay * attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate prompts from reference chunks.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs" / "reference_chunks.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "reference_docs" / "reference_prompts.jsonl",
    )
    parser.add_argument("--limit", type=int, help="Max total samples per run.")
    parser.add_argument("--per-source", type=int, default=5, help="Limit per source document.")
    parser.add_argument("--append", dest="append", action="store_true", help="Append to existing prompts file.")
    parser.add_argument("--no-append", dest="append", action="store_false", help="Overwrite the prompts file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(append=True)
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    random.seed(args.seed)

    chunks = list(load_chunks(args.chunks_file))
    selected_chunks = sample_chunks(chunks, args.per_source)
    existing = load_existing_records(args.output_file) if args.append else set()
    filtered_chunks = [
        chunk for chunk in selected_chunks if (chunk["source"], chunk["chunk_id"]) not in existing
    ]
    random.shuffle(filtered_chunks)
    if args.limit:
        filtered_chunks = filtered_chunks[: args.limit]
    print(f"Generating samples for {len(filtered_chunks)} chunk(s)...")

    client = OpenAI(api_key=api_key)
    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    success = 0
    with output_path.open(mode, encoding="utf-8") as out_f:
        for chunk in filtered_chunks:
            result = call_openai(client, chunk)
            if not result:
                continue
            record = {
                "source": chunk["source"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                **result,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1
            print(f"✅ Generated sample from {chunk['source']}#{chunk['chunk_id']}")

    print(f"\nDone. Generated {success} samples saved to {output_path}")


if __name__ == "__main__":
    main()
