#!/usr/bin/env python3
"""
Download and prepare medical datasets from Hugging Face for CHW assistant fine-tuning.
"""

import os
import json
from pathlib import Path
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    # Africa-specific datasets
    "afrimedqa": {
        "name": "intronhealth/afrimedqa_v2",
        "description": "Pan-African Medical QA - 15k questions from 16 African countries",
        "priority": "high",
        "requires_auth": True,
    },
    
    # General medical instruction datasets (no auth required)
    "medical_chatbot": {
        "name": "ruslanmv/ai-medical-chatbot",
        "description": "250k+ patient-doctor dialogues",
        "priority": "high",
        "requires_auth": False,
    },
    "symptom_diagnosis": {
        "name": "gretelai/symptom_to_diagnosis",
        "description": "Symptom to diagnosis mapping (22 conditions)",
        "priority": "high",
        "requires_auth": False,
    },
    "medmcqa": {
        "name": "openlifescienceai/medmcqa",
        "description": "194k medical MCQs from Indian exams",
        "priority": "medium",
        "requires_auth": False,
    },
    "pubmedqa": {
        "name": "qiaojin/PubMedQA",
        "description": "Biomedical research question answering",
        "priority": "medium",
        "requires_auth": False,
    },
    "chatdoctor": {
        "name": "lavita/ChatDoctor-HealthCareMagic-100k",
        "description": "112k medical instruction tuning conversations",
        "priority": "high",
        "requires_auth": False,
    },
    "medinstruct": {
        "name": "lavita/AlpaCare-MedInstruct-52k",
        "description": "52k medical instruction tuning examples",
        "priority": "medium",
        "requires_auth": False,
    },
    "wikidoc": {
        "name": "medalpaca/medical_meadow_wikidoc",
        "description": "Medical QA from WikiDoc (10k-100k)",
        "priority": "medium",
        "requires_auth": False,
    },
    "wikidoc_patient": {
        "name": "medalpaca/medical_meadow_wikidoc_patient_information",
        "description": "Patient information from WikiDoc",
        "priority": "medium",
        "requires_auth": False,
    },
    "diseases_symptoms": {
        "name": "QuyenAnhDE/Diseases_Symptoms",
        "description": "Disease-symptom mappings (400 entries)",
        "priority": "medium",
        "requires_auth": False,
    },
    "health_facts": {
        "name": "health_fact",
        "description": "Health fact verification dataset",
        "priority": "low",
        "requires_auth": False,
    },
}


def download_dataset(key: str, config: dict) -> bool:
    """Download a single dataset from Hugging Face."""
    print(f"\n{'='*60}")
    print(f"Downloading: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Priority: {config['priority']}")
    print(f"{'='*60}")
    
    output_dir = RAW_DIR / key
    output_dir.mkdir(exist_ok=True)
    
    try:
        if config.get("requires_auth"):
            print(f"⚠️  {key} requires authentication. Skipping for now.")
            print(f"   To download, run: huggingface-cli login")
            print(f"   Then request access at: https://huggingface.co/datasets/{config['name']}")
            return False
        
        # Load dataset
        print(f"Loading {config['name']}...")
        ds = load_dataset(config["name"], trust_remote_code=True)
        
        # Save each split
        for split_name, split_data in ds.items():
            output_file = output_dir / f"{split_name}.parquet"
            print(f"  Saving {split_name} ({len(split_data)} examples) -> {output_file}")
            split_data.to_parquet(output_file)
        
        # Save metadata
        metadata = {
            "source": config["name"],
            "description": config["description"],
            "priority": config["priority"],
            "splits": {name: len(data) for name, data in ds.items()},
            "columns": list(ds[list(ds.keys())[0]].column_names),
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Successfully downloaded {key}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {key}: {e}")
        return False


def download_all():
    """Download all configured datasets."""
    print("\n" + "="*60)
    print("CHW Assistant Dataset Downloader")
    print("="*60)
    
    results = {}
    
    # Sort by priority
    sorted_datasets = sorted(
        DATASETS.items(),
        key=lambda x: 0 if x[1]["priority"] == "high" else 1
    )
    
    for key, config in sorted_datasets:
        success = download_dataset(key, config)
        results[key] = success
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    print(f"\n✅ Successfully downloaded ({len(successful)}):")
    for k in successful:
        print(f"   - {k}: {DATASETS[k]['name']}")
    
    if failed:
        print(f"\n❌ Failed/Skipped ({len(failed)}):")
        for k in failed:
            print(f"   - {k}: {DATASETS[k]['name']}")
    
    print(f"\nData saved to: {RAW_DIR}")
    return results


if __name__ == "__main__":
    download_all()
