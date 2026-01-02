#!/usr/bin/env python3
"""
Combine and standardize all downloaded medical datasets into a master dataset
for CHW assistant fine-tuning.

Target format (instruction-tuning):
- instruction: The task/question
- input: Additional context (can be empty)
- output: The expected response
- source: Dataset source identifier
- category: Type of medical content
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def process_chatdoctor(raw_dir: Path) -> List[Dict]:
    """Process ChatDoctor dataset (instruction, input, output format)."""
    records = []
    parquet_file = raw_dir / "chatdoctor" / "train.parquet"
    
    if not parquet_file.exists():
        print("  ⚠️ ChatDoctor not found, skipping")
        return records
    
    df = pd.read_parquet(parquet_file)
    print(f"  Processing {len(df)} ChatDoctor examples...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  ChatDoctor"):
        records.append({
            "instruction": row["instruction"],
            "input": row["input"] if pd.notna(row["input"]) else "",
            "output": row["output"],
            "source": "chatdoctor",
            "category": "patient_consultation"
        })
    
    return records


def process_medical_chatbot(raw_dir: Path) -> List[Dict]:
    """Process AI Medical Chatbot dataset (Patient, Doctor dialogues)."""
    records = []
    parquet_file = raw_dir / "medical_chatbot" / "train.parquet"
    
    if not parquet_file.exists():
        print("  ⚠️ Medical Chatbot not found, skipping")
        return records
    
    df = pd.read_parquet(parquet_file)
    print(f"  Processing {len(df)} Medical Chatbot examples...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  MedChatbot"):
        # Convert to instruction format
        instruction = "You are a medical assistant. A patient describes their symptoms. Provide helpful medical guidance."
        patient_query = row["Patient"] if pd.notna(row["Patient"]) else ""
        description = row["Description"] if pd.notna(row["Description"]) else ""
        
        input_text = patient_query
        if description and description != patient_query:
            input_text = f"Context: {description}\n\nPatient: {patient_query}"
        
        records.append({
            "instruction": instruction,
            "input": input_text,
            "output": row["Doctor"] if pd.notna(row["Doctor"]) else "",
            "source": "medical_chatbot",
            "category": "patient_consultation"
        })
    
    return records


def process_medinstruct(raw_dir: Path) -> List[Dict]:
    """Process MedInstruct dataset (instruction tuning format)."""
    records = []
    parquet_file = raw_dir / "medinstruct" / "train.parquet"
    
    if not parquet_file.exists():
        print("  ⚠️ MedInstruct not found, skipping")
        return records
    
    df = pd.read_parquet(parquet_file)
    print(f"  Processing {len(df)} MedInstruct examples...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  MedInstruct"):
        records.append({
            "instruction": row["instruction"] if pd.notna(row["instruction"]) else "",
            "input": row["input"] if pd.notna(row["input"]) else "",
            "output": row["output"] if pd.notna(row["output"]) else "",
            "source": "medinstruct",
            "category": "medical_instruction"
        })
    
    return records


def process_medmcqa(raw_dir: Path) -> List[Dict]:
    """Process MedMCQA dataset (medical MCQs)."""
    records = []
    dataset_dir = raw_dir / "medmcqa"
    
    if not dataset_dir.exists():
        print("  ⚠️ MedMCQA not found, skipping")
        return records
    
    # Load all splits
    for split in ["train", "validation"]:  # Skip test (no answers)
        parquet_file = dataset_dir / f"{split}.parquet"
        if not parquet_file.exists():
            continue
            
        df = pd.read_parquet(parquet_file)
        print(f"  Processing {len(df)} MedMCQA {split} examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  MedMCQA-{split}"):
            # Build MCQ format
            question = row["question"]
            options = [
                f"A) {row['opa']}" if pd.notna(row['opa']) else "",
                f"B) {row['opb']}" if pd.notna(row['opb']) else "",
                f"C) {row['opc']}" if pd.notna(row['opc']) else "",
                f"D) {row['opd']}" if pd.notna(row['opd']) else "",
            ]
            options_text = "\n".join([o for o in options if o])
            
            # Get correct answer
            cop = row["cop"]  # 0=A, 1=B, 2=C, 3=D
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            correct_letter = answer_map.get(cop, "A")
            correct_text = [row['opa'], row['opb'], row['opc'], row['opd']][cop] if pd.notna(cop) else ""
            
            explanation = row["exp"] if pd.notna(row["exp"]) else ""
            subject = row["subject_name"] if pd.notna(row["subject_name"]) else ""
            topic = row["topic_name"] if pd.notna(row["topic_name"]) else ""
            
            # Build output
            output = f"The correct answer is {correct_letter}) {correct_text}"
            if explanation:
                output += f"\n\nExplanation: {explanation}"
            
            records.append({
                "instruction": f"Answer the following medical question. Subject: {subject}, Topic: {topic}",
                "input": f"{question}\n\n{options_text}",
                "output": output,
                "source": "medmcqa",
                "category": "medical_mcq"
            })
    
    return records


def process_symptom_diagnosis(raw_dir: Path) -> List[Dict]:
    """Process Symptom to Diagnosis dataset."""
    records = []
    dataset_dir = raw_dir / "symptom_diagnosis"
    
    if not dataset_dir.exists():
        print("  ⚠️ Symptom Diagnosis not found, skipping")
        return records
    
    for split in ["train", "test"]:
        parquet_file = dataset_dir / f"{split}.parquet"
        if not parquet_file.exists():
            continue
            
        df = pd.read_parquet(parquet_file)
        print(f"  Processing {len(df)} Symptom Diagnosis {split} examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  SymptomDx-{split}"):
            records.append({
                "instruction": "Based on the patient's symptoms, provide the most likely diagnosis.",
                "input": row["input_text"] if pd.notna(row["input_text"]) else "",
                "output": f"Based on the symptoms described, the likely diagnosis is: {row['output_text']}" if pd.notna(row["output_text"]) else "",
                "source": "symptom_diagnosis",
                "category": "diagnosis"
            })
    
    return records


def process_afrimedqa(raw_dir: Path) -> List[Dict]:
    """Process AfriMed-QA dataset from GitHub."""
    records = []
    
    # AfriMed-QA 15k dataset
    afrimedqa_file = raw_dir / "afrimedqa" / "afri_med_qa_15k_v2.5_phase_2_15275.csv"
    if afrimedqa_file.exists():
        df = pd.read_csv(afrimedqa_file)
        print(f"  Processing {len(df)} AfriMed-QA examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  AfriMedQA"):
            question = row["question_clean"] if pd.notna(row["question_clean"]) else row["question"]
            prompt = row["prompt"] if pd.notna(row["prompt"]) else ""
            answer_options = row["answer_options"] if pd.notna(row["answer_options"]) else ""
            correct_answer = row["correct_answer"] if pd.notna(row["correct_answer"]) else ""
            rationale = row["answer_rationale"] if pd.notna(row["answer_rationale"]) else ""
            specialty = row["specialty"] if pd.notna(row["specialty"]) else ""
            country = row["country"] if pd.notna(row["country"]) else ""
            q_type = row["question_type"] if pd.notna(row["question_type"]) else ""
            
            # Build instruction based on question type
            if q_type == "consumer_queries":
                instruction = f"You are a medical assistant helping patients in Africa. Answer the following health question."
            else:
                instruction = f"Answer this medical question. Specialty: {specialty}" if specialty else "Answer this medical question."
            
            # Build input
            input_text = f"{prompt}\n\n{question}".strip() if prompt else question
            if answer_options:
                input_text += f"\n\nOptions: {answer_options}"
            
            # Build output
            output = correct_answer if correct_answer else ""
            if rationale:
                output += f"\n\nExplanation: {rationale}"
            
            if output:  # Only add if we have an answer
                records.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "source": "afrimedqa",
                    "category": "african_medical_qa",
                    "metadata": {"country": country, "specialty": specialty}
                })
    
    # MedQA-USMLE train
    usmle_train = raw_dir / "afrimedqa" / "MedQA-USMLE-4-options-train.csv"
    if usmle_train.exists():
        df = pd.read_csv(usmle_train)
        print(f"  Processing {len(df)} MedQA-USMLE train examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  USMLE-train"):
            question = row["question"] if pd.notna(row["question"]) else ""
            options = row["options"] if pd.notna(row["options"]) else ""
            correct = row["correct_answer"] if pd.notna(row["correct_answer"]) else ""
            
            if question and correct:
                records.append({
                    "instruction": "Answer this USMLE-style medical question.",
                    "input": f"{question}\n\nOptions: {options}" if options else question,
                    "output": correct,
                    "source": "medqa_usmle",
                    "category": "medical_mcq"
                })
    
    # MedQA-USMLE test
    usmle_test = raw_dir / "afrimedqa" / "MedQA-USMLE-4-options-test.csv"
    if usmle_test.exists():
        df = pd.read_csv(usmle_test)
        print(f"  Processing {len(df)} MedQA-USMLE test examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  USMLE-test"):
            question = row["question"] if pd.notna(row["question"]) else ""
            options = row["options"] if pd.notna(row["options"]) else ""
            correct = row["correct_answer"] if pd.notna(row["correct_answer"]) else ""
            
            if question and correct:
                records.append({
                    "instruction": "Answer this USMLE-style medical question.",
                    "input": f"{question}\n\nOptions: {options}" if options else question,
                    "output": correct,
                    "source": "medqa_usmle",
                    "category": "medical_mcq"
                })
    
    return records


def process_wikidoc(raw_dir: Path) -> List[Dict]:
    """Process WikiDoc medical QA dataset."""
    records = []
    
    for key in ["wikidoc", "wikidoc_patient"]:
        parquet_file = raw_dir / key / "train.parquet"
        if not parquet_file.exists():
            continue
            
        df = pd.read_parquet(parquet_file)
        print(f"  Processing {len(df)} {key} examples...")
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {key}"):
            records.append({
                "instruction": row["instruction"] if pd.notna(row["instruction"]) else "",
                "input": row["input"] if pd.notna(row["input"]) else "",
                "output": row["output"] if pd.notna(row["output"]) else "",
                "source": key,
                "category": "medical_qa"
            })
    
    return records


def process_diseases_symptoms(raw_dir: Path) -> List[Dict]:
    """Process Diseases-Symptoms dataset."""
    records = []
    parquet_file = raw_dir / "diseases_symptoms" / "train.parquet"
    
    if not parquet_file.exists():
        print("  ⚠️ Diseases-Symptoms not found, skipping")
        return records
    
    df = pd.read_parquet(parquet_file)
    print(f"  Processing {len(df)} disease-symptom mappings...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  DiseaseSymptom"):
        name = row["Name"] if pd.notna(row["Name"]) else ""
        symptoms = row["Symptoms"] if pd.notna(row["Symptoms"]) else ""
        treatments = row["Treatments"] if pd.notna(row["Treatments"]) else ""
        
        # Create symptom-to-diagnosis example
        if symptoms and name:
            records.append({
                "instruction": "Based on the symptoms described, identify the likely condition and recommend treatment.",
                "input": f"Patient symptoms: {symptoms}",
                "output": f"Likely condition: {name}\n\nRecommended treatments: {treatments}",
                "source": "diseases_symptoms",
                "category": "diagnosis"
            })
    
    return records


def create_chw_specific_examples() -> List[Dict]:
    """Create CHW-specific instruction examples for African context."""
    chw_examples = [
        {
            "instruction": "You are a Community Health Worker assistant. Help triage this patient case.",
            "input": "A 2-year-old child has had diarrhea for 3 days, is refusing to drink, and appears lethargic. The family lives in a rural area with limited access to clean water.",
            "output": "This is a SEVERE case requiring URGENT referral:\n\n**Assessment:**\n- Duration >3 days with dehydration signs (refusing fluids, lethargy)\n- High risk of severe dehydration\n\n**Immediate Actions:**\n1. Begin ORS (Oral Rehydration Solution) immediately if the child can swallow\n2. Refer urgently to nearest health facility\n3. Keep the child warm during transport\n4. Continue breastfeeding if applicable\n\n**Danger Signs to Monitor:**\n- Sunken eyes\n- Skin pinch returns slowly\n- Unable to drink or vomiting everything\n\n**Prevention Counseling for Family:**\n- Boil drinking water or use water purification\n- Wash hands with soap before feeding child\n- Safe food storage practices",
            "source": "chw_synthetic",
            "category": "chw_triage"
        },
        {
            "instruction": "You are a Community Health Worker assistant. Provide guidance on malaria prevention.",
            "input": "A mother asks how to protect her family from malaria during the rainy season.",
            "output": "**Malaria Prevention for Rainy Season:**\n\n**1. Insecticide-Treated Nets (ITNs):**\n- Sleep under treated mosquito nets every night\n- Ensure nets have no holes\n- Tuck nets under mattress\n- All family members should use nets, especially children under 5 and pregnant women\n\n**2. Environmental Control:**\n- Remove stagnant water near home (pots, tires, containers)\n- Clear bushes around the house\n- Keep doors and windows closed at dusk\n\n**3. Personal Protection:**\n- Wear long sleeves and pants in the evening\n- Use mosquito repellent if available\n\n**4. Seek Care Early:**\n- If anyone develops fever, seek malaria testing within 24 hours\n- Complete full course of antimalarials if prescribed\n\n**5. Pregnant Women:**\n- Attend antenatal care for IPTp (preventive treatment)\n- Extra important to sleep under treated net",
            "source": "chw_synthetic",
            "category": "chw_prevention"
        },
        {
            "instruction": "You are a Community Health Worker assistant. Help assess this child for danger signs.",
            "input": "A mother brings a 6-month-old baby with fever and fast breathing. The baby is still breastfeeding.",
            "output": "**IMCI Assessment - Possible Pneumonia:**\n\n**Danger Signs to Check:**\n1. **Breathing rate:** Count breaths for 1 full minute\n   - For 2-12 months: Fast breathing = 50+ breaths/minute\n   - If chest indrawing present = SEVERE\n\n2. **General danger signs:**\n   - Unable to breastfeed? → DANGER\n   - Vomiting everything? → DANGER\n   - Convulsions? → DANGER\n   - Lethargic/unconscious? → DANGER\n\n**If ANY danger sign present:**\n- Give first dose of antibiotic if available\n- Refer URGENTLY to health facility\n- Keep baby warm\n- Continue breastfeeding during transport\n\n**If NO danger signs (just fast breathing + fever):**\n- Refer to health facility for assessment\n- Give paracetamol for fever\n- Continue breastfeeding frequently\n- Return immediately if condition worsens\n\n**Counsel Mother:**\n- Increase fluid intake\n- Watch for danger signs\n- Complete any treatment given",
            "source": "chw_synthetic",
            "category": "chw_triage"
        },
    ]
    return chw_examples


def combine_all_datasets():
    """Main function to combine all datasets."""
    print("\n" + "="*60)
    print("COMBINING DATASETS INTO MASTER FORMAT")
    print("="*60)
    
    all_records = []
    
    # Process each dataset
    processors = [
        ("AfriMed-QA", process_afrimedqa),
        ("ChatDoctor", process_chatdoctor),
        ("Medical Chatbot", process_medical_chatbot),
        ("MedInstruct", process_medinstruct),
        ("MedMCQA", process_medmcqa),
        ("Symptom Diagnosis", process_symptom_diagnosis),
        ("WikiDoc", process_wikidoc),
        ("Diseases-Symptoms", process_diseases_symptoms),
    ]
    
    for name, processor in processors:
        print(f"\n📦 {name}:")
        records = processor(RAW_DIR)
        print(f"  ✅ Added {len(records)} records")
        all_records.extend(records)
    
    # Add CHW-specific examples
    print(f"\n📦 CHW Synthetic Examples:")
    chw_records = create_chw_specific_examples()
    print(f"  ✅ Added {len(chw_records)} CHW-specific records")
    all_records.extend(chw_records)
    
    # Create DataFrame
    print(f"\n📊 Creating master dataset...")
    df = pd.DataFrame(all_records)
    
    # Remove empty outputs
    df = df[df["output"].str.len() > 0]
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=["input", "output"])
    dedup_count = len(df)
    print(f"  Removed {initial_count - dedup_count} duplicates")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save master dataset
    output_file = PROCESSED_DIR / "master_dataset.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\n💾 Saved master dataset: {output_file}")
    
    # Also save as JSONL for easier inspection
    jsonl_file = PROCESSED_DIR / "master_dataset.jsonl"
    df.to_json(jsonl_file, orient="records", lines=True)
    print(f"💾 Saved JSONL version: {jsonl_file}")
    
    # Statistics
    print("\n" + "="*60)
    print("MASTER DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal records: {len(df):,}")
    print(f"\nBy source:")
    for source, count in df["source"].value_counts().items():
        print(f"  - {source}: {count:,}")
    print(f"\nBy category:")
    for cat, count in df["category"].value_counts().items():
        print(f"  - {cat}: {count:,}")
    
    # Save statistics
    stats = {
        "total_records": len(df),
        "by_source": df["source"].value_counts().to_dict(),
        "by_category": df["category"].value_counts().to_dict(),
        "columns": list(df.columns),
    }
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Master dataset created successfully!")
    print(f"   Location: {PROCESSED_DIR}")
    
    return df


if __name__ == "__main__":
    combine_all_datasets()
