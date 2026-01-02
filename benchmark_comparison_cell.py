# Comparative Benchmarking with AfriMed-QA
# Run this cell to evaluate on an external African Medical Benchmark

import torch
from tqdm import tqdm
import pandas as pd

# 1. Hardcoded Sample of AfriMed-QA (20 Examples: 10 MCQ, 10 Consumer)
# Sourced from: afrimedqa/afrimedqa_v2 (Local CSV)
samples = [
  {
    "id": "AFMQ-MCQ-0",
    "type": "Multiple Choice",
    "prompt": "Which of the following is associated with the highest risk of transmission of HIV?\n\nOptions:\n{\"option1\": \"Receptive anal intercourse\", \"option2\": \"Insertive anal intercourse\", \"option3\": \"Receptive penile-vaginal intercourse\", \"option4\": \"Insertive penile-vaginal intercourse\", \"option5\": \"Oral sex\"}",
    "ref_answer": "option1"
  },
  {
    "id": "AFMQ-MCQ-1",
    "type": "Multiple Choice",
    "prompt": "A 16-year-old girl is brought to the physician by her mother because of a 1-year history of nonprogressive excessive hair growth. Physical examination shows Tanner stage 3 breast development and Tanner stage 4 pubic hair. There is hair on the face, chest, and lower abdomen. Pelvic examination shows no abnormalities. Serum studies show:\nFollicle-stimulating hormone:  4 mIU/mL\nLuteinizing hormone:  12 mIU/mL\n17-Hydroxyprogesterone:  110 ng/dL\nTestosterone:  90 ng/dL\nDHEA-S:  300 \u00b5g/dL\nWhich of the following is the most likely diagnosis?\n\nOptions:\n{\"option1\": \"Adrenal tumor\", \"option2\": \"Congenital adrenal hyperplasia\", \"option3\": \"Constitutional hirsutism\", \"option4\": \"Ovarian tumor\", \"option5\": \"Polycystic ovarian syndrome\"}",
    "ref_answer": "option5"
  },
  {
    "id": "AFMQ-MCQ-2",
    "type": "Multiple Choice",
    "prompt": "Which of the following is characteristic of kwashiorkor but not marasmus?\n\nOptions:\n{\"option1\": \"Edema\", \"option2\": \"Muscle wasting\", \"option3\": \"Loss of subcutaneous fat\", \"option4\": \"Growth retardation\", \"option5\": \"Anemia\"}",
    "ref_answer": "option1"
  },
  {
    "id": "AFMQ-MCQ-3",
    "type": "Multiple Choice",
    "prompt": "A 25-year-old woman comes to the office because of a 3-day history of fever, chills, severe headache, weakness, muscle pain, loss of appetite, vomiting, diarrhea, and moderate abdominal pain. She is in nursing school and returned from a medical missions trip in West Africa 10 days ago. Her symptoms began abruptly while she was shopping in a supermarket after her return. Temperature is 39.0\u00b0C (102.2\u00b0F), pulse is 100/min, respirations are 22/min, and blood pressure is 110/70 mm Hg. The patient appears ill and in mild respiratory distress. Physical examination discloses poor skin turgor and hyperactive bowel sounds. Muscle strength is 4/5 throughout. Laboratory studies show leukopenia and thrombocytopenia. Which of the following is the most sensitive and specific test for detection of the suspected viral genome in this patient?\n\nOptions:\n{\"option1\": \"Microarray analysis\", \"option2\": \"Northern blot\", \"option3\": \"Reverse transcription-polymerase chain reaction test\", \"option4\": \"Southern blot\", \"option5\": \"Western blot\"}",
    "ref_answer": "option3"
  },
  {
    "id": "AFMQ-MCQ-4",
    "type": "Multiple Choice",
    "prompt": "The following diseases must be screened for in blood for transfusion in Kenya;\n\nOptions:\n{\"option1\": \"Hepatitis\", \"option2\": \"HIV\", \"option3\": \"Syphillis\", \"option4\": \"All of the above\", \"option5\": \"N/A\"}",
    "ref_answer": "option4"
  },
  {
    "id": "AFMQ-CQ-0",
    "type": "Consumer Query",
    "prompt": "Your female coworker complains of lump in breast, changes in skin and thinks she has Breast Cancer and is going to visit the nearest doctor. Question: What are the signs of breast cancer?",
    "ref_answer": "Open-Ended"
  },
  {
    "id": "AFMQ-CQ-1",
    "type": "Consumer Query",
    "prompt": "Your female friend complains of bleeding, neck pain and thinks she has Crimean-Congo Hemorrhagic Fever and is going to visit the nearest doctor. Question: What do you mean by saying Crimean Congo Hemorrhagic fever?How does this fever affect my level of productivity?",
    "ref_answer": "Open-Ended"
  },
  {
    "id": "AFMQ-CQ-2",
    "type": "Consumer Query",
    "prompt": "Your female classmate complains of itchy, red eyes and thinks she has Conjunctivitis (Allergic) and is going to visit the nearest doctor. Question: Are there any specific triggers or allergens that may be causing my allergic conjunctivitis, and how can I best avoid or minimize exposure to them?",
    "ref_answer": "Open-Ended"
  },
  {
    "id": "AFMQ-CQ-3",
    "type": "Consumer Query",
    "prompt": "Your male coworker complains of running stomach, bloating and thinks he has Lactose Intolerance and is going to visit the nearest doctor. Question: What are some hidden sources of lactose that I should avoid?",
    "ref_answer": "Open-Ended"
  },
  {
    "id": "AFMQ-CQ-4",
    "type": "Consumer Query",
    "prompt": "Your female friend complains of vaginal bleeding, cramping and thinks she has Miscarriage and is going to visit the nearest doctor. Suggest an important question she might likely forget to ask. Question: What causes bleeding during pregnancy and how do I prevent its occurrence?",
    "ref_answer": "Open-Ended"
  }
]

# 2. Run Inference
if samples:
    results = []
    print(f"🧪 Running Inference on {len(samples)} hardcoded AfriMed-QA samples...")
    
    SYSTEM_PROMPT = "You are a Community Health Worker (CHW) assistant. Provide helpful, safe, and accurate medical advice."
    
    for case in tqdm(samples):
        # Construct formatting prompt
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{case['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                temperature=0.1
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("assistant\n\n")[-1].strip()
        
        results.append({
            "Type": case["type"],
            "Question": case["prompt"][:100] + "...",
            "Model Response": response[:150] + "...",
            "Reference": case["ref_answer"]
        })

    # 3. Display Comparison
    df_comp = pd.DataFrame(results)
    pd.set_option('display.max_colwidth', None)
    print("\n📊 AfriMed-QA Comparison (Sample):")
    display(df_comp)
else:
    print("⚠️ No samples to run.")
