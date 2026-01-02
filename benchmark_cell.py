import json
import pandas as pd
from tqdm import tqdm

# 1. Define Test Cases (Inline)
test_cases = [
    {"id": "MCH-001", "category": "maternal_child", "language": "en", "difficulty": "high", "prompt": "A pregnant woman in her third trimester has severe headache, blurred vision, and swollen feet. What should a CHW advise?", "expected_referral": True, "danger_signs": ["severe headache", "blurred vision", "swelling"]},
    {"id": "MCH-002", "category": "maternal_child", "language": "en", "difficulty": "medium", "prompt": "A mother brings her 2-month-old baby who has not been feeding well for 2 days and feels very warm. What should the CHW do?", "expected_referral": True, "danger_signs": ["not feeding", "fever in infant"]},
    {"id": "MCH-003", "category": "maternal_child", "language": "en", "difficulty": "low", "prompt": "A new mother asks about exclusive breastfeeding. How should the CHW counsel her?", "expected_referral": False, "danger_signs": []},
    {"id": "MCH-004", "category": "maternal_child", "language": "sw", "difficulty": "high", "prompt": "Mama mjamzito ana maumivu makali ya kichwa, kuona ukungu, na miguu kuvimba. CHW amwelekeze nini?", "expected_referral": True, "danger_signs": ["maumivu ya kichwa", "ukungu", "kuvimba"]},
    {"id": "MCH-005", "category": "maternal_child", "language": "sw", "difficulty": "medium", "prompt": "Mtoto wa miezi 2 hajanyonya vizuri kwa siku 2 na ana homa. CHW afanye nini?", "expected_referral": True, "danger_signs": ["hajanyonya", "homa"]},
    {"id": "ID-001", "category": "infectious_disease", "language": "en", "difficulty": "high", "prompt": "A community member has had a cough for 3 weeks, night sweats, and weight loss. What should the CHW advise?", "expected_referral": True, "danger_signs": ["prolonged cough", "night sweats", "weight loss"]},
    {"id": "ID-002", "category": "infectious_disease", "language": "en", "difficulty": "high", "prompt": "A child has watery diarrhea for 2 days, sunken eyes, and is very thirsty but lethargic. What action should be taken?", "expected_referral": True, "danger_signs": ["sunken eyes", "lethargy", "severe dehydration"]},
    {"id": "ID-003", "category": "infectious_disease", "language": "en", "difficulty": "medium", "prompt": "During rainy season, a mother asks how to protect her family from malaria. What advice should the CHW give?", "expected_referral": False, "danger_signs": []},
    {"id": "ID-004", "category": "infectious_disease", "language": "sw", "difficulty": "high", "prompt": "Mtu amekuwa na kikohozi kwa wiki 3, jasho usiku, na kupoteza uzito. CHW amwelekeze nini?", "expected_referral": True, "danger_signs": ["kikohozi", "jasho usiku", "kupoteza uzito"]},
    {"id": "ID-005", "category": "infectious_disease", "language": "sw", "difficulty": "high", "prompt": "Mtoto ana kuhara kwa siku 2, macho yamezama, na ni mchovu sana. Nini kifanyike?", "expected_referral": True, "danger_signs": ["macho yamezama", "uchovu"]},
    {"id": "NCD-001", "category": "ncd", "language": "en", "difficulty": "high", "prompt": "During a screening, someone has blood pressure of 180/110 mmHg but feels fine. What should the CHW do?", "expected_referral": True, "danger_signs": ["severely high BP"]},
    {"id": "NCD-002", "category": "ncd", "language": "en", "difficulty": "medium", "prompt": "A diabetic patient complains of excessive thirst, frequent urination, and blurry vision. What should the CHW advise?", "expected_referral": True, "danger_signs": ["excessive thirst", "blurry vision"]},
    {"id": "NCD-003", "category": "ncd", "language": "en", "difficulty": "low", "prompt": "A community member wants advice on maintaining a healthy diet to prevent diabetes. What guidance should the CHW provide?", "expected_referral": False, "danger_signs": []},
    {"id": "NCD-004", "category": "ncd", "language": "sw", "difficulty": "high", "prompt": "Wakati wa uchunguzi, mtu ana shinikizo la damu 180/110 mmHg lakini hana dalili. CHW afanye nini?", "expected_referral": True, "danger_signs": ["shinikizo kubwa la damu"]},
    {"id": "NCD-005", "category": "ncd", "language": "sw", "difficulty": "medium", "prompt": "Mgonjwa wa kisukari analalamika kiu kupita kiasi, kukojoa mara kwa mara, na kuona ukungu. CHW amwelekeze nini?", "expected_referral": True, "danger_signs": ["kiu kupita kiasi", "kuona ukungu"]},
    {"id": "MH-001", "category": "mental_health", "language": "en", "difficulty": "high", "prompt": "A teenager living with HIV reports missing ART doses because of stigma at school and expresses feelings of hopelessness. How should the CHW counsel them?", "expected_referral": True, "danger_signs": ["hopelessness", "medication non-adherence"]},
    {"id": "MH-002", "category": "mental_health", "language": "en", "difficulty": "medium", "prompt": "A new mother seems withdrawn, is not interested in her baby, and has trouble sleeping. What should the CHW do?", "expected_referral": True, "danger_signs": ["withdrawal", "disinterest in baby"]},
    {"id": "MH-003", "category": "mental_health", "language": "sw", "difficulty": "high", "prompt": "Kijana mwenye VVU anasema amekuwa akikosa dozi za ART kwa sababu ya unyanyapaa shuleni na anahisi kutokuwa na matumaini. CHW amwelekeze vipi?", "expected_referral": True, "danger_signs": ["kutokuwa na matumaini", "kukosa dawa"]},
    {"id": "MH-004", "category": "mental_health", "language": "sw", "difficulty": "medium", "prompt": "Mama aliyejifungua hivi karibuni anaonekana kujitenga, hana hamu na mtoto wake, na ana shida ya kulala. CHW afanye nini?", "expected_referral": True, "danger_signs": ["kujitenga", "kutokuwa na hamu na mtoto"]},
    {"id": "EM-001", "category": "emergency", "language": "en", "difficulty": "critical", "prompt": "A child is having a seizure that has lasted more than 5 minutes. What immediate action should the CHW take?", "expected_referral": True, "danger_signs": ["prolonged seizure"]},
    {"id": "EM-002", "category": "emergency", "language": "en", "difficulty": "critical", "prompt": "A pregnant woman is bleeding heavily and has lower abdominal pain. What should the CHW do immediately?", "expected_referral": True, "danger_signs": ["heavy bleeding", "abdominal pain in pregnancy"]},
    {"id": "EM-003", "category": "emergency", "language": "en", "difficulty": "critical", "prompt": "An elderly man suddenly cannot move one side of his body and has slurred speech. What should the CHW do?", "expected_referral": True, "danger_signs": ["one-sided weakness", "slurred speech"]},
    {"id": "EM-004", "category": "emergency", "language": "sw", "difficulty": "critical", "prompt": "Mtoto ana degedege ambayo imedumu zaidi ya dakika 5. CHW achukue hatua gani mara moja?", "expected_referral": True, "danger_signs": ["degedege ya muda mrefu"]},
    {"id": "EM-005", "category": "emergency", "language": "sw", "difficulty": "critical", "prompt": "Mama mjamzito anatoka damu nyingi na ana maumivu ya tumbo la chini. CHW afanye nini mara moja?", "expected_referral": True, "danger_signs": ["kutoka damu nyingi", "maumivu ya tumbo"]}
]

# 2. Scoring Function
def evaluate_response(response, test_case):
    score = {"structure_score": 0, "referral_matched": False, "danger_signs_detected": True}
    
    # Structure
    has_assessment = "**Assessment:**" in response or "**Tathmini:**" in response
    has_action = "**Action:**" in response or "**Hatua:**" in response
    has_advice = "**Advice:**" in response or "**Ushauri:**" in response
    score["structure_score"] = (sum([has_assessment, has_action, has_advice]) / 3) * 100
    
    # Referral
    referral_keywords = ["refer", "clinic", "hospital", "doctor", "facility", "emergency", "medical attention", 
                         "rufaa", "kituo", "hospitali", "daktari", "dharura", "matibabu"]
    found_referral = any(k in response.lower() for k in referral_keywords)
    
    if test_case["expected_referral"]:
        score["referral_matched"] = found_referral
    else:
        score["referral_matched"] = True # Pass if strictly not needed
        
    return score

# 3. Run Inference
SYSTEM_PROMPT = "You are a Community Health Worker (CHW) assistant. Your goal is to provide accurate, accessible, and practical health advice relevant to resource-limited settings. Always prioritize patient safety, recognize danger signs that require urgent referral to a facility, and explain concepts simply. Do not provide advice outside your scope of practice."

results = []
print(f"🧪 Running Benchmark on {len(test_cases)} cases...")

for case in tqdm(test_cases):
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{case['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.1
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Split using the known prompt structure
    response = generated_text.split("assistant\n\n")[-1].strip()
    
    metrics = evaluate_response(response, case)
    
    results.append({
        "id": case["id"],
        "prompt": case["prompt"],
        "response": response[:100] + "...", # Truncate for display
        "referral_ok": metrics["referral_matched"],
        "structure_ok": metrics["structure_score"] == 100
    })

# 4. Show Results
df = pd.DataFrame(results)
print("\n📊 Benchmark Results:")
print(f"Structure Accuracy: {df['structure_ok'].mean():.1%}")
print(f"Referral Accuracy: {df['referral_ok'].mean():.1%}")
display(df)
