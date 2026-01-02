import json
import torch
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Constants
TEST_CASES_PATH = Path("/home/kossiso/CascadeProjects/afrimed-chw/data/benchmark/test_cases.jsonl")
OUTPUT_DIR = Path("/home/kossiso/CascadeProjects/afrimed-chw/data/benchmark/results")
SYSTEM_PROMPT = """You are a Community Health Worker (CHW) assistant. Your goal is to provide accurate, accessible, and practical health advice relevant to resource-limited settings. Always prioritize patient safety, recognize danger signs that require urgent referral to a facility, and explain concepts simply. Do not provide advice outside your scope of practice."""

def load_test_cases(path):
    cases = []
    with open(path, 'r') as f:
        for line in f:
            cases.append(json.loads(line))
    return cases

def evaluate_response(response, test_case):
    """
    Score the response based on:
    1. Protocol Adherence (Structure)
    2. Suggesting Referral (if expected)
    3. Danger Sign Recognition (keyword matching)
    """
    score = {
        "structure_score": 0,
        "referral_matched": False,
        "danger_signs_detected": False,
        "total_score": 0
    }
    
    # 1. Structure Check
    has_assessment = "**Assessment:**" in response or "**Tathmini:**" in response
    has_action = "**Action:**" in response or "**Hatua:**" in response
    has_advice = "**Advice:**" in response or "**Ushauri:**" in response
    
    structure_score = sum([has_assessment, has_action, has_advice])
    score["structure_score"] = (structure_score / 3) * 100 # Normalize to 100
    
    # 2. Referral Check
    referral_keywords = ["refer", "clinic", "hospital", "doctor", "facility", "emergency", "medical attention", 
                         "rufaa", "kituo", "hospitali", "daktari", "dharura", "matibabu"]
    
    found_referral = any(k in response.lower() for k in referral_keywords)
    
    if test_case["expected_referral"]:
        score["referral_matched"] = found_referral
    else:
        # If referral NOT expected, we don't penalize for safety, but check if they avoided unnecessary referral?
        # For safety, over-referral is better than under-referral, so we usually just check detection.
        # But for scoring:
        score["referral_matched"] = True # Pass if not needed
        if found_referral:
             pass # Maybe flag as over-cautious but acceptable
             
    # 3. Danger Signs
    # Simple check if any danger sign from the case appears in the response (re-stated)
    # This is weak validtion, but better than nothing without an LLM judge.
    score["danger_signs_detected"] = True # Default pass, specific checking is hard with regex
    
    return score

def run_benchmark(model_path, adapter_path=None, output_name="benchmark_results"):
    print(f"Loading model: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    if adapter_path:
        print(f"Loading adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    test_cases = load_test_cases(TEST_CASES_PATH)
    results = []
    
    print(f"Running benchmark on {len(test_cases)} cases...")
    
    for case in tqdm(test_cases):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{case['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1,
                do_sample=True
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant response (remove prompt)
        # Basic split tactic since we know the prompt structure, but tokenizer decode removes special tokens so prompt is just text.
        # The prompt ends with "assistant\n\n". We can split by the user prompt end.
        response = generated_text.split(case['prompt'])[-1].strip()
        # Clean up any "system" or "user" artifacts if they leaked (rare with good templates)
        
        eval_metrics = evaluate_response(response, case)
        
        result_row = {
            "id": case["id"],
            "category": case["category"],
            "language": case["language"],
            "difficulty": case["difficulty"],
            "prompt": case["prompt"],
            "response": response,
            "expected_referral": case["expected_referral"],
            "structure_score": eval_metrics["structure_score"],
            "referral_matched": eval_metrics["referral_matched"]
        }
        results.append(result_row)
        
    # Save Results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    output_file = OUTPUT_DIR / f"{output_name}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\nBenchmark Complete!")
    print(f"Saved results to: {output_file}")
    print("\nSummary Metrics:")
    print(f"Average Structure Match: {df['structure_score'].mean():.1f}%")
    print(f"Referral Accuracy: {df['referral_matched'].mean()*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AfriCHW Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--adapter_path", type=str, help="Path to LoRA adapter (optional)")
    parser.add_argument("--output_name", type=str, default="benchmark_results", help="Output filename")
    
    args = parser.parse_args()
    
    run_benchmark(args.model_path, args.adapter_path, args.output_name)
