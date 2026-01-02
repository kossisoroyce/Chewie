from datasets import load_dataset
import pandas as pd
import json

def sample_afrimed_qa(output_path="data/benchmark/afrimed_qa_sample.jsonl", num_samples=50):
    print("⏳ Loading AfriMed-QA dataset from Hugging Face...")
    try:
        # Load specific subset if possible, or main dataset
        dataset = load_dataset("afrimedqa/afrimedqa_v2", split="train", streaming=True)
        
        # We prefer "Consumer Queries" or similar open-ended questions if labeled, 
        # but for now we'll take a general sample and filter for relevance if needed.
        # Based on research, it has multiple subsets. We'll inspect the first few.
        
        samples = []
        count = 0
        
        for example in dataset:
            # Structure check: looking for medical questions
            # Example fields might vary, trying to adapt generic
            question = example.get('question') or example.get('instruction') or example.get('input')
            answer = example.get('answer') or example.get('output')
            
            if question and answer:
                samples.append({
                    "id": f"AFMQ-{count}",
                    "source": "AfriMed-QA",
                    "prompt": question,
                    "expected_answer": answer,
                    "category": "general_medical" 
                })
                count += 1
                
            if count >= num_samples:
                break
                
        # Save to JSONL
        with open(output_path, 'w') as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
                
        print(f"✅ Saved {len(samples)} samples to {output_path}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

if __name__ == "__main__":
    sample_afrimed_qa()
