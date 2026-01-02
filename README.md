# Chewie: African Community Health Worker Assistant 🌍🏥

**Chewie** is a specialized, open-source Large Language Model (LLM) designed to assist Community Health Workers (CHWs) in sub-Saharan Africa. Fine-tuned on the **Llama 3.2 3B** architecture, it is optimized for:

*   **Clinical Safety:** Strict adherence to triage protocols (Assessment -> Action -> Advice).
*   **Bilingual Support:** Fluent in **English** and **Swahili**.
*   **Edge Deployment:** Lightweight (3B params) for potential mobile use.

## 🚀 Features
- **Protocol Adherence:** Trained to follow WHO/MoH clinical guidelines.
- **Danger Sign Detection:** Prioritizes immediate referrals for critical conditions (e.g., Pre-eclampsia, Sepsis).
- **Empathy & Education:** Provides culturally aware advice and efficient patient counseling.

## 📥 Dataset
The model was fine-tuned on **Chewie Instruct**, a curated dataset of ~3,000+ medical instructions.
- **Hugging Face:** [electricsheepafrica/chewie-instruct](https://huggingface.co/datasets/electricsheepafrica/chewie-instruct)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kossisoroyce/Chewie.git
cd Chewie

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Usage

### Inference (Python)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "electricsheepafrica/chewie-llama-3b" # Coming soon

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = "A child has fever and stiff neck. What should I do?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

## 📊 Benchmarks
The model was evaluated against 25 clinical test cases and the **AfriMed-QA** benchmark.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Structure Adherence** | 95.8% | Consistently follows protocols |
| **Referral Accuracy** | 91.7% | Correctly identifies emergencies |
| **Swahili Fluency** | High | No degradation in clinical safety |

## 📜 License
This project is licensed under the **Apache 2.0 License**.

## 🤝 Contributing
Contributions are welcome! Please view the [Issues](https://github.com/kossisoroyce/Chewie/issues) tab.
