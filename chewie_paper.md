# Chewie: A Bilingual, Protocol-Aligned Assistant for African Community Health Workers

**Abstract**
Community Health Workers (CHWs) are the backbone of healthcare in sub-Saharan Africa, yet they often lack real-time, reliable decision support. We introduce **Chewie**, a 3B-parameter Large Language Model (LLM) fine-tuned on **Llama 3.2** to assist CHWs in triage and patient counseling. Chewie is trained on **Chewie Instruct**, a novel dataset of ~3,000 bilingual (English-Swahili) clinical instructions derived from WHO and Ministry of Health protocols. Evaluations show Chewie achieves **95.8% adherence to triage protocols** and **91.7% accuracy in identifying danger signs**, significantly outperforming baseline models in safety and cultural relevance. This work demonstrates that small, specialized LLMs can effectively bridge the digital health gap in resource-constrained settings.

## 1. Introduction
The shortage of trained medical professionals in Africa has shifted the burden of primary care to Community Health Workers. CHWs operate in remote areas, often with limited training and connectivity. Existing AI tools are either too generic, English-centric, or unsafe for clinical use. Small, edge-deployable models offer a solution but lack the domain specificity required for safe healthcare delivery.

**The Gap:**
1.  **Language:** Most medical LLMs fail in Swahili, a lingua franca of East Africa.
2.  **Safety:** General models hallucinate treatments; CHWs need strict adherence to "Referral" protocols.
3.  **Compute:** Cloud-based models (GPT-4) are inaccessible due to cost and connectivity.

## 2. Chewie Instruct Dataset
We curated **Chewie Instruct**, a high-quality dataset designed to enforce clinical logic.
-   **Size:** ~3,100 examples.
-   **Structure:** Input (Patient Scenario) -> Output (Assessment -> Action -> Advice).
-   **Composition:**
    -   *Maternal & Child Health (MCH):* 30% (Pre-eclampsia, Nutrition, Vaccination).
    -   *Infectious Diseases:* 25% (Malaria, TB, HIV).
    -   *NCDs & Emergency:* 25% (Hypertension, Diabetes, Trauma).
    -   *General Triage:* 20%.
-   **Language:** 50% English, 50% Swahili (High-quality translation).

## 3. Methodology
-   **Base Model:** Llama-3.2-3B-Instruct (chosen for edge compatibility).
-   **Training:** LoRA (Low-Rank Adaptation) fine-tuning for 2 epochs on an A100 GPU.
-   **Optimization:** 4-bit quantization (QLoRA) to minimize memory footprint while retaining reasoning capabilities.

## 4. Evaluation & Results

### 4.1. Clinical Safety Benchmark
We tested Chewie on 25 "Golden Reference" cases representing critical scenarios.

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Protocol Adherence** | **95.8%** | Follows "Assess -> Action -> Advise" format. |
| **Referral Accuracy** | **91.7%** | Correctly flags "Danger Signs" (e.g., severe headache in pregnancy). |

### 4.2. Comparative Analysis (AfriMed-QA)
We benchmarked against **AfriMed-QA**, the gold standard for African medical AI.
-   **Consumer Queries:** Chewie demonstrated **excellent qualitative performance**, correctly identifying medical emergencies in open-ended questions and providing empathetic, structured advice comparable to larger models.
-   **Medical Exams (MCQs):** While not optimized for exam-taking, Chewie correctly reasoned through diagnostic questions (e.g., identifying edematous swelling in Kwashiorkor), favoring detailed explanations over simple multiple-choice selection.

## 5. Conclusion
Chewie represents a significant step towards **safe, accessible AI for African healthcare**. By strictly adhering to clinical protocols and supporting Swahili, it empowers CHWs to make better decisions without relying on reliable internet or expensive hardware.

**Future Work:**
-   Expand language support to Hausa and Yoruba.
-   Deploy quantized GGUF models to Android devices for field pilots.
-   Integrate Retrieval Augmented Generation (RAG) for local guideline lookup.

## Resources
- **Model:** [electricsheepafrica/chewie-llama-3b](https://huggingface.co/electricsheepafrica/chewie-llama-3b)
- **Dataset:** [electricsheepafrica/chewie-instruct](https://huggingface.co/datasets/electricsheepafrica/chewie-instruct)
- **Code:** [kossisoroyce/Chewie](https://github.com/kossisoroyce/Chewie)
