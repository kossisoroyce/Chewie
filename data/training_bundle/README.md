---
language:
- en
- sw
license: cc-by-nc-4.0
task_categories:
- text-generation
- question-answering
tags:
- medical
- healthcare
- africa
- chw
- clinical
pretty_name: Chewie Instruct
size_categories:
- 1K<n<10K
---

# Chewie Instruct

![Chewie Banner](chewie_banner.png)

**Chewie Instruct** is a bilingual (English-Swahili) medical instruction dataset designed to train large language models (LLMs) to act as assistants for **Community Health Workers (CHWs)** in Africa.

## Dataset Summary

- **Total Samples:** ~3,000+ High-Quality Instructions
- **Languages:** English, Swahili
- **Focus:** Primary Healthcare, Triage, Maternal & Child Health, Infectious Diseases (Malaria, HIV/TB), NCDs, and Emergency Danger Signs.
- **Source:** Synthetic generation based on WHO/Ministry of Health Clinical Guidelines, curated by AI and reviewed for protocol adherence.

## Purpose

The primary goal of this dataset is to fine-tune models to:
1.  **Adhere to CHW Protocols:** Assessment -> Action -> Advice.
2.  **Detect Danger Signs:** Prioritize immediate referral for critical conditions.
3.  **Bridge the Language Gap:** Provide accurate medical guidance in Swahili.
4.  **Display Empathy:** Support CHWs in dealing with patients facing stigma or difficult diagnoses.

## Dataset Structure

Each example contains:
- `instruction`: A scenario or question posed by a CHW or community member.
- `output`: A structured, safe, and actionable response following established health protocols.

### Example (English)
```json
{
  "instruction": "A pregnant woman in her third trimester has severe headache, blurred vision, and swollen feet. What should I do?",
  "output": "**Assessment:** These are classic danger signs of Pre-eclampsia.\n**Action:** Refer to the health facility IMMEDIATELY.\n**Advice:** Do not delay. This condition can trigger seizures (Eclampsia) which is fatal."
}
```

### Example (Swahili)
```json
{
  "instruction": "Mama mjamzito anaumwa kichwa sana na anaona giza. Nifanye nini?",
  "output": "**Tathmini:** Hizi ni dalili za hatari za Kifafa cha Mimba (Pre-eclampsia).\n**Hatua:** Mpeleke hospitali mara moja.\n**Ushauri:** Hali hii ni ya dharura na inahitaji matibabu ya haraka."
}
```

## Methodology

The dataset was curated using a "Clinical Logic" pipeline:
1.  **Guideline Extraction:** Key protocols from WHO and African Ministries of Health were identified.
2.  **Scenario Generation:** Synthetic patient scenarios were generated to cover diverse demographics and conditions.
3.  **Translation:** Professional-grade translation into Swahili with medical terminology verification.
4.  **Filtering:** Rigorous removal of "hallucinated" treatments or non-standard advice.

## Usage

This dataset is ideal for sft (supervised fine-tuning) of base models (like Llama-3, Mistral, Gemma) to create specialized healthcare assistants for resource-limited settings.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
Intended for research and humanitarian use.
