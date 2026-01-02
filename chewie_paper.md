# The Story is Our Escort: Building Chewie for African Community Health Workers

*"Among the Igbo, proverbs are the palm oil with which words are eaten."* — Chinua Achebe

In many villages across sub-Saharan Africa, the distance between a sick child and the nearest hospital isn't just measured in kilometers. It's measured in uncertainty. When a mother reaches out to a Community Health Worker (CHW), she isn't looking for a Wikipedia entry. She needs to know, right now: *Is this an emergency? Do I need to find a way to the city?*

We built **Chewie** to answer those questions. Chewie is a 3B-parameter model designed to sit in the pocket of a CHW—bilingual, protocol-aligned, and grounded in the reality of African primary care.

## The Stakes of Silence

Community Health Workers are the backbone of our health systems, yet they often work in a vacuum. A doctor might be a three-day journey away. In that gap, silence is dangerous. If a CHW misses the signs of pre-eclampsia because the protocol wasn't clear, or if a chatbot suggests a home remedy for what is actually cerebral malaria, the cost isn't just a "bad user experience." It's a life.

Existing AI models are largely built for the Silicon Valley patient—someone with a high-speed connection and a pharmacy around the corner. They struggle with Swahili, they hallucinate non-existent treatments, and they are too expensive to run at scale in a rural clinic.

## Building the Path: Chewie Instruct

We realized that for a model to be useful here, it had to follow a strict path. We created **Chewie Instruct**, a dataset of ~3,100 clinical scenarios. We didn't just want the model to be "smart"; we wanted it to be disciplined. 

The dataset is a 50/50 bilingual split (English and Swahili), covering Maternal & Child Health (30%), Infectious Diseases (25%), and Emergency Triage. We didn't just translate words; we translated protocols. 

To train the model, we used a **Llama-3.2-3B-Instruct** base. We chose the 3B parameter size deliberately—it is the sweet spot for edge deployment on the hardware actually found in African clinics. We fine-tuned it using **LoRA (Low-Rank Adaptation)** for 2 epochs on an A100 GPU, applying **4-bit quantization (QLoRA)** to ensure the resulting model remains lightweight without losing its reasoning edge.

Every response follows a simple, grounded rhythm:
1. **Assessment:** What is the situation?
2. **Action:** What must be done immediately?
3. **Advice:** What should the patient know?

If a pregnant woman has a severe headache and blurred vision, Chewie doesn't hedge. It assesses it as a danger sign, actions an immediate referral, and advises on the gravity of the condition. 

## Insight: The Protocol is the Engine

The most important thing we learned is that a small model can be safer than a massive one if it is anchored by a protocol. 

The story is our escort. By grounding the model in specific clinical guidelines (WHO and local Ministry of Health protocols), we turned a general-purpose "toy" into a functional tool. We evaluated Chewie against a "Golden Reference" set of 25 critical cases and compared it to the **AfriMed-QA** benchmark.

The results validated the technical choices:
- **Protocol Adherence:** 95.8% (Consistently follows the Assess-Action-Advice structure).
- **Referral Accuracy:** 91.7% (Correctly flags high-risk danger signs for immediate medical attention).
- **Qualitative Reasoning:** In AfriMed-QA comparisons, Chewie prioritized reasoning through domestic clinical scenarios over simple option selection, behaving more like a practitioner than a test-taker.

It isn't a doctor, and it doesn't try to be. It's a guide. Like a neighbor who knows the way to the well, it points the direction to safety.

It isn't a doctor, and it doesn't try to be. It's a guide. Like a neighbor who knows the way to the well, it points the direction to safety.

## Implications: Intelligence at the Edge

So what follows? 

First, the cost of intelligence must fall. Cloud compute paid in dollars or naira is a tax on African innovation. By keeping Chewie small (3B parameters) and optimizing it for edge devices, we move the intelligence from the cloud to the clinic. 

Second, the future of AI in Africa must be local. We don't need "global" models that see our context as an edge case. We need models where our context is the core.

Chewie is just a beginning. The goal is to ensure that when a CHW reaches for their phone, they find a partner that understands their language, respects their protocol, and shares their burden.

---

### Resources
- **Model:** [electricsheepafrica/chewie-llama-3b](https://huggingface.co/electricsheepafrica/chewie-llama-3b)
- **Dataset:** [electricsheepafrica/chewie-instruct](https://huggingface.co/datasets/electricsheepafrica/chewie-instruct)
- **Code:** [kossisoroyce/Chewie](https://github.com/kossisoroyce/Chewie)
