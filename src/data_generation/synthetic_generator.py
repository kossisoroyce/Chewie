"""
Synthetic Training Data Generator for AfriMed CHW Assistant

Generates instruction-tuning pairs for maternal health scenarios using:
1. Template-based generation from WHO guidelines
2. LLM-assisted generation with clinical validation prompts
3. Translation to Swahili for multilingual support
"""

import os
import json
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import click
from deep_translator import GoogleTranslator
import google.generativeai as genai
from tqdm import tqdm
import structlog

logger = structlog.get_logger()


@dataclass
class TrainingExample:
    """A single training example for fine-tuning."""
    messages: list[dict]
    category: str
    language: str
    source: str  # template, llm_generated, translated
    validated: bool = False


class SyntheticGenerator:
    """Generates synthetic training data for AfriMed."""
    
    SYSTEM_PROMPT = """You are AfriMed, a medical assistant helping Community Health Workers (CHWs) 
in East Africa provide maternal and newborn healthcare. 

Your role:
- Provide clear, actionable guidance for maternal health issues
- Identify danger signs that require immediate facility referral
- Support antenatal, delivery, postpartum, and newborn care decisions
- Communicate in simple, clear language

Safety rules:
- NEVER diagnose conditions - only support CHW decision-making
- ALWAYS recommend facility referral for danger signs
- When uncertain, advise consulting a supervisor or health facility"""

    # Scenario templates for different categories
    SCENARIO_TEMPLATES = {
        "antenatal_danger": [
            {
                "scenario": "pregnant woman with severe headache and blurred vision",
                "gestational_age": ["28 weeks", "32 weeks", "36 weeks", "38 weeks"],
                "symptoms": ["severe headache", "blurred vision", "seeing spots"],
                "danger_level": "high",
                "action": "immediate_referral",
                "condition_hint": "possible pre-eclampsia"
            },
            {
                "scenario": "pregnant woman with vaginal bleeding",
                "gestational_age": ["12 weeks", "20 weeks", "28 weeks", "36 weeks"],
                "symptoms": ["vaginal bleeding", "spotting", "blood clots"],
                "danger_level": "high",
                "action": "immediate_referral",
                "condition_hint": "possible placental problems"
            },
            {
                "scenario": "pregnant woman with reduced fetal movement",
                "gestational_age": ["28 weeks", "32 weeks", "36 weeks", "40 weeks"],
                "symptoms": ["baby not moving", "reduced kicks", "no movement today"],
                "danger_level": "high",
                "action": "immediate_referral",
                "condition_hint": "fetal distress"
            },
            {
                "scenario": "pregnant woman with fever",
                "gestational_age": ["any trimester"],
                "symptoms": ["high fever", "chills", "body aches", "too weak to stand"],
                "danger_level": "high",
                "action": "immediate_referral",
                "condition_hint": "possible infection"
            },
        ],
        "antenatal_routine": [
            {
                "scenario": "first antenatal visit",
                "questions": [
                    "What tests should be done at the first ANC visit?",
                    "What supplements should I give to a pregnant woman?",
                    "How do I calculate the due date?",
                ],
            },
            {
                "scenario": "nutrition counseling",
                "questions": [
                    "What foods should a pregnant woman eat?",
                    "How much weight should she gain?",
                    "Can she continue working?",
                ],
            },
            {
                "scenario": "birth preparedness",
                "questions": [
                    "What should be in a birth preparedness plan?",
                    "When should she go to the facility?",
                    "What are the signs of labor?",
                ],
            },
        ],
        "postnatal_danger": [
            {
                "scenario": "postpartum hemorrhage",
                "timing": ["immediately after birth", "1 day after", "1 week after"],
                "symptoms": ["heavy bleeding", "soaking more than one pad per hour", "feeling faint"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
            {
                "scenario": "postpartum infection",
                "timing": ["3 days after birth", "1 week after", "2 weeks after"],
                "symptoms": ["high fever", "foul-smelling discharge", "severe abdominal pain"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
            {
                "scenario": "postpartum pre-eclampsia",
                "timing": ["1 day after birth", "3 days after", "1 week after"],
                "symptoms": ["severe headache", "blurred vision", "swelling", "high blood pressure"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
        ],
        "newborn_danger": [
            {
                "scenario": "newborn not feeding",
                "age": ["1 day old", "3 days old", "1 week old"],
                "symptoms": ["refuses breast", "too weak to suck", "vomiting everything"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
            {
                "scenario": "newborn with fever",
                "age": ["1 day old", "5 days old", "2 weeks old"],
                "symptoms": ["hot to touch", "temperature above 37.5°C", "fever"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
            {
                "scenario": "newborn with fast breathing",
                "age": ["1 day old", "3 days old", "1 week old"],
                "symptoms": ["breathing fast", "chest moving in", "grunting"],
                "danger_level": "high",
                "action": "immediate_referral",
            },
            {
                "scenario": "newborn jaundice",
                "age": ["1 day old", "3 days old", "5 days old", "1 week old"],
                "symptoms": ["yellow skin", "yellow eyes", "very sleepy"],
                "danger_level": "varies",  # high if day 1, moderate if day 3+
                "action": "depends_on_timing",
            },
        ],
        "newborn_routine": [
            {
                "scenario": "breastfeeding support",
                "questions": [
                    "How often should the baby breastfeed?",
                    "How do I know the baby is getting enough milk?",
                    "What if the mother has cracked nipples?",
                ],
            },
            {
                "scenario": "cord care",
                "questions": [
                    "How do I care for the umbilical cord?",
                    "When does the cord fall off?",
                    "What are signs of cord infection?",
                ],
            },
            {
                "scenario": "thermal care",
                "questions": [
                    "How do I keep the baby warm?",
                    "What is kangaroo mother care?",
                    "How do I know if the baby is too cold?",
                ],
            },
        ],
    }
    
    def __init__(self, output_dir: str = "data/synthetic", use_llm: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm = use_llm
        self.translator = GoogleTranslator(source='en', target='sw')
        
        if use_llm:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            else:
                logger.warning("No API key found, LLM generation disabled")
                self.use_llm = False
    
    def generate_danger_sign_example(self, template: dict, category: str) -> TrainingExample:
        """Generate a training example for a danger sign scenario."""
        
        # Build the user query
        if category.startswith("antenatal"):
            gestational_age = random.choice(template.get("gestational_age", ["32 weeks"]))
            symptom = random.choice(template["symptoms"])
            query_templates = [
                f"A pregnant woman at {gestational_age} has {symptom}. What should I do?",
                f"I'm visiting a mother who is {gestational_age} pregnant. She says she has {symptom}. Is this serious?",
                f"Mama ana mimba ya {gestational_age}. Ana {symptom}. Nifanye nini?",  # Swahili variant
            ]
        elif category.startswith("postnatal"):
            timing = random.choice(template.get("timing", ["1 week after birth"]))
            symptom = random.choice(template["symptoms"])
            query_templates = [
                f"A mother who gave birth {timing} has {symptom}. What should I do?",
                f"I'm checking on a mother {timing} delivery. She has {symptom}. Is this normal?",
            ]
        else:  # newborn
            age = random.choice(template.get("age", ["3 days old"]))
            symptom = random.choice(template["symptoms"])
            query_templates = [
                f"A {age} baby has {symptom}. What should I do?",
                f"The newborn is {age} and the mother says the baby has {symptom}. Is this serious?",
            ]
        
        user_query = random.choice(query_templates)
        
        # Build the response
        response = self._generate_danger_response(template, category)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ]
        
        return TrainingExample(
            messages=messages,
            category=category,
            language="en",
            source="template",
            validated=True  # Template-based are pre-validated
        )
    
    def _generate_danger_response(self, template: dict, category: str) -> str:
        """Generate appropriate response for danger sign."""
        
        scenario = template["scenario"]
        condition_hint = template.get("condition_hint", "a serious condition")
        
        response = f"""⚠️ DANGER SIGN - REFER IMMEDIATELY

This is a serious situation that requires URGENT medical attention.

**What you're seeing:** {scenario}
**Why it's serious:** This could indicate {condition_hint}

**Actions to take NOW:**
1. 🚨 Arrange transport to the nearest health facility IMMEDIATELY
2. Stay calm and reassure the mother/family
3. Do NOT give any medications unless instructed by a health professional
4. Keep the mother comfortable during transport
5. Call ahead to the facility if possible to alert them

**While waiting for transport:**
- Keep the mother lying on her left side if pregnant
- Monitor her breathing and consciousness
- Keep her warm
- Do not give food or drinks in case surgery is needed

**Call your supervisor** to inform them of the emergency.

This is a medical emergency - do not delay referral."""
        
        return response
    
    def generate_routine_example(self, template: dict, category: str) -> TrainingExample:
        """Generate a training example for routine care scenarios."""
        
        question = random.choice(template["questions"])
        
        # Use LLM to generate response if available
        if self.use_llm:
            response = self._llm_generate_response(question, category)
        else:
            response = self._template_routine_response(question, category)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
        
        return TrainingExample(
            messages=messages,
            category=category,
            language="en",
            source="llm_generated" if self.use_llm else "template",
            validated=False  # LLM-generated need human validation
        )
    
    def _llm_generate_response(self, question: str, category: str) -> str:
        """Use Gemini to generate a response."""
        prompt = f"""You are a medical assistant helping train Community Health Workers in East Africa.

Generate a helpful, accurate response to this question from a CHW:
"{question}"

Context: This is about {category} care in a low-resource setting.

Requirements:
- Be clear and use simple language
- Provide actionable steps
- Include when to refer to a health facility
- Be culturally appropriate for East Africa
- Keep response under 300 words
- Use bullet points for clarity

Response:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error("LLM generation failed", error=str(e))
            return self._template_routine_response(question, category)
    
    def _template_routine_response(self, question: str, category: str) -> str:
        """Generate a template-based response for routine queries."""
        return f"""Thank you for your question about {category} care.

Here's what you should know:

**Key Points:**
- Follow your training guidelines for this situation
- Document your findings in the patient record
- Schedule appropriate follow-up visits

**When to refer:**
- If you notice any danger signs
- If you are unsure about the situation
- If the mother or baby is not improving

**Remember:**
- Always wash your hands before and after patient contact
- Counsel the family on warning signs to watch for
- Ensure the mother has a plan to reach the health facility if needed

Please consult your supervisor or the nearest health facility if you need additional guidance."""
    
    def translate_to_swahili(self, example: TrainingExample) -> TrainingExample:
        """Translate an English example to Swahili."""
        translated_messages = []
        
        for msg in example.messages:
            if msg["role"] == "system":
                # Keep system prompt in English or use pre-translated version
                translated_messages.append(msg)
            else:
                try:
                    translated_content = self.translator.translate(msg["content"])
                    translated_messages.append({
                        "role": msg["role"],
                        "content": translated_content
                    })
                except Exception as e:
                    logger.warning("Translation failed", error=str(e))
                    translated_messages.append(msg)
        
        return TrainingExample(
            messages=translated_messages,
            category=example.category,
            language="sw",
            source="translated",
            validated=False  # Translations need human validation
        )
    
    def generate_dataset(
        self,
        num_examples: int = 1000,
        include_swahili: bool = True,
        swahili_ratio: float = 0.3
    ) -> list[TrainingExample]:
        """Generate a complete training dataset."""
        
        examples = []
        
        # Calculate how many of each type to generate
        danger_examples = int(num_examples * 0.4)  # 40% danger signs (critical)
        routine_examples = num_examples - danger_examples
        
        logger.info(f"Generating {danger_examples} danger sign examples...")
        
        # Generate danger sign examples
        danger_categories = ["antenatal_danger", "postnatal_danger", "newborn_danger"]
        for _ in tqdm(range(danger_examples), desc="Danger signs"):
            category = random.choice(danger_categories)
            template = random.choice(self.SCENARIO_TEMPLATES[category])
            example = self.generate_danger_sign_example(template, category)
            examples.append(example)
        
        logger.info(f"Generating {routine_examples} routine care examples...")
        
        # Generate routine care examples
        routine_categories = ["antenatal_routine", "newborn_routine"]
        for _ in tqdm(range(routine_examples), desc="Routine care"):
            category = random.choice(routine_categories)
            template = random.choice(self.SCENARIO_TEMPLATES[category])
            example = self.generate_routine_example(template, category)
            examples.append(example)
        
        # Add Swahili translations
        if include_swahili:
            num_swahili = int(len(examples) * swahili_ratio)
            logger.info(f"Translating {num_swahili} examples to Swahili...")
            
            swahili_examples = []
            for example in tqdm(random.sample(examples, num_swahili), desc="Translating"):
                swahili_example = self.translate_to_swahili(example)
                swahili_examples.append(swahili_example)
            
            examples.extend(swahili_examples)
        
        random.shuffle(examples)
        return examples
    
    def save_dataset(
        self,
        examples: list[TrainingExample],
        filename: str = "training_data.jsonl",
        split_validation: bool = True,
        validation_ratio: float = 0.1
    ) -> tuple[Path, Optional[Path]]:
        """Save dataset to JSONL files."""
        
        if split_validation:
            split_idx = int(len(examples) * (1 - validation_ratio))
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            train_path = self.output_dir / filename
            val_path = self.output_dir / filename.replace(".jsonl", "_validation.jsonl")
            
            self._write_jsonl(train_examples, train_path)
            self._write_jsonl(val_examples, val_path)
            
            logger.info(f"Saved {len(train_examples)} training examples to {train_path}")
            logger.info(f"Saved {len(val_examples)} validation examples to {val_path}")
            
            return train_path, val_path
        else:
            output_path = self.output_dir / filename
            self._write_jsonl(examples, output_path)
            logger.info(f"Saved {len(examples)} examples to {output_path}")
            return output_path, None
    
    def _write_jsonl(self, examples: list[TrainingExample], path: Path):
        """Write examples to JSONL file."""
        with open(path, "w") as f:
            for example in examples:
                # Convert to the format expected by fine-tuning
                output = {
                    "messages": example.messages,
                    "metadata": {
                        "category": example.category,
                        "language": example.language,
                        "source": example.source,
                        "validated": example.validated
                    }
                }
                f.write(json.dumps(output) + "\n")


@click.command()
@click.option("--output", default="data/synthetic", help="Output directory")
@click.option("--num-examples", default=1000, help="Number of examples to generate")
@click.option("--include-swahili/--no-swahili", default=True, help="Include Swahili translations")
@click.option("--swahili-ratio", default=0.3, help="Ratio of Swahili examples")
@click.option("--use-llm/--no-llm", default=True, help="Use LLM for generation")
def main(output: str, num_examples: int, include_swahili: bool, swahili_ratio: float, use_llm: bool):
    """Generate synthetic training data for AfriMed."""
    
    generator = SyntheticGenerator(output_dir=output, use_llm=use_llm)
    
    print(f"🏥 AfriMed Synthetic Data Generator")
    print(f"   Output: {output}")
    print(f"   Examples: {num_examples}")
    print(f"   Swahili: {include_swahili} ({swahili_ratio*100:.0f}%)")
    print(f"   LLM: {use_llm}")
    print()
    
    examples = generator.generate_dataset(
        num_examples=num_examples,
        include_swahili=include_swahili,
        swahili_ratio=swahili_ratio
    )
    
    train_path, val_path = generator.save_dataset(examples)
    
    print(f"\n✅ Generation complete!")
    print(f"   Training data: {train_path}")
    if val_path:
        print(f"   Validation data: {val_path}")
    
    # Print statistics
    categories = {}
    languages = {"en": 0, "sw": 0}
    for ex in examples:
        categories[ex.category] = categories.get(ex.category, 0) + 1
        languages[ex.language] = languages.get(ex.language, 0) + 1
    
    print(f"\n📊 Statistics:")
    print(f"   By category:")
    for cat, count in sorted(categories.items()):
        print(f"      {cat}: {count}")
    print(f"   By language:")
    for lang, count in languages.items():
        print(f"      {lang}: {count}")


if __name__ == "__main__":
    main()
