
import re

# The FIXED pattern
DOCTOR_PATTERN = r'\bDr\.?\s*[A-Z][a-z]+(?:[.\s]+[A-Z]\.?\s*)?(?:\s*[A-Z][a-z]+)?(?:\s*[,;]\s*[^,;\n]{0,60})?'
REPLACEMENT = "[DOCTOR]"

test_cases = [
    "Monitor the child's hydration levels.",
    "Signs of dehydration include dry mouth.",
    "Ensure the patient is well-hydrated.",
    "Contact Dr. Smith immediately.",
    "Dr. John Doe will see you now.",
    "Thanks, Dr. A. B. Cde.",
    "Regards, Dr. Jane Doe, Cardiologist",
    "Take 5ml of the drops.",
    "The drug is effective.",
    "Syndrome X is rare.",
    "Children are susceptible."
]

print(f"Testing pattern: {DOCTOR_PATTERN}\n")

for text in test_cases:
    cleaned = re.sub(DOCTOR_PATTERN, REPLACEMENT, text, flags=re.IGNORECASE)
    status = "✅" if cleaned == text and "Dr." not in text else ("✅" if "[DOCTOR]" in cleaned and "Dr." in text else "❌")
    
    # Special check for hydration/drops/etc preservation
    if "hydration" in text and "hydration" not in cleaned: status = "❌ BROKEN"
    if "drops" in text and "drops" not in cleaned: status = "❌ BROKEN"
    
    print(f"{status} Original: {text}")
    print(f"   Cleaned:  {cleaned}")
