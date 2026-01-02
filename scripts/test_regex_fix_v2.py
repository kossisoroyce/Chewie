
import re

# Refined Pattern:
# \bDr              -> Start with word boundary and "Dr"
# (?:[\.\s])        -> MUST follow with a dot or whitespace (fixes "drops", "drug")
# \s*               -> Optional extra whitespace
# (?:...)+          -> One or more name parts
# Name parts: [A-Z][a-z]+ (Word) OR [A-Z]\.? (Initial with optional dot)
# Separated by spaces or dots
NEW_PATTERN = r'\bDr(?:[\.\s])\s*(?:[A-Z](?:[a-z]+|\.)(?:\s+[A-Z](?:[a-z]+|\.)){0,4})(?:\s*[,;]\s*[^,;\n]{0,60})?'

REPLACEMENT = "[DOCTOR]"

test_cases = [
    # Should NOT match
    "Monitor the child's hydration levels.",
    "Signs of dehydration include dry mouth.",
    "Ensure the patient is well-hydrated.",
    "Take 5ml of the drops.",
    "The drug is effective.",
    "Syndrome X is rare.",
    "Children are susceptible.",
    "My dream is to fly.",
    "Address the issue.",
    
    # Should MATCH
    "Contact Dr. Smith immediately.",
    "Dr. John Doe will see you now.",
    "Thanks, Dr. A. B. Cde.",
    "Regards, Dr. Jane Doe, Cardiologist",
    "Dr. House",
    "dr. strange (lowercase start)", 
    "Dr A. Smith",
    "Dr.B.Jones",
    "I am Dr. Who.",
    "Hello Dr. Phil,"
]

print(f"Testing pattern: {NEW_PATTERN}\n")

for text in test_cases:
    cleaned = re.sub(NEW_PATTERN, REPLACEMENT, text, flags=re.IGNORECASE)
    
    # logic for success/fail
    is_bad = False
    
    # Should NOT match cases
    if "hydration" in text and "[DOCTOR]" in cleaned: is_bad = True
    if "drops" in text and "[DOCTOR]" in cleaned: is_bad = True
    if "drug" in text and "[DOCTOR]" in cleaned: is_bad = True
    if "Address" in text and "[DOCTOR]" in cleaned: is_bad = True
    
    # Should MATCH cases
    if "Dr." in text or "dr." in text:
        if "[DOCTOR]" not in cleaned: is_bad = True
        
    status = "❌" if is_bad else "✅"
    
    print(f"{status} Original: {text}")
    print(f"   Cleaned:  {cleaned}")
