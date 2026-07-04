# Author: Mohan Raju Amuri
"""
Named Entity Recognition (NER)
--------------------------------
One-liner: Identify and classify real-world objects (people, places, orgs, dates) in text.

spaCy entity labels:
  PERSON    → people (Steve Jobs)
  ORG       → companies, institutions (Apple, MIT)
  GPE       → countries, cities (India, New York)
  LOC       → mountains, rivers (Himalayas)
  DATE      → dates, periods (January 2024, last year)
  MONEY     → monetary values ($10 million)
  PRODUCT   → products (iPhone)
  EVENT     → named events (Olympics)

Remember:
- spaCy NER works out of the box — no training needed for common entities
- doc.ents gives all entities; ent.label_, ent.start_char, ent.end_char
- For custom entities (product names, medical terms) → train custom NER
- en_core_web_sm has fewer entity types than en_core_web_lg

Don't:
- Don't expect NER to catch domain-specific entities without fine-tuning
  (medical drugs, legal terms, internal product codes won't be recognized)
- Don't rely on NER for privacy scrubbing alone — it misses things
- Don't use sm model for high-accuracy production NER — use lg or a fine-tuned model
"""

import spacy

nlp = spacy.load("en_core_web_sm")

# --- Basic NER ---
texts = [
    "Elon Musk founded SpaceX in 2002 and Tesla in 2003 in California.",
    "Amazon acquired Whole Foods for $13.7 billion in 2017.",
    "The 2024 Olympics were held in Paris, France.",
    "Dr. Sarah Johnson from MIT published a paper on GPT-4 last January.",
]

print("=== Named Entity Recognition ===\n")
for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"  [{ent.label_:<10}] {ent.text}")
    print()

# --- Extract specific entity types ---
print("=== Extract only ORG and PERSON ===")
news = """
Google CEO Sundar Pichai met with Microsoft founder Bill Gates at the
World Economic Forum in Davos. Both Apple and Meta also sent representatives.
"""
doc = nlp(news)
people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
orgs   = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
print(f"People : {people}")
print(f"Orgs   : {orgs}")

# --- Entity with char positions (useful for highlighting) ---
print("\n=== Entity positions (for highlighting) ===")
text = "Apple launched iPhone 15 in September 2023 at their Cupertino campus."
doc = nlp(text)
for ent in doc.ents:
    print(f"  {ent.text:<20} label={ent.label_:<10} chars=[{ent.start_char}:{ent.end_char}]")

# --- Count entity types in a document ---
from collections import Counter
long_text = " ".join(texts)
doc = nlp(long_text)
entity_counts = Counter(ent.label_ for ent in doc.ents)
print("\n=== Entity type distribution ===")
for label, count in entity_counts.most_common():
    print(f"  {label:<12} {count}  ({spacy.explain(label)})")
