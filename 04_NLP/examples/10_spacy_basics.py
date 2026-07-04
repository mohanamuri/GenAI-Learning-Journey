"""
spaCy Basics
------------
One-liner: Industrial-strength NLP — tokenization, POS, NER, dependency parsing in one pipeline.

Remember:
- Load model once, reuse: nlp = spacy.load("en_core_web_sm")
- doc = nlp(text) runs the FULL pipeline (tokenize → POS → NER → dep parse)
- Disable unused pipes for speed: nlp(text, disable=["ner", "parser"])
- Models: en_core_web_sm (fast), en_core_web_md (+ vectors), en_core_web_lg (best)
- token.lemma_ → base form | token.pos_ → universal POS | token.ent_type_ → entity

Setup:
  pip install spacy
  python -m spacy download en_core_web_sm

Don't:
- Don't reload the model in a loop — load once at startup
- Don't process text char by char — process full sentences/docs
- Don't use sm model for word similarity — it has no vectors (use md/lg)
- Don't ignore pipe disabling for production: parsing slows things down
"""

import spacy

# Load once
nlp = spacy.load("en_core_web_sm")

text = "Apple was founded by Steve Jobs in Cupertino, California in 1976."
doc = nlp(text)

# --- Tokens ---
print("=== Tokens ===")
print(f"{'Token':<15} {'Lemma':<12} {'POS':<8} {'Tag':<6} {'Stop?'}")
print("-" * 55)
for token in doc:
    print(f"{token.text:<15} {token.lemma_:<12} {token.pos_:<8} {token.tag_:<6} {token.is_stop}")

# --- Named Entities ---
print("\n=== Named Entities ===")
for ent in doc.ents:
    print(f"  {ent.text:<25} → {ent.label_:<10} ({spacy.explain(ent.label_)})")
# Apple     → ORG        (Companies, agencies)
# Steve Jobs → PERSON
# Cupertino → GPE        (Geopolitical entity)
# California → GPE
# 1976      → DATE

# --- Dependency parsing ---
print("\n=== Dependencies (key verbs) ===")
for token in doc:
    if token.dep_ in ("nsubj", "dobj", "ROOT"):
        print(f"  {token.text:<12} dep={token.dep_:<8} head={token.head.text}")

# --- Sentence splitting ---
print("\n=== Sentences ===")
multi = nlp("First sentence here. Second sentence follows. And a third.")
for sent in multi.sents:
    print(f"  {sent.text.strip()}")

# --- Efficient processing: disable unused pipes ---
print("\n=== Speed: disable parser + ner for classification ===")
texts = ["Machine learning is great", "Python is powerful", "spaCy is fast"]
for doc in nlp.pipe(texts, disable=["parser", "ner"]):  # batch processing
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    print(f"  {tokens}")
