"""
Stemming
--------
One-liner: Chop word endings using rules — fast but crude (running → run, studies → studi).

Remember:
- Stemming does NOT produce real words — just common root forms
- Porter stemmer is the most common (NLTK default)
- Use for search engines / IR where exact word form doesn't matter
- Faster than lemmatization, no dictionary lookup needed

Don't:
- Don't use stemming when the output word needs to be human-readable
  → "studies" → "studi"  (not a real word)
- Don't use stemming for NLP models that expect real vocabulary
- Don't use stemming when context changes meaning:
  → "meeting" → "meet" (could be noun or verb — context lost)
"""

from nltk.stem import PorterStemmer, SnowballStemmer

ps = PorterStemmer()
ss = SnowballStemmer("english")

words = [
    "running", "runs", "runner",
    "studies", "studying", "studied",
    "happiness", "happily", "happy",
    "beautiful", "beautifully",
    "connection", "connected", "connecting",
]

print(f"{'Word':<15} {'Porter':<15} {'Snowball':<15}")
print("-" * 45)
for w in words:
    print(f"{w:<15} {ps.stem(w):<15} {ss.stem(w):<15}")

# Word            Porter          Snowball
# running         run             run
# studies         studi           studi         ← not a real word
# happiness       happi           happi         ← not a real word
# beautiful       beauti          beauti

# --- When stemming makes sense: search index ---
print("\n--- Search use case ---")
query = "running shoes"
docs = [
    "I love to run in the park",
    "She runs marathons every year",
    "Best running gear for beginners",
    "Python programming tutorial",
]
query_stems = {ps.stem(w) for w in query.split()}
print("Query stems:", query_stems)
for doc in docs:
    doc_stems = {ps.stem(w) for w in doc.lower().split()}
    if query_stems & doc_stems:  # intersection
        print(f"  Match: {doc}")
