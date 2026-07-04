# Author: Mohan Raju Amuri
"""
Lemmatization
-------------
One-liner: Reduce word to its base dictionary form using vocabulary + grammar context.

Remember:
- Lemmatization produces REAL words (running → run, better → good)
- POS tag improves accuracy: lemmatize("meeting", pos="n") → "meeting"
                              lemmatize("meeting", pos="v") → "meet"
- Slower than stemming, needs WordNet dictionary
- Prefer lemmatization when output will be read or fed to a model

Don't:
- Don't skip POS tag — default is noun, gives wrong results for verbs
  → lemmatize("running")        → "running"  ✗ (treated as noun)
  → lemmatize("running", "v")   → "run"      ✓
- Don't use NLTK lemmatizer for production — use spaCy (more accurate, faster)
- Don't lemmatize when you've already stemmed (pick one)
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

lem = WordNetLemmatizer()

# --- POS matters ---
print("--- POS tag comparison ---")
words_with_pos = [
    ("running",  "n", "v"),
    ("meeting",  "n", "v"),
    ("better",   "n", "a"),
    ("studies",  "n", "v"),
    ("wolves",   "n", "v"),
]
print(f"{'Word':<12} {'noun form':<15} {'verb/adj form':<15}")
print("-" * 42)
for word, pos1, pos2 in words_with_pos:
    print(f"{word:<12} {lem.lemmatize(word, pos1):<15} {lem.lemmatize(word, pos2):<15}")

# running        running         run          ← noun vs verb matters!
# better         better          good         ← adjective gives "good"


# --- Auto POS tagging helper ---
def get_wordnet_pos(treebank_tag: str) -> str:
    """Map NLTK POS tag to WordNet POS."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_sentence(sentence: str) -> list[str]:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(sentence.lower())
    tagged = nltk.pos_tag(tokens)
    return [lem.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]


sentence = "The dogs are running faster and the cats were sleeping"
print("\n--- Full sentence lemmatization ---")
print("Input :", sentence)
print("Output:", lemmatize_sentence(sentence))
# Output: ['the', 'dog', 'be', 'run', 'fast', 'and', 'the', 'cat', 'be', 'sleep']

# --- Stemming vs Lemmatization summary ---
print("\n--- Stemming vs Lemmatization ---")
print(f"{'Word':<12} {'Stem':<12} {'Lemma':<12}")
print("-" * 36)
from nltk.stem import PorterStemmer
ps = PorterStemmer()
test = ["studies", "running", "happiness", "wolves", "better"]
for w in test:
    print(f"{w:<12} {ps.stem(w):<12} {lem.lemmatize(w, wordnet.VERB):<12}")
