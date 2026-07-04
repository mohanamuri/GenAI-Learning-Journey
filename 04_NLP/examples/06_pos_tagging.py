# Author: Mohan Raju Amuri
"""
POS Tagging (Part-of-Speech)
-----------------------------
One-liner: Label each word as noun, verb, adjective etc. — gives grammar context to NLP.

Remember:
- NLTK uses Penn Treebank tags: NN=noun, VB=verb, JJ=adjective, RB=adverb
- spaCy uses Universal POS tags: NOUN, VERB, ADJ, ADV (simpler)
- POS tagging is a prerequisite for: lemmatization, NER, dependency parsing
- Context changes POS: "book a flight" (VB) vs "read a book" (NN)

Don't:
- Don't run POS tagging on raw text — tokenize first
- Don't ignore POS when lemmatizing (biggest mistake in NLP pipelines)
- Don't use NLTK pos_tag on the full string — it needs a list of tokens
"""

import nltk
from nltk.tokenize import word_tokenize

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

# --- Basic POS tagging ---
text = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(text)
tagged = nltk.pos_tag(tokens)

print("Tagged:", tagged)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]

# --- Tag reference (most common) ---
tag_map = {
    "NN": "Noun (singular)",
    "NNS": "Noun (plural)",
    "NNP": "Proper Noun",
    "VB": "Verb (base)",
    "VBG": "Verb (gerund: running)",
    "VBD": "Verb (past: ran)",
    "JJ": "Adjective",
    "JJR": "Adjective (comparative: faster)",
    "RB": "Adverb",
    "DT": "Determiner (the, a)",
    "IN": "Preposition",
    "PRP": "Personal pronoun (I, he)",
    "CC": "Conjunction (and, but)",
}

print("\n--- Tag meanings ---")
for word, tag in tagged:
    meaning = tag_map.get(tag, tag)
    print(f"  {word:<10} {tag:<6} {meaning}")

# --- Context changes POS ---
print("\n--- Same word, different POS ---")
for sentence in [
    "Please book a flight to New York",   # book = verb
    "I left my book on the table",        # book = noun
]:
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if word.lower() == "book":
            print(f"  '{sentence}' → 'book' is {tag_map.get(tag, tag)}")

# --- Extract only nouns from text ---
print("\n--- Extract nouns only ---")
news = "Apple announced a new iPhone at the WWDC conference in California"
tokens = word_tokenize(news)
tagged = nltk.pos_tag(tokens)
nouns = [w for w, t in tagged if t in ("NN", "NNS", "NNP", "NNPS")]
print("Nouns:", nouns)
