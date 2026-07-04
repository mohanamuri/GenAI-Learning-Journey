"""
Tokenization
------------
One-liner: Splitting text into units (words/sentences) the model can process.

Remember:
- word_tokenize handles punctuation better than split()
- sentence tokenization uses period + context (not just ".")
- NLTK needs: nltk.download('punkt_tab')
- spaCy tokenizer is more accurate for production

Don't:
- Don't use text.split() and call it tokenization
  → "don't" → ["don't"]  ✓ with NLTK
  → "don't" → ["don't"]  ✗ with split() — loses contraction split
- Don't forget to download NLTK data before first use
- Don't tokenize before cleaning (unless you need original casing)
"""

import nltk

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize

text = "I can't believe it's not butter! This is amazing. Let's go."

# Word tokenization
word_tokens = word_tokenize(text)
print("Word tokens:", word_tokens)
# ['I', 'ca', "n't", 'believe', "it's", 'not', 'butter', '!', ...]
# Notice: "can't" → ["ca", "n't"]  — NLTK splits contractions

# Sentence tokenization
sent_tokens = sent_tokenize(text)
print("\nSentences:")
for i, s in enumerate(sent_tokens, 1):
    print(f"  {i}: {s}")

# --- Why split() fails ---
print("\n--- split() vs word_tokenize ---")
dirty = "Hello, world! It's a test."
print("split()        :", dirty.split())
print("word_tokenize():", word_tokenize(dirty))
# split()        : ['Hello,', 'world!', "It's", 'a', 'test.']  ← punctuation stuck
# word_tokenize(): ['Hello', ',', 'world', '!', 'It', "'s", 'a', 'test', '.']  ← clean

# --- Token count (useful for LLM context window awareness) ---
print(f"\nToken count: {len(word_tokens)}")
