"""
Word2Vec
--------
One-liner: Words with similar contexts get similar vector representations (king - man + woman ≈ queen).

Remember:
- Two architectures: CBOW (context → center) and Skip-gram (center → context)
- Skip-gram is better for rare words; CBOW is faster
- vector_size=100-300, window=5, min_count=2 are good defaults
- Similar words: model.wv.most_similar("king")
- Arithmetic: model.wv.most_similar(positive=["king", "woman"], negative=["man"])
- Pretrained models (Google News: 3M words, 300d) beat small custom training

Don't:
- Don't train Word2Vec on tiny corpora — needs millions of words to be meaningful
- Don't expect it to know words outside training vocab (KeyError on unknown words)
- Don't use Word2Vec for sentence-level tasks — it's word-level only
  → Use sentence embeddings (sentence-transformers) for sentence similarity
- Don't confuse Word2Vec with LLM embeddings — W2V is static, LLM is contextual
  → "bank" (river) and "bank" (money) have SAME vector in Word2Vec
"""

from gensim.models import Word2Vec
import numpy as np

# --- Train on toy corpus ---
sentences = [
    ["king", "rules", "the", "kingdom"],
    ["queen", "rules", "the", "kingdom"],
    ["man", "goes", "to", "the", "market"],
    ["woman", "goes", "to", "the", "market"],
    ["prince", "is", "son", "of", "king"],
    ["princess", "is", "daughter", "of", "queen"],
    ["dog", "is", "a", "loyal", "animal"],
    ["cat", "is", "a", "small", "animal"],
    ["python", "is", "a", "programming", "language"],
    ["java", "is", "a", "programming", "language"],
    ["machine", "learning", "uses", "data"],
    ["deep", "learning", "uses", "neural", "networks"],
]

model = Word2Vec(
    sentences,
    vector_size=50,    # embedding dimensions
    window=3,          # context window size
    min_count=1,       # include words with freq >= 1 (use 2+ in real use)
    workers=1,         # reproducibility
    epochs=100,
    seed=42,
)

# --- Most similar words ---
print("Most similar to 'king':", model.wv.most_similar("king", topn=3))
print("Most similar to 'python':", model.wv.most_similar("python", topn=3))

# --- Word arithmetic (may not work well on tiny corpus) ---
print("\nking - man + woman:")
result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
print(" ", result)

# --- Cosine similarity between words ---
print("\nSimilarity scores:")
pairs = [("king", "queen"), ("king", "man"), ("python", "java"), ("dog", "cat")]
for w1, w2 in pairs:
    sim = model.wv.similarity(w1, w2)
    print(f"  {w1:10} ↔ {w2:10} = {sim:.3f}")

# --- Vector for a word ---
print(f"\nVector for 'king' (first 10 dims): {model.wv['king'][:10].round(3)}")
print(f"Vocabulary size: {len(model.wv)}")

# --- Handle unknown words ---
print("\n--- Unknown word handling ---")
try:
    _ = model.wv["blockchain"]
except KeyError:
    print("'blockchain' not in vocab — use try/except or check with: 'word' in model.wv")
    print("In vocab:", "python" in model.wv)
