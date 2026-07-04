# Author: Mohan Raju Amuri
"""
Text Similarity
---------------
One-liner: Measure how alike two texts are — cosine similarity is the standard.

Methods (in order of increasing quality):
  1. Jaccard similarity   — word overlap / union (no weights)
  2. Cosine + TF-IDF      — weighted word overlap (fast, no model needed)
  3. Word2Vec / GloVe avg — semantic meaning, handles synonyms
  4. Sentence embeddings  — best quality, context-aware (production choice)

Remember:
- Cosine similarity measures DIRECTION not magnitude → length-normalized
- Range: 0 (completely different) to 1 (identical)
- TF-IDF cosine is good enough for document-level similarity
- Use sentence-transformers for semantic search / FAQ matching
- "dog" and "puppy" → low TF-IDF similarity, high embedding similarity

Don't:
- Don't use Euclidean distance on text vectors — use cosine
- Don't average Word2Vec vectors naively for long docs — use SIF weighting
- Don't use simple string matching (difflib) for semantic similarity
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Method 1: Jaccard Similarity ---
def jaccard(s1: str, s2: str) -> float:
    a, b = set(s1.lower().split()), set(s2.lower().split())
    return len(a & b) / len(a | b)

# --- Method 2: TF-IDF Cosine ---
def tfidf_similarity(texts: list[str]) -> np.ndarray:
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    return cosine_similarity(X)

# --- Texts to compare ---
sentences = [
    "Machine learning is a subset of artificial intelligence",
    "AI and machine learning are closely related fields",
    "Deep learning uses neural networks with many layers",
    "I love eating pizza and pasta for dinner",
]

print("=== Jaccard Similarity ===")
ref = sentences[0]
for s in sentences[1:]:
    print(f"  {jaccard(ref, s):.3f}  |  {s[:50]}")

print("\n=== TF-IDF Cosine Similarity ===")
sim_matrix = tfidf_similarity(sentences)
print(f"{'':30}", end="")
for i in range(len(sentences)):
    print(f"  S{i+1}", end="")
print()
for i, row in enumerate(sim_matrix):
    print(f"S{i+1} {sentences[i][:27]:<28}", end="")
    for v in row:
        print(f"  {v:.2f}", end="")
    print()

# --- Method 3: Sentence embeddings (best practice) ---
# pip install sentence-transformers
print("\n=== Sentence Transformers (semantic) ===")
try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences)
    sim = cosine_similarity(embeddings)
    print("Semantic similarity matrix:")
    for i, row in enumerate(sim):
        print(f"  S{i+1}: {[round(v, 2) for v in row]}")
    # Notice S1 & S2 are much more similar semantically than via TF-IDF
except ImportError:
    print("  Install: pip install sentence-transformers")
    print("  Semantic similarity would rank S1~S2 much higher than TF-IDF does")

# --- FAQ matching (practical use case) ---
print("\n=== FAQ matching ===")
faqs = [
    "How do I reset my password?",
    "What is the refund policy?",
    "How do I track my order?",
    "What payment methods are accepted?",
]
user_query = "I forgot my password and can't login"
all_texts = faqs + [user_query]
sim = tfidf_similarity(all_texts)
scores = sim[-1][:-1]  # query vs each FAQ
best_match_idx = np.argmax(scores)
print(f"Query: {user_query}")
print(f"Match: {faqs[best_match_idx]} (score: {scores[best_match_idx]:.3f})")
