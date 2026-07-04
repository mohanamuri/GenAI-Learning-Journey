"""
Embeddings — Text as Vectors in Semantic Space
------------------------------------------------
One-liner: Embeddings convert text into numbers that capture meaning — similar text → nearby vectors.

Remember:
- Embedding = dense vector (e.g. 384, 768, 1536 dimensions)
- Similar meaning → small cosine distance
- All-MiniLM-L6-v2: 384 dims, fast, great quality (default choice)
- OpenAI text-embedding-3-small: 1536 dims, API call, excellent quality
- Embeddings are the backbone of RAG (Module 07)

Use cases:
  Semantic search      → find docs by meaning, not keywords
  Clustering           → group similar documents
  Recommendation       → "similar to this item"
  Duplicate detection  → near-identical content
  Classification       → embed → feed to classifier

Don't:
- Don't use TF-IDF for semantic search — embeddings understand synonyms
- Don't compare embeddings from different models — they're incompatible
- Don't recompute embeddings every time — cache/store them in a vector DB
- Don't embed very long texts without chunking — quality degrades
  (max 256–512 tokens for best results with sentence-transformers)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── 1. Embed and compare sentences ────────────────────────────────────────
print("=== 1. Semantic Similarity ===\n")
reference = "How do I reset my password?"
candidates = [
    "I forgot my password and can't log in",       # semantically close
    "Steps to change account credentials",          # related
    "How to enable two-factor authentication",      # somewhat related
    "What is the weather today?",                   # unrelated
    "Password recovery steps for my account",       # very close
]

ref_emb = model.encode(reference)
print(f"Reference: '{reference}'\n")
for c in candidates:
    emb = model.encode(c)
    sim = cosine_sim(ref_emb, emb)
    bar = "█" * int(sim * 25)
    print(f"  {sim:.3f}  {bar:<25}  {c}")

# ── 2. Semantic search (FAQ matching) ────────────────────────────────────
print("\n=== 2. Semantic Search ===\n")
faqs = [
    "How do I cancel my subscription?",
    "What payment methods do you accept?",
    "How long does shipping take?",
    "Can I return a product?",
    "How do I track my order?",
    "What is your privacy policy?",
]
faq_embeddings = model.encode(faqs)

user_queries = [
    "I want to stop my membership",      # → cancel subscription
    "Do you take credit cards?",          # → payment methods
    "Where is my package?",               # → track order
]

for query in user_queries:
    query_emb = model.encode(query)
    scores = [cosine_sim(query_emb, faq_emb) for faq_emb in faq_embeddings]
    best_idx = int(np.argmax(scores))
    print(f"  Query  : {query}")
    print(f"  Matched: {faqs[best_idx]} (score: {scores[best_idx]:.3f})\n")

# ── 3. Clustering by topic ────────────────────────────────────────────────
print("=== 3. Clustering Documents ===\n")
from sklearn.cluster import KMeans

documents = [
    "Python is great for data science",
    "Machine learning needs lots of data",
    "Deep learning uses neural networks",
    "Football is played on a field",
    "Basketball requires dribbling skills",
    "Soccer is the world's most popular sport",
    "Pasta is made from flour and water",
    "Italian cuisine uses olive oil",
    "Pizza originated in Naples, Italy",
]

embeddings = model.encode(documents)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

clusters: dict[int, list[str]] = {}
for doc, label in zip(documents, labels):
    clusters.setdefault(label, []).append(doc)

for cluster_id, docs in sorted(clusters.items()):
    print(f"  Cluster {cluster_id}:")
    for doc in docs:
        print(f"    - {doc}")
    print()

# ── 4. Embedding dimensions ────────────────────────────────────────────────
print("=== 4. Embedding Info ===")
emb = model.encode("hello world")
print(f"Model  : all-MiniLM-L6-v2")
print(f"Dims   : {emb.shape[0]}")
print(f"Sample : {emb[:5].round(4)} ...")
print(f"\nCommon models:")
print(f"  all-MiniLM-L6-v2        384 dims  fast, good quality")
print(f"  all-mpnet-base-v2        768 dims  slower, better quality")
print(f"  text-embedding-3-small  1536 dims  OpenAI API, excellent")
print(f"  text-embedding-3-large  3072 dims  OpenAI API, best quality")
