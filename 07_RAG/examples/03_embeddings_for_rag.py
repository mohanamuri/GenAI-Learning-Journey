# Author: Mohan Raju Amuri
"""
03_embeddings_for_rag.py — Convert text chunks into vectors for similarity search

What to remember:
- Embedding = a list of floats that captures the meaning of text
- Similar meaning → vectors are close (high cosine similarity)
- In RAG: embed ALL chunks once at index time, embed query at search time
- sentence-transformers all-MiniLM-L6-v2 is the go-to: 80 MB, fast, good quality

What NOT to do:
- Don't re-embed your entire corpus on every search — embed once, store, reuse
- Don't compare embeddings from different models — dimensions and scales differ
- Don't use raw word-count vectors (TF-IDF) for RAG — they miss semantic meaning
  (e.g., "automobile" and "car" score 0 similarity with TF-IDF, ~0.9 with embeddings)

Interview one-liner:
  "Embeddings map text to a vector space where semantic similarity = cosine distance."
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ── Load the embedding model ──────────────────────────────────────────────────
# Downloads ~80 MB on first run, cached at ~/.cache/huggingface
# 384-dimensional vectors — good balance of speed and quality
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")


# ── Cosine Similarity ─────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = dot(a, b) / (|a| * |b|)  — range [-1, 1], higher = more similar"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Embed a list of texts (batch) ─────────────────────────────────────────────
def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns shape (N, embedding_dim) numpy array. Normalised by default."""
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Step 3: Embeddings for Retrieval")
    print("=" * 60)

    # 1. Show what an embedding looks like
    sample = "TechCorp AI Platform supports GPU training."
    vec = model.encode(sample, normalize_embeddings=True)
    print(f"\n[1] Embedding shape for one sentence: {vec.shape}")
    print(f"    First 8 values: {vec[:8].round(4)}")
    print(f"    Norm (should be 1.0 after normalize): {np.linalg.norm(vec):.4f}")

    # 2. Semantic similarity: same meaning, different words
    print("\n[2] Semantic similarity — same meaning, different words")
    pairs = [
        ("What is the price of the Starter plan?", "How much does the Starter tier cost?"),
        ("What is the price of the Starter plan?", "What GPUs are supported?"),
        ("GPU training is available", "Training on graphics cards"),
        ("GPU training is available", "The weather is nice today"),
    ]
    for a, b in pairs:
        va, vb = model.encode([a, b], normalize_embeddings=True)
        sim = cosine_similarity(va, vb)
        print(f"    {sim:+.3f}  |  '{a[:40]}' ↔ '{b[:40]}'")

    # 3. Embed knowledge base chunks
    print("\n[3] Embedding knowledge base chunks")
    kb_chunks = [
        "Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",
        "Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",
        "Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",
        "The platform supports TensorFlow, PyTorch, and scikit-learn.",
        "SOC 2 Type II certified. Data encrypted at rest (AES-256) and in transit (TLS 1.3).",
        "99.9% uptime SLA for Professional and Enterprise plans.",
        "14-day free trial with full Professional features, no credit card required.",
        "Models can be deployed as REST API endpoints, batch pipelines, or edge devices.",
        "Starter plan support: community forum + email (48h response time).",
        "Enterprise support: 24/7 phone + Slack, dedicated support engineer.",
    ]

    chunk_embeddings = embed_texts(kb_chunks)
    print(f"    Embedded {len(kb_chunks)} chunks → shape: {chunk_embeddings.shape}")

    # 4. Retrieve top-k for a query
    print("\n[4] Query-time retrieval (top-3 most relevant chunks)")
    queries = [
        "How much does the starter plan cost?",
        "Is my data secure on TechCorp?",
        "What kind of support do I get with the enterprise plan?",
    ]

    for query in queries:
        q_vec = model.encode(query, normalize_embeddings=True)
        sims = [cosine_similarity(q_vec, c_vec) for c_vec in chunk_embeddings]

        # Get top-3 indices sorted by similarity (descending)
        top_k = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]

        print(f"\n    Query: '{query}'")
        for rank, idx in enumerate(top_k):
            print(f"      #{rank+1} ({sims[idx]:.3f})  {kb_chunks[idx][:80]}")

    # 5. Why embeddings beat keyword search
    print("\n[5] Keyword vs semantic search comparison")
    query = "What does the platform cost?"
    keyword_target = "TechCorp pricing and plans"   # low keyword overlap
    semantic_target = "$99/month — up to 5 users"  # actually answers the question

    q_vec = model.encode(query, normalize_embeddings=True)
    k_vec = model.encode(keyword_target, normalize_embeddings=True)
    s_vec = model.encode(semantic_target, normalize_embeddings=True)

    print(f"    Query: '{query}'")
    print(f"    Keyword-ish match '{keyword_target}': {cosine_similarity(q_vec, k_vec):.3f}")
    print(f"    Semantic match   '{semantic_target}': {cosine_similarity(q_vec, s_vec):.3f}")
    print("    → Semantic search finds the right answer even without keyword overlap")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - embed once at index time, embed query at search time")
    print("  - cosine similarity: > 0.7 = strong match, < 0.3 = unrelated")
    print("  - normalize_embeddings=True makes dot product == cosine similarity")
    print("  - all-MiniLM-L6-v2: 384-dim, 80 MB, best default for RAG")
    print("  - Next: store these embeddings in FAISS for fast search over millions")
    print("=" * 60)
