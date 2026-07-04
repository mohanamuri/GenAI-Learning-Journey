# Framework: FAISS (Facebook AI Similarity Search)

## What is it?
An in-memory C++ library (with Python bindings) for fast nearest-neighbor search over dense float vectors.
Built by Meta AI Research. The de-facto standard for vector similarity search.

## What existed before?
- **Brute-force numpy**: `np.dot(query, corpus.T)` — exact but O(N), no scaling
- **scikit-learn NearestNeighbors**: works for small datasets, slow at scale
- **KD-trees / Ball trees**: fast for low dimensions, degrade badly above ~20 dims

## Why we picked FAISS
- Handles 1M+ vectors with millisecond latency
- Multiple index types: exact (Flat) and approximate (IVF, HNSW, PQ)
- GPU support (faiss-gpu) for trillion-scale search
- Used in production at Meta, Spotify, Pinterest, Airbnb

## Core concepts

| Index | Type | When to use |
|-------|------|-------------|
| `IndexFlatL2` | Exact, Euclidean | < 50k vectors, need 100% accuracy |
| `IndexFlatIP` | Exact, cosine (normalize first) | < 50k vectors, cosine similarity |
| `IndexIVFFlat` | Approximate, clusters | > 100k vectors, good speed/recall |
| `IndexHNSW` | Approximate, graph-based | Low latency, high recall at any scale |
| `IndexPQ` | Compressed, approximate | Billions of vectors, memory-constrained |

## Key code pattern

```python
import faiss, numpy as np

dim = 384  # embedding dimension
index = faiss.IndexFlatIP(dim)  # cosine with normalized vectors

# Add vectors (float32 required)
vecs = encoder.encode(texts, normalize_embeddings=True).astype("float32")
index.add(vecs)

# Search
q = encoder.encode([query], normalize_embeddings=True).astype("float32")
scores, ids = index.search(q, k=5)  # ids = row indices in original array
# ⚠️  FAISS returns IDs, not text — maintain your own id→text mapping!

# Persist
faiss.write_index(index, "index.faiss")
index = faiss.read_index("index.faiss")
```

## When to use vs skip

**Use FAISS when:**
- Speed is the top priority
- You have > 100k vectors
- You need GPU-accelerated search (faiss-gpu)
- You're building a custom retrieval system

**Skip FAISS (use ChromaDB) when:**
- You need metadata storage and filtering
- You want persistence without writing custom code
- You're prototyping and want simpler API

## Install
```bash
pip install faiss-cpu          # CPU-only (most common)
pip install faiss-gpu          # GPU version (requires CUDA)
```
