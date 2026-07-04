# Author: Mohan Raju Amuri
"""
04_faiss_basics.py — Fast similarity search over millions of vectors with FAISS

What to remember:
- FAISS = Facebook AI Similarity Search — in-memory, blazing fast, no server needed
- IndexFlatL2 = exact brute-force search (use for < 100k vectors or when accuracy matters)
- IndexIVFFlat = approximate search with inverted file index (use for millions of vectors)
- FAISS returns distances, not similarities — with L2: lower distance = more similar

What NOT to do:
- Don't use IndexFlatL2 for millions of vectors — O(N) scan, gets slow fast
- Don't forget to normalize vectors if you want cosine similarity with IndexFlatIP
- Don't lose the mapping from FAISS index ID → original chunk text — FAISS only stores vectors

Two index types:
  IndexFlatL2     — exact, no training, best for < 50k vectors
  IndexFlatIP     — exact, cosine similarity (use with normalized vectors)
  IndexIVFFlat    — approximate, needs training, scales to millions

Interview one-liner:
  "FAISS is an in-memory vector index — add embeddings, query with a vector, get top-k IDs back."
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading embedding model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
DIM = model.get_sentence_embedding_dimension()  # 384 for MiniLM


# ── Build a FAISS index from a list of texts ──────────────────────────────────
class FAISSIndex:
    """Simple wrapper: maps FAISS integer IDs back to original text chunks."""

    def __init__(self, use_cosine: bool = True):
        # IndexFlatIP with normalized vectors = cosine similarity
        # IndexFlatL2 = Euclidean distance (lower = better)
        self.index = faiss.IndexFlatIP(DIM) if use_cosine else faiss.IndexFlatL2(DIM)
        self.id_to_text: dict[int, str] = {}
        self.id_to_meta: dict[int, dict] = {}
        self.use_cosine = use_cosine

    def add(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """Embed texts and add to FAISS. Assigns sequential integer IDs."""
        if not texts:
            return
        metadatas = metadatas or [{} for _ in texts]

        # Embed — normalize for cosine similarity
        vecs = model.encode(texts, normalize_embeddings=self.use_cosine,
                            show_progress_bar=False).astype("float32")

        start_id = self.index.ntotal
        self.index.add(vecs)

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            self.id_to_text[start_id + i] = text
            self.id_to_meta[start_id + i] = meta

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top-k results with score, text, and metadata."""
        q_vec = model.encode([query], normalize_embeddings=self.use_cosine,
                              show_progress_bar=False).astype("float32")
        scores, ids = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:  # FAISS returns -1 when fewer than k results exist
                continue
            results.append({
                "score": float(score),
                "text": self.id_to_text[idx],
                "metadata": self.id_to_meta[idx],
            })
        return results

    def __len__(self):
        return self.index.ntotal


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Step 4: FAISS Vector Index")
    print("=" * 60)

    # 1. Build index from TechCorp KB chunks
    print("\n[1] Building FAISS index from knowledge base chunks")
    chunks = [
        ("Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",
         {"section": "pricing", "plan": "starter"}),
        ("Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",
         {"section": "pricing", "plan": "professional"}),
        ("Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",
         {"section": "pricing", "plan": "enterprise"}),
        ("The platform supports TensorFlow, PyTorch, and scikit-learn frameworks.",
         {"section": "overview"}),
        ("SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",
         {"section": "security"}),
        ("99.9% uptime SLA for Professional and Enterprise plans. Starter is 99.5%.",
         {"section": "sla"}),
        ("14-day free trial with full Professional features, no credit card required.",
         {"section": "faq"}),
        ("Models deploy as REST API endpoints (auto-scaling, pay-per-request).",
         {"section": "deployment"}),
        ("Batch inference pipeline: scheduled or triggered processing of large datasets.",
         {"section": "deployment"}),
        ("Edge deployment: ONNX export for running models on edge devices offline.",
         {"section": "deployment"}),
        ("Enterprise support: 24/7 phone + Slack, dedicated support engineer.",
         {"section": "support", "plan": "enterprise"}),
        ("Starter support: community forum + email with 48-hour response time.",
         {"section": "support", "plan": "starter"}),
        ("Supported NLP models: BERT, GPT-2, T5, and custom transformer architectures.",
         {"section": "models", "type": "nlp"}),
        ("Supported vision models: ResNet, EfficientNet, YOLO for computer vision tasks.",
         {"section": "models", "type": "cv"}),
        ("Python SDK: client = techcorp.Client(api_key=...) then model.predict(data).",
         {"section": "sdk"}),
    ]

    index = FAISSIndex(use_cosine=True)
    texts, metas = zip(*chunks)
    index.add(list(texts), list(metas))
    print(f"  Indexed {len(index)} chunks  |  vector dim: {DIM}")

    # 2. Run queries
    print("\n[2] Running similarity queries")
    queries = [
        "How much does the cheapest plan cost?",
        "Is the platform GDPR compliant and secure?",
        "How do I deploy my model to production?",
        "What Python code do I need to make a prediction?",
        "Do you offer enterprise customer support?",
    ]

    for query in queries:
        results = index.search(query, top_k=2)
        print(f"\n  Query: '{query}'")
        for r in results:
            print(f"    [{r['score']:.3f}]  ({r['metadata'].get('section','?')})  {r['text'][:70]}")

    # 3. Show index vs flat comparison (when to upgrade)
    print("\n[3] Index types — when to use which")
    print("  IndexFlatL2 / IndexFlatIP:")
    print("    ✓ Exact results")
    print("    ✓ No training step")
    print("    ✗ O(N) scan — 1M vectors ~1s per query")
    print("    → Use for: prototypes, < 50k vectors")
    print()
    print("  IndexIVFFlat (Approximate):")
    print("    ✓ ~100x faster at scale")
    print("    ✗ Needs training on representative data")
    print("    ✗ May miss some true top-k (recall ~95%)")
    print("    → Use for: production, > 100k vectors")

    # 4. Demonstrate IVF index (approximate, fast at scale)
    print("\n[4] IVFFlat index (approximate search, scales to millions)")
    # nlist = number of cluster centroids — rule of thumb: sqrt(N)
    nlist = 4  # small value for demo (normally sqrt(num_vectors))
    quantizer = faiss.IndexFlatIP(DIM)
    ivf_index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT)

    # IVF requires training before adding vectors
    all_vecs = model.encode(list(texts), normalize_embeddings=True,
                             show_progress_bar=False).astype("float32")
    ivf_index.train(all_vecs)
    ivf_index.add(all_vecs)
    ivf_index.nprobe = 2  # search 2 of 4 clusters — trades recall for speed

    q_vec = model.encode(["How much does the starter plan cost?"],
                          normalize_embeddings=True, show_progress_bar=False).astype("float32")
    scores, ids = ivf_index.search(q_vec, 2)
    print(f"  IVF top-2 results for 'How much does the starter plan cost?'")
    for score, idx in zip(scores[0], ids[0]):
        print(f"    [{score:.3f}]  {texts[idx][:70]}")

    # 5. Save & reload (important for production)
    print("\n[5] Saving and loading FAISS index")
    faiss.write_index(ivf_index, "/tmp/techcorp_faiss.index")
    loaded = faiss.read_index("/tmp/techcorp_faiss.index")
    print(f"  Saved to /tmp/techcorp_faiss.index")
    print(f"  Loaded index contains {loaded.ntotal} vectors")
    print("  Note: FAISS only saves vectors — you must also pickle id_to_text separately!")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - FAISS stores vectors + returns IDs, not text — keep a mapping dict")
    print("  - IndexFlatIP + normalized vectors = cosine similarity, range [0,1]")
    print("  - IVFFlat is approximate but 100x faster at scale")
    print("  - nprobe controls speed/recall tradeoff in IVF (higher = more accurate)")
    print("  - For persistent storage with metadata, use ChromaDB (next example)")
    print("=" * 60)
