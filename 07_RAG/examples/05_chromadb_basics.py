# Author: Mohan Raju Amuri
"""
05_chromadb_basics.py — Persistent vector store with metadata filtering

What to remember:
- ChromaDB = vector DB with persistence, metadata, and filtering built in
- Unlike FAISS, it stores text + metadata + embeddings together — no external mapping
- Supports metadata filters: {"section": "pricing"} to narrow search scope
- Runs as an in-process library (no server) or as a standalone server

What NOT to do:
- Don't forget to delete the collection before re-running (or use get_or_create)
- Don't assume ChromaDB is faster than FAISS — it's slower but much more convenient
- Don't use ChromaDB for billions of vectors — it's a dev/small-scale tool

FAISS vs ChromaDB:
  FAISS       — faster, in-memory, no metadata filtering, manual ID management
  ChromaDB    — persistent, metadata filtering, text storage, easier API

Interview one-liner:
  "ChromaDB is a batteries-included vector store: embeddings + metadata + text, with filtering."
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

# ── Setup ─────────────────────────────────────────────────────────────────────
# Use a persistent directory so the index survives restarts
PERSIST_DIR = "/tmp/chromadb_techcorp"
COLLECTION_NAME = "techcorp_kb"

print(f"Initializing ChromaDB at: {PERSIST_DIR}")
client = chromadb.PersistentClient(path=PERSIST_DIR)

# Tell ChromaDB which embedding model to use — it auto-embeds on add/query
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# ── Create / load collection ──────────────────────────────────────────────────
# delete_collection is idempotent — safe to call if it doesn't exist
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # use cosine distance
)
print(f"Created collection: '{COLLECTION_NAME}'")


# ── Add documents ─────────────────────────────────────────────────────────────
# Each document needs: id (unique string), document (text), metadata (dict)
documents = [
    "Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",
    "Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",
    "Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",
    "SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",
    "99.9% uptime SLA for Professional and Enterprise plans. Starter is 99.5%.",
    "14-day free trial with full Professional features, no credit card required.",
    "Models deploy as REST API endpoints (auto-scaling, pay-per-request).",
    "Edge deployment: ONNX export for running models on edge devices offline.",
    "Enterprise support: 24/7 phone + Slack, dedicated support engineer.",
    "Starter support: community forum + email with 48-hour response time.",
    "Supported NLP models: BERT, GPT-2, T5, custom transformer architectures.",
    "Supported vision models: ResNet, EfficientNet, YOLO for computer vision.",
    "Python SDK: import techcorp; client = techcorp.Client(api_key='your-key')",
    "Data formats supported: CSV, Parquet, JSON, images (JPEG/PNG), SDK connectors.",
    "RBAC with SSO support. Audit logs retained for 90 days.",
]

metadatas = [
    {"section": "pricing", "plan": "starter"},
    {"section": "pricing", "plan": "professional"},
    {"section": "pricing", "plan": "enterprise"},
    {"section": "security"},
    {"section": "sla"},
    {"section": "faq"},
    {"section": "deployment", "type": "api"},
    {"section": "deployment", "type": "edge"},
    {"section": "support", "plan": "enterprise"},
    {"section": "support", "plan": "starter"},
    {"section": "models", "type": "nlp"},
    {"section": "models", "type": "cv"},
    {"section": "sdk"},
    {"section": "faq"},
    {"section": "security"},
]

ids = [f"doc_{i}" for i in range(len(documents))]

collection.add(documents=documents, metadatas=metadatas, ids=ids)
print(f"Added {collection.count()} documents to collection")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Step 5: ChromaDB — Persistent Vector Store")
    print("=" * 60)

    # 1. Basic semantic search
    print("\n[1] Basic semantic search (no filter)")
    results = collection.query(
        query_texts=["How much does the cheapest plan cost?"],
        n_results=3,
    )
    print("  Query: 'How much does the cheapest plan cost?'")
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        print(f"    #{i+1} dist={dist:.3f}  [{meta.get('section','?')}]  {doc[:70]}")

    # 2. Metadata filtering — only look in the pricing section
    print("\n[2] With metadata filter: section=pricing")
    results = collection.query(
        query_texts=["What is included in enterprise?"],
        n_results=3,
        where={"section": "pricing"},  # pre-filter before vector search
    )
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
    )):
        print(f"    #{i+1} [{meta.get('plan','?')}]  {doc[:70]}")

    # 3. Multiple filters with $and
    print("\n[3] Filter: section=support AND plan=enterprise")
    results = collection.query(
        query_texts=["support options"],
        n_results=2,
        where={"$and": [{"section": "support"}, {"plan": "enterprise"}]},
    )
    for doc in results["documents"][0]:
        print(f"    {doc}")

    # 4. Get by ID (no search, direct lookup)
    print("\n[4] Direct get by ID (no vector search)")
    item = collection.get(ids=["doc_0", "doc_1"])
    for doc, meta in zip(item["documents"], item["metadatas"]):
        print(f"    {meta}  {doc[:60]}")

    # 5. Update a document
    print("\n[5] Update a document (simulate KB refresh)")
    collection.update(
        ids=["doc_0"],
        documents=["Starter Plan: $89/month (REVISED) — up to 5 users, 100 GB storage"],
        metadatas=[{"section": "pricing", "plan": "starter", "version": "2"}],
    )
    updated = collection.get(ids=["doc_0"])
    print(f"    Updated: {updated['documents'][0]}")

    # 6. Collection metadata and count
    print("\n[6] Collection info")
    print(f"    Name:  {collection.name}")
    print(f"    Count: {collection.count()} documents")
    print(f"    Path:  {PERSIST_DIR}")

    # 7. When to use FAISS vs ChromaDB
    print("\n[7] FAISS vs ChromaDB — quick comparison")
    comparison = [
        ("Persistence",         "Manual (write_index)", "Built-in"),
        ("Metadata storage",    "You manage it",        "Built-in"),
        ("Metadata filtering",  "Not supported",        "Supported ($and, $or, $in)"),
        ("Speed",               "Faster",               "Slightly slower"),
        ("Best for",            "Speed at scale",       "Prototyping & metadata-rich KBs"),
    ]
    print(f"    {'Feature':<25} {'FAISS':<25} {'ChromaDB'}")
    for feat, f, c in comparison:
        print(f"    {feat:<25} {f:<25} {c}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - ChromaDB auto-embeds on add and query — you just pass text")
    print("  - Metadata filters narrow the search space before vector similarity")
    print("  - PersistentClient survives restarts; EphemeralClient is in-memory only")
    print("  - ids must be unique strings — use filename+chunk_index in production")
    print("  - Next: combine loading + chunking + ChromaDB into a full RAG pipeline")
    print("=" * 60)
