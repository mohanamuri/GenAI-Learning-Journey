# Framework: ChromaDB

## What is it?
An open-source vector database designed for AI applications.
Stores embeddings + metadata + original text together.
Can run in-process (no server), as a persistent local DB, or as a client-server system.

## What existed before?
- **FAISS + pickle**: manual id-to-text mapping, no metadata, no persistence without extra code
- **Pinecone**: managed cloud vector DB — no free tier, vendor lock-in
- **Weaviate / Qdrant**: powerful but require running a Docker service
- **pgvector**: PostgreSQL extension — good for existing Postgres users but complex setup

## Why we picked ChromaDB
- Zero-config: runs in-process with `chromadb.EphemeralClient()` or `PersistentClient()`
- Stores text + metadata alongside vectors — no external mapping required
- Metadata filtering with `$and`, `$or`, `$in`, `$ne` operators
- Auto-embedding: plug in any embedding function, it handles the rest
- Great for prototyping; production-ready with the server mode

## Core concepts

| Client type | Persistence | Use when |
|-------------|-------------|----------|
| `EphemeralClient()` | In-memory only | Testing, demos |
| `PersistentClient(path)` | Saved to disk | Development, local apps |
| `HttpClient(host, port)` | Remote server | Production, multi-process |

## Key code pattern

```python
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Setup
client = chromadb.PersistentClient(path="/tmp/my_db")
ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection("my_kb", embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})

# Add documents (auto-embeds)
col.add(
    documents=["text1", "text2"],
    metadatas=[{"section": "pricing"}, {"section": "support"}],
    ids=["doc_0", "doc_1"],   # must be unique strings
)

# Query (auto-embeds the query)
results = col.query(
    query_texts=["how much does it cost?"],
    n_results=3,
    where={"section": "pricing"},   # optional metadata filter
)
# results["documents"][0]  → list of matching texts
# results["distances"][0]  → cosine distances (0=identical, 2=opposite)
# results["metadatas"][0]  → list of metadata dicts

# Distance → similarity
sim = 1.0 - distance / 2.0  # converts [0,2] → [0,1]
```

## Metadata filter operators

```python
# Single condition
where={"section": "pricing"}

# AND
where={"$and": [{"company": "techcorp"}, {"section": "pricing"}]}

# OR
where={"$or": [{"section": "pricing"}, {"section": "faq"}]}

# IN (multiple values)
where={"section": {"$in": ["pricing", "support", "sla"]}}

# NOT EQUAL
where={"section": {"$ne": "pricing"}}

# Greater than (for numeric metadata)
where={"year": {"$gte": 2024}}
```

## When to use vs skip

**Use ChromaDB when:**
- You need metadata filtering
- You want batteries-included (text + vectors + metadata together)
- Prototyping or building local apps
- You want simple Python API without running Docker

**Skip ChromaDB (use FAISS or Qdrant) when:**
- You need maximum search speed at billion-vector scale
- You need strict ACID transactions
- You already have Postgres and want pgvector

## Install
```bash
pip install chromadb
```
