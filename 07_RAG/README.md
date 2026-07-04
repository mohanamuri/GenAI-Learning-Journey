# Module 07 — RAG (Retrieval-Augmented Generation)

Stop hallucinating. Retrieve facts, then generate — ground LLM answers in your own data.

---

## What We Built

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | `01_document_loading.py` | Document class with content + metadata, loaders for text/JSON |
| 02 | `02_chunking.py` | Fixed-size, paragraph-aware, and recursive chunking strategies |
| 03 | `03_embeddings_for_rag.py` | Sentence embeddings with all-MiniLM-L6-v2, cosine similarity |
| 04 | `04_faiss_basics.py` | FAISS vector index — add, search, save, load (IndexFlatIP + IVFFlat) |
| 05 | `05_chromadb_basics.py` | ChromaDB — persistent vector store with metadata filtering |
| 06 | `06_basic_rag_pipeline.py` | Full pipeline: load → chunk → embed → retrieve → generate |
| 07 | `07_metadata_filtering.py` | Filter by company/section/plan before vector search |
| 08 | `08_hybrid_search.py` | BM25 (keyword) + semantic (vector) fused with RRF |
| 09 | `09_reranking.py` | Bi-encoder retrieval → cross-encoder reranking |
| 10 | `10_rag_evaluation.py` | Precision@k, Recall@k, MRR, Faithfulness, Answer Relevance |
| 11 | `11_rag_with_ollama.py` | Full RAG system with Ollama — interactive Q&A |
| — | `project/document_qa/` | CLI Q&A app with multi-turn history and RAG |

---

## Key Concepts to Remember

- **RAG = retrieve → augment prompt → generate**: LLM never guesses, always cites from retrieved chunks
- **Chunk size matters**: too small = missing context; too large = retrieval noise — 512 tokens is a good start
- **Overlap between chunks**: 50–100 token overlap prevents answer from falling at chunk boundary
- **Embedding model must match at query time**: embed docs with `all-MiniLM-L6-v2`, query must use same model
- **FAISS = in-memory**: fast, no persistence by default — use `write_index` / `read_index`
- **ChromaDB = persistent**: survives restarts, supports metadata filters — use for production
- **Cosine similarity for semantic search**: normalize vectors first, then dot product = cosine similarity
- **BM25 = keyword matching**: great for exact terms (product names, IDs), bad for paraphrases
- **Hybrid search**: BM25 + semantic fused with RRF — always better than either alone
- **Reranking**: bi-encoder is fast (retrieval), cross-encoder is accurate (reranking) — use both in sequence
- **MIN_CONFIDENCE threshold**: filter out low-score chunks before sending to LLM — prevents irrelevant answers
- **Agentic RAG**: RAG as a tool in an agent loop — agent retrieves only when needed, not on every query

---

## What NOT to Do

- Don't embed at query time with a different model than index time — vectors are incompatible
- Don't use one giant chunk per document — too much noise for retrieval
- Don't skip chunking overlap — answers at chunk boundaries get split and lost
- Don't use cosine without normalizing — raw dot product ≠ cosine similarity
- Don't add all documents to ChromaDB in a loop — use `col.add(documents=[...])` in batch
- Don't retrieve top-1 only — always retrieve top-3 to top-5, reranker picks the best
- Don't evaluate RAG only on answer quality — measure retrieval (precision/recall) separately
- Don't force RAG on every query — use confidence threshold to fall back gracefully

---

## Quick Start

```bash
pip install chromadb sentence-transformers openai rank-bm25

# Most examples work without Ollama
python 01_document_loading.py
python 02_chunking.py
python 04_faiss_basics.py
python 05_chromadb_basics.py
python 10_rag_evaluation.py

# After ollama pull llama3.2:3b && ollama serve
python 06_basic_rag_pipeline.py
python 11_rag_with_ollama.py

cd project/document_qa/
python app.py --docs data/ --query "What is the pricing?"
```

---

## Frameworks

| Framework | Purpose |
|-----------|---------|
| `sentence-transformers` | Embed text to dense vectors — all-MiniLM-L6-v2 (~80 MB) |
| `chromadb` | Persistent vector store with metadata filters |
| `faiss-cpu` | Fast in-memory vector search (IndexFlatIP, IVFFlat) |
| `rank-bm25` | BM25 keyword scoring for hybrid search |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking for precision |
