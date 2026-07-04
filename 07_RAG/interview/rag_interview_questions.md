# RAG Interview Questions & Answers

## Core Concepts

**Q: What is RAG and why do we need it?**
A: RAG (Retrieval-Augmented Generation) grounds LLM answers in external documents.
Without RAG, LLMs hallucinate, have stale knowledge (training cutoff), and can't answer about private data.
With RAG: retrieve relevant chunks → inject as context → LLM generates a cited, grounded answer.

**Q: What are the steps in a RAG pipeline?**
A: Load documents → Chunk → Embed chunks → Store in vector DB → At query time: embed query → retrieve top-k chunks → build prompt with context → LLM generates answer.

**Q: What is a chunk and why does chunk size matter?**
A: A chunk is a piece of a document fed to the embedding model.
Too small (< 100 chars): no context, hallucinations spike.
Too large (> 2000 chars): noisy retrieval, model buries the answer.
Sweet spot: 300-600 chars with 50-100 char overlap.

**Q: What is chunk overlap and why is it needed?**
A: Overlap is repeated content at the boundary of adjacent chunks.
Without overlap, an answer that spans a chunk boundary gets split — neither chunk has the full context.
Rule: overlap = 10-20% of chunk size.

---

## Retrieval

**Q: How does vector similarity search work?**
A: Text → embedding model → float vector (e.g. 384 dims).
At index time: embed all chunks, store vectors.
At query time: embed query, compute cosine similarity with all stored vectors, return top-k.

**Q: What is the difference between bi-encoder and cross-encoder?**
A: Bi-encoder: encodes query and document SEPARATELY. Fast, O(N) build-once, used for retrieval.
Cross-encoder: reads query+document TOGETHER. Much more accurate but O(k) per query. Used for reranking.
Best practice: bi-encoder retrieves top-20, cross-encoder reranks to top-3.

**Q: What is Reciprocal Rank Fusion (RRF)?**
A: A technique to merge multiple ranked lists (e.g., semantic + BM25) without score normalization.
Score = sum(1 / (k + rank_i)) where k=60 is a standard constant.
Works because it only cares about rank, not the raw score scale of each system.

**Q: When does hybrid search outperform pure semantic search?**
A: When the query contains exact keywords (product IDs, model names, codes).
Semantic search excels at paraphrasing; BM25 excels at exact term matching.
Hybrid captures both — typically +5-15% NDCG vs either alone.

**Q: What is FAISS and what index types should you know?**
A: FAISS is Meta's in-memory vector search library.
IndexFlatL2/IP: exact, use for < 50k vectors.
IndexIVFFlat: approximate with inverted file index, use for millions.
IndexHNSW: graph-based, best recall/speed tradeoff at any scale.

---

## Vector Databases

**Q: FAISS vs ChromaDB — when do you use each?**
A: FAISS: maximum speed, no metadata storage, need manual id→text mapping. Use at scale.
ChromaDB: batteries-included (text+vectors+metadata), metadata filtering, simpler API. Use for prototyping and apps that need filtering.

**Q: What is metadata filtering in RAG and why does it matter?**
A: Pre-filtering the vector search to a subset of documents matching metadata conditions.
Example: where={"company": "techcorp"} searches only TechCorp docs.
Prevents cross-tenant contamination in multi-tenant RAG, improves precision.

---

## Generation

**Q: What should a RAG system prompt contain?**
A: (1) Role: "You are a helpful assistant for [product]."
(2) Constraint: "Answer ONLY using the provided context."
(3) Fallback: "If not in context, say 'I don't have that information.'"
(4) Format: citation instructions like "Cite sources with [source_name]."

**Q: Why use low temperature for RAG?**
A: Temperature 0.0-0.1 keeps the LLM closer to the retrieved context.
Higher temperature introduces creativity, which in RAG = hallucination (inventing facts not in context).

**Q: What is faithfulness in RAG evaluation?**
A: Faithfulness measures whether the generated answer only uses information from the retrieved context.
A faithful answer = grounded. An unfaithful answer = hallucinated facts the LLM invented.

---

## Evaluation

**Q: How do you evaluate a RAG pipeline?**
A: Evaluate retrieval and generation separately.
Retrieval: Precision@k (retrieved chunks that are relevant), Recall@k (relevant chunks retrieved), MRR.
Generation: Faithfulness (answer grounded in context?), Answer Relevance, Context Precision.
Build a labeled Q&A test set of 50-200 questions with expected source chunks.

**Q: What is the most common failure mode in RAG?**
A: The two main failures:
(1) Retrieval failure: wrong chunks retrieved → LLM lacks the answer → hallucinates.
(2) Generation failure: correct chunks retrieved but LLM ignores them → hallucinates anyway.
Fix (1) by improving chunking, embedding model, or adding metadata filters.
Fix (2) by lowering temperature, tightening the system prompt, reducing context size.

---

## System Design

**Q: How would you build RAG for a multi-tenant SaaS product?**
A: Store tenant_id in chunk metadata. At query time, filter by tenant_id before vector search.
This ensures users only see their own data. Never rely on prompt instructions alone for tenant isolation — always enforce at the retrieval layer.

**Q: How do you handle documents that are too long for one embedding?**
A: Chunking. Standard approach: paragraph-aware chunking with 300-600 char chunks and 10-20% overlap.
For very long documents: use a sliding window, or summarize sections before embedding.

**Q: What is reranking and when should you add it?**
A: Reranking uses a cross-encoder to re-score retrieved candidates before sending to the LLM.
Add it when retrieval precision is low (wrong chunks reaching the LLM).
Two-stage: retrieve top-20 with bi-encoder → rerank to top-3 with cross-encoder.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 is the standard lightweight choice.

**Q: What is the difference between RAG and fine-tuning?**
A: RAG: retrieves external knowledge at inference time. No model training needed. Handles dynamic/private data.
Fine-tuning: bakes knowledge into model weights. Expensive, static, good for style/behavior.
Use RAG for factual grounding on changing data; fine-tuning for teaching the model a new skill or tone.
