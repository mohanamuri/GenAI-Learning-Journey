# sentence-transformers

**What it is:** Library for computing dense sentence/paragraph embeddings using pre-trained transformer models. Produces vectors where semantically similar text is close in space.

**Before sentence-transformers:**
- Word2Vec/GloVe: word-level only, average vectors for sentences (lossy)
- USE (Universal Sentence Encoder): good but Google-only
- Raw BERT: you had to manually extract [CLS] token, normalize — 20+ lines of code

**Why we picked it here (briefly):**
Introduced in example 15 (text_similarity) as the "right answer" for semantic similarity. Enough to see the API and understand the gap vs TF-IDF — deep dive comes in Module 07 (RAG).

**When to use:**
- Semantic search ("find documents similar to this query")
- FAQ matching, duplicate detection
- Clustering documents by meaning
- Anything where synonym-awareness matters

**When NOT to use:**
- Simple keyword matching — TF-IDF is faster
- When you need the model to generate text — use an LLM
- Latency-critical edge (model is ~80MB+)

**vs TF-IDF cosine:**
TF-IDF: "car" ≠ "automobile" (different words). sentence-transformers: "car" ≈ "automobile" (same meaning).

**Key API:**
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["sentence one", "sentence two"])
similarity = util.cos_sim(embeddings[0], embeddings[1])
```

**Popular models:**
- `all-MiniLM-L6-v2` — fast, 80MB, good quality (default choice)
- `all-mpnet-base-v2` — slower, more accurate
- `multi-qa-MiniLM-L6-cos-v1` — tuned for Q&A / retrieval

**Coming up:** Used heavily in Module 07 (RAG) for document chunking + retrieval.

**Install:** `pip install sentence-transformers`

**Introduced in:** Module 04 — NLP (example 15)
