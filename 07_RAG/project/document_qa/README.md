# Document Q&A with RAG

A production-style Q&A system that answers questions from your own documents using a local LLM.

## What it does

- Ingests any `.txt` or `.md` documents
- Paragraph-aware chunking with overlap
- ChromaDB vector store (persisted to disk)
- Retrieves top-k relevant chunks per query
- Answers with source citations via Ollama (local, free)
- Multi-turn conversation history (last 3 turns)

## Setup

```bash
pip install chromadb sentence-transformers openai
ollama pull llama3.2:3b
ollama serve
```

## Run

```bash
# Interactive mode (uses sample TechCorp KB by default)
python app.py

# With your own documents
python app.py --docs my_doc.txt another_doc.md

# One-shot query (non-interactive)
python app.py --query "What is the pricing?" --docs my_doc.txt
```

## Example session

```
Loading 1 document(s)...
  Indexed 12 chunks
  LLM: Ollama (llama3.2:3b)

Document Q&A — Interactive Mode
Documents loaded: techcorp_knowledge_base.txt
================================================

You: What are the pricing plans?
Assistant: TechCorp offers three plans [techcorp_knowledge_base.txt]:
- Starter: $99/month, up to 5 users, CPU-only
- Professional: $499/month, up to 25 users, GPU training
- Enterprise: custom pricing, unlimited users

You: Is there a free trial?
Assistant: Yes, there is a 14-day free trial with full Professional features and no credit card required [techcorp_knowledge_base.txt].
```

## Architecture

```
Your Document(s)
      │
      ▼
  Paragraph Chunking (600 chars, 100 overlap)
      │
      ▼
  Embedding (all-MiniLM-L6-v2, ~80 MB)
      │
      ▼
  ChromaDB (persisted to /tmp/document_qa_system)
      │
   [Query]
      │
      ▼
  Vector Retrieval (top-4 chunks, cosine similarity)
      │
      ▼
  Context → Ollama llama3.2:3b (local, free)
      │
      ▼
  Answer with citations
```

## Key Config (in app.py)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CHUNK_SIZE` | 600 | Larger = more context per chunk, slower retrieval |
| `OVERLAP` | 100 | Prevents answers being cut at chunk boundaries |
| `TOP_K` | 4 | More chunks = more context, but can confuse LLM |
| `MIN_CONFIDENCE` | 0.30 | Below this score, chunk is ignored |
| `LLM_MODEL` | llama3.2:3b | Swap to llama3.1:8b for better quality |
