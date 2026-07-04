# Prerequisites — Module 07 RAG (Retrieval-Augmented Generation)

## Runs Locally — Ollama for Free LLM

Most examples use sentence-transformers, ChromaDB, and FAISS.
Examples 06, 11 and the project use Ollama for free local LLM generation.
No cloud API keys required.

---

## Install

```bash
pip install faiss-cpu chromadb sentence-transformers openai numpy
```

Verify:
```bash
python -c "import faiss, chromadb, sentence_transformers, openai, numpy; print('All good')"
```

> **Apple Silicon (M1/M2/M3)?** faiss-cpu works fine. If you hit issues:
> ```bash
> conda install -c conda-forge faiss-cpu
> ```

---

## Ollama Setup (for examples 06, 11, and project)

```bash
# 1. Install Ollama
#    macOS: brew install ollama  OR  download from https://ollama.ai
#    Linux: curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull model (one-time, ~2 GB)
ollama pull llama3.2:3b

# 3. Start server (auto-starts on macOS after install)
ollama serve
```

Verify:
```bash
curl http://localhost:11434/api/tags  # should list llama3.2:3b
```

---

## What Each Example Does & How to Run

| Example | What it does | Needs | Run |
|---------|-------------|-------|-----|
| `01_document_loading.py` | Load .txt/.json files into Document objects with metadata | stdlib only | `python 01_document_loading.py` |
| `02_chunking.py` | Fixed-size, paragraph, and recursive chunking strategies | stdlib only | `python 02_chunking.py` |
| `03_embeddings_for_rag.py` | Embed chunks, cosine similarity, why semantic beats keyword | sentence-transformers | `python 03_embeddings_for_rag.py` |
| `04_faiss_basics.py` | Build FAISS index, search, save/load, IndexFlatIP vs IVFFlat | faiss-cpu, sentence-transformers | `python 04_faiss_basics.py` |
| `05_chromadb_basics.py` | ChromaDB CRUD, metadata filtering, $and/$or/$in | chromadb, sentence-transformers | `python 05_chromadb_basics.py` |
| `06_basic_rag_pipeline.py` | Full RAG: load → chunk → embed → retrieve → generate | chromadb, sentence-transformers, Ollama | `python 06_basic_rag_pipeline.py` |
| `07_metadata_filtering.py` | Multi-tenant isolation, intent-based routing | chromadb, sentence-transformers | `python 07_metadata_filtering.py` |
| `08_hybrid_search.py` | BM25 + semantic search fused with RRF | sentence-transformers, numpy | `python 08_hybrid_search.py` |
| `09_reranking.py` | Two-stage retrieval: bi-encoder + cross-encoder reranker | sentence-transformers | `python 09_reranking.py` |
| `10_rag_evaluation.py` | Precision@k, Recall, MRR, Faithfulness metrics | sentence-transformers, numpy | `python 10_rag_evaluation.py` |
| `11_rag_with_ollama.py` | Production RAG with citations, confidence threshold, interactive Q&A | chromadb, sentence-transformers, Ollama | `python 11_rag_with_ollama.py` |

---

## Suggested Run Order

**Start here — no downloads beyond pip:**
```bash
python 01_document_loading.py
python 02_chunking.py
```

**After `pip install sentence-transformers` (~80 MB model download on first run):**
```bash
python 03_embeddings_for_rag.py
python 04_faiss_basics.py
python 05_chromadb_basics.py
python 07_metadata_filtering.py
python 08_hybrid_search.py
python 10_rag_evaluation.py
```

**After `ollama pull llama3.2:3b` (~2 GB, one-time):**
```bash
python 06_basic_rag_pipeline.py
python 11_rag_with_ollama.py
```

**Reranker model (~80 MB additional download on first run):**
```bash
python 09_reranking.py
```

**Full project:**
```bash
cd project/document_qa/
python app.py
# interactive Q&A over the sample knowledge base
```

---

## Model Downloads (Automatic on First Run)

| Example | Model | Size | Cached at |
|---------|-------|------|-----------|
| `03` – `11`, project | `all-MiniLM-L6-v2` | ~80 MB | `~/.cache/huggingface` |
| `09_reranking.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80 MB | `~/.cache/huggingface` |
| `06`, `11`, project | `llama3.2:3b` (via Ollama) | ~2 GB | `~/.ollama/models` |

Total first-run download: **~2.2 GB** (mostly the Ollama model).

---

## Check Cache

```bash
du -sh ~/.cache/huggingface   # sentence-transformers models
du -sh ~/.ollama/models        # Ollama models
```

---

## No API Keys — No Cost

Everything runs locally. `all-MiniLM-L6-v2` is reused across examples — downloads once.
The Ollama LLM (llama3.2:3b) runs on your CPU/GPU — no internet after first download.
