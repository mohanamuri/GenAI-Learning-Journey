# Author: Mohan Raju Amuri
"""
06_basic_rag_pipeline.py — End-to-end RAG: load → chunk → index → retrieve → generate

What to remember:
- RAG = Retrieval-Augmented Generation: give the LLM only the relevant context
- Pipeline: Document → Chunks → Embeddings → Vector Store → Retrieve → Prompt → LLM → Answer
- The LLM never sees the full document — only the top-k retrieved chunks
- This prevents hallucination by grounding the LLM in your actual data

What NOT to do:
- Don't stuff all documents into the prompt — LLMs degrade with too much context
- Don't skip the "no relevant context found" path — always handle low-confidence retrievals
- Don't use RAG for tasks where the LLM's own knowledge is sufficient (e.g. math, coding)

Key RAG components:
  Retriever    — vector similarity search over embedded chunks
  Prompt       — system message + retrieved context + user question
  Generator    — LLM that reads the context and answers

Interview one-liner:
  "RAG grounds LLM answers in your documents — retrieve relevant chunks, inject as context, then generate."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# This example uses Ollama for free, local LLM inference.
# No API key, no cost. One-time setup:
#
#   1. Install Ollama: https://ollama.ai
#   2. Pull a model:   ollama pull llama3.2:3b
#   3. Start server:   ollama serve   (or it auto-starts)
#
# Falls back to a mock generator if Ollama is not running.
# ============================================================

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
import re

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama3.2:3b"
TOP_K       = 3        # retrieve 3 chunks per query
CHUNK_SIZE  = 400      # characters per chunk
OVERLAP     = 60       # overlap between adjacent chunks
MIN_SCORE   = 0.3      # discard chunks below this cosine similarity


# ── Step 1: Load & Chunk the Knowledge Base ──────────────────────────────────
def load_and_chunk(filepath: str) -> list[dict]:
    """Load a text file and split into overlapping character chunks."""
    text = Path(filepath).read_text(encoding="utf-8")
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip tiny trailing fragments
            chunks.append({
                "id": f"{Path(filepath).stem}_chunk_{idx}",
                "text": chunk,
                "source": Path(filepath).name,
                "chunk_index": idx,
            })
            idx += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks


# ── Step 2: Build Vector Store ────────────────────────────────────────────────
def build_vector_store(chunks: list[dict], collection_name: str = "rag_demo") -> chromadb.Collection:
    """Embed chunks and store in ChromaDB."""
    client = chromadb.EphemeralClient()  # in-memory for this demo
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    col = client.create_collection(collection_name, embedding_function=ef,
                                    metadata={"hnsw:space": "cosine"})
    col.add(
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
        ids=[c["id"] for c in chunks],
    )
    return col


# ── Step 3: Retrieve Relevant Chunks ─────────────────────────────────────────
def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Similarity search — returns list of {text, source, score}."""
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score [0, 1]: sim = 1 - dist/2
        similarity = 1.0 - dist / 2.0
        if similarity >= MIN_SCORE:
            hits.append({"text": doc, "source": meta["source"], "score": similarity})
    return hits


# ── Step 4: Build RAG Prompt ──────────────────────────────────────────────────
def build_prompt(query: str, context_chunks: list[dict]) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the LLM."""
    if not context_chunks:
        context_text = "[No relevant context found in the knowledge base.]"
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(f"[Source: {chunk['source']} | Relevance: {chunk['score']:.2f}]\n{chunk['text']}")
        context_text = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a helpful assistant for TechCorp AI Platform. "
        "Answer questions using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't have that information.' "
        "Be concise and cite the source section when possible."
    )
    user = f"Context:\n{context_text}\n\nQuestion: {query}"
    return system, user


# ── Step 5: Generate Answer via Ollama ───────────────────────────────────────
def generate_answer(system: str, user: str) -> str:
    """Call Ollama (local LLM). Falls back to mock if Ollama not running."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.1,  # low temperature for factual answers
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Ollama not running — show what would be sent to the LLM
        return (
            f"[Ollama not running — install at https://ollama.ai]\n"
            f"Would send to {LLM_MODEL}:\n"
            f"SYSTEM: {system[:100]}...\n"
            f"USER:   {user[:200]}..."
        )


# ── Full RAG Pipeline ─────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, doc_path: str):
        print(f"  Loading document: {doc_path}")
        chunks = load_and_chunk(doc_path)
        print(f"  Created {len(chunks)} chunks")

        print(f"  Building vector store ({EMBED_MODEL}) ...")
        self.collection = build_vector_store(chunks)
        print(f"  Vector store ready with {self.collection.count()} vectors")

    def ask(self, question: str, verbose: bool = False) -> dict:
        """Full RAG: retrieve → build prompt → generate → return."""
        # Retrieve
        hits = retrieve(self.collection, question)

        # Build prompt
        system, user = build_prompt(question, hits)

        # Generate
        answer = generate_answer(system, user)

        return {
            "question": question,
            "answer": answer,
            "sources": [h["source"] for h in hits],
            "context_chunks": hits,
            "num_retrieved": len(hits),
        }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Step 6: Full RAG Pipeline")
    print("=" * 60)

    kb_path = Path(__file__).parent.parent / "sample_docs" / "techcorp_knowledge_base.txt"

    print("\nInitializing RAG pipeline...")
    rag = RAGPipeline(str(kb_path))

    questions = [
        "How much does the Professional plan cost per month?",
        "What security certifications does TechCorp have?",
        "Can I run models on edge devices?",
        "Is there a free trial available?",
        "What is the uptime guarantee?",
        "What machine learning frameworks are supported?",
    ]

    print("\n" + "=" * 60)
    print("Q&A Session")
    print("=" * 60)

    for q in questions:
        result = rag.ask(q, verbose=False)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"   [Retrieved {result['num_retrieved']} chunks from: {', '.join(set(result['sources']))}]")

    # Show the retrieved context for one question
    print("\n" + "=" * 60)
    print("Detailed view — what the LLM actually sees")
    print("=" * 60)
    detail_q = "What support options are available?"
    result = rag.ask(detail_q)
    print(f"\nQuery: {detail_q}")
    print(f"\nRetrieved chunks (sent as context to LLM):")
    for i, chunk in enumerate(result["context_chunks"], 1):
        print(f"  Chunk {i} (score={chunk['score']:.3f}):")
        print(f"    {chunk['text'][:120]}")
    print(f"\nAnswer: {result['answer']}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Full RAG = load → chunk → embed → store → retrieve → prompt → generate")
    print("  - LLM only sees top-k chunks, not the whole document")
    print("  - Low temperature (0.1) for factual RAG answers")
    print("  - Always handle the 'no relevant chunks' case gracefully")
    print("  - MIN_SCORE threshold filters out irrelevant retrievals")
    print("=" * 60)
