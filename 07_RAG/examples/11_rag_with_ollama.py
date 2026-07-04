# Author: Mohan Raju Amuri
"""
11_rag_with_ollama.py — Production-style RAG system powered by a local LLM (free, no API key)

What to remember:
- This is the capstone example: everything from examples 01-10 combined
- Local LLM (Ollama) = no API costs, no data leaving your machine, no rate limits
- Production RAG adds: source citations, conversation history, confidence thresholds
- The retrieval quality determines 80% of RAG answer quality — garbage in, garbage out

What NOT to do:
- Don't send more than 3-5 chunks to the LLM — quality degrades with too much context
- Don't skip source citations — users need to verify AI-generated answers
- Don't ignore low-confidence retrievals — better to say "I don't know" than hallucinate

This example builds a full Document QA system with:
  ✓ Multi-document ingestion
  ✓ Paragraph-aware chunking with overlap
  ✓ ChromaDB persistent vector store
  ✓ Ollama local LLM (llama3.2:3b)
  ✓ Source citations in every answer
  ✓ Confidence threshold to handle out-of-scope queries
  ✓ Interactive Q&A loop

Interview one-liner:
  "Full RAG stack: ChromaDB retrieval + Ollama generation — grounded, cited, and free to run."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# Install Ollama: https://ollama.ai
# Pull model:    ollama pull llama3.2:3b
# Start server:  ollama serve   (or it auto-starts on macOS)
#
# Estimated cost: $0.00 — runs entirely on your machine
# ============================================================

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
import re

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"
LLM_MODEL         = "llama3.2:3b"
CHUNK_SIZE        = 500       # characters per chunk
OVERLAP           = 80        # overlap between chunks
TOP_K             = 4         # chunks retrieved per query
MIN_CONFIDENCE    = 0.35      # below this: "I don't have that information"
COLLECTION_NAME   = "rag_ollama_demo"


# ── Document Ingestion ────────────────────────────────────────────────────────
def load_and_chunk_file(filepath: str) -> list[dict]:
    """Paragraph-aware chunking with overlap."""
    text = Path(filepath).read_text(encoding="utf-8")
    source = Path(filepath).name

    # Split on paragraph boundaries first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_SIZE:
            chunks.append({
                "id":     f"{Path(filepath).stem}_chunk_{chunk_idx}",
                "text":   current.strip(),
                "source": source,
                "index":  chunk_idx,
            })
            chunk_idx += 1
            # Keep overlap: last part of current chunk
            tail = current[-OVERLAP:].strip() if len(current) > OVERLAP else current
            current = tail + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append({
            "id":     f"{Path(filepath).stem}_chunk_{chunk_idx}",
            "text":   current.strip(),
            "source": source,
            "index":  chunk_idx,
        })

    return chunks


# ── Vector Store ──────────────────────────────────────────────────────────────
class VectorStore:
    def __init__(self, persist_dir: str = "/tmp/rag_ollama_demo"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        # Fresh collection each run for demo
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.col = self.client.create_collection(
            COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, file_paths: list[str]) -> int:
        """Ingest one or more files into the vector store."""
        all_chunks = []
        for path in file_paths:
            all_chunks.extend(load_and_chunk_file(path))

        self.col.add(
            documents=[c["text"]   for c in all_chunks],
            metadatas=[{"source": c["source"], "chunk_index": c["index"]} for c in all_chunks],
            ids=[c["id"] for c in all_chunks],
        )
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve top-k chunks and convert ChromaDB distance to similarity score."""
        results = self.col.query(query_texts=[query], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist / 2.0  # convert cosine dist → similarity
            if similarity >= MIN_CONFIDENCE:
                hits.append({
                    "text":    doc,
                    "source":  meta["source"],
                    "score":   similarity,
                    "chunk_i": meta["chunk_index"],
                })
        return hits


# ── LLM via Ollama ────────────────────────────────────────────────────────────
class OllamaRAG:
    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            # Probe: quick test to see if Ollama is running
            self.client.models.list()
            self.available = True
            print(f"  Ollama connected — using model: {LLM_MODEL}")
        except Exception:
            self.available = False
            print(f"  Ollama not running — answers will show the prompt instead")
            print(f"  To enable: install Ollama, run `ollama pull {LLM_MODEL}`, then `ollama serve`")

    def answer(self, question: str, context_chunks: list[dict]) -> str:
        if not context_chunks:
            return "I don't have relevant information in my knowledge base to answer that question."

        # Build context block with sources
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[{i}] Source: {chunk['source']} (relevance: {chunk['score']:.0%})\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        system = (
            "You are a knowledgeable assistant for TechCorp AI Platform. "
            "Answer questions using ONLY the provided context sections. "
            "Cite sources using [1], [2], etc. after each claim. "
            "If the answer is not in the context, say 'I don't have that information in the knowledge base.' "
            "Keep answers concise and factual."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        if not self.available:
            return f"[PROMPT PREVIEW — Ollama not running]\nSYSTEM: {system[:120]}...\nUSER context length: {len(context)} chars"

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
                max_tokens=400,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error generating answer: {e}]"


# ── Full RAG System ───────────────────────────────────────────────────────────
class RAGSystem:
    def __init__(self, doc_paths: list[str]):
        print("\nInitializing RAG system...")

        # 1. Build vector store
        print("  Setting up vector store...")
        self.store = VectorStore()
        n = self.store.add_documents(doc_paths)
        print(f"  Indexed {n} chunks from {len(doc_paths)} document(s)")

        # 2. Connect LLM
        self.llm = OllamaRAG()

    def ask(self, question: str) -> dict:
        """Full RAG: retrieve → generate → return with citations."""
        chunks = self.store.retrieve(question)
        answer = self.llm.answer(question, chunks)

        return {
            "question": question,
            "answer":   answer,
            "sources":  list(set(c["source"] for c in chunks)),
            "chunks":   chunks,
            "retrieved": len(chunks),
        }

    def interactive(self):
        """Simple REPL for interactive Q&A."""
        print("\n" + "=" * 60)
        print("TechCorp Knowledge Base — Interactive Q&A")
        print("Type your question (or 'quit' to exit)")
        print("=" * 60)

        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            result = self.ask(question)
            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                print(f"\n[Sources: {', '.join(result['sources'])} — {result['retrieved']} chunk(s) retrieved]")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Step 11: Full RAG with Ollama (Local LLM)")
    print("=" * 60)

    kb_path = str(Path(__file__).parent.parent / "sample_docs" / "techcorp_knowledge_base.txt")
    rag = RAGSystem(doc_paths=[kb_path])

    # Demo Q&A
    demo_questions = [
        "What are the pricing plans and their costs?",
        "How does TechCorp ensure data security?",
        "What machine learning frameworks are supported?",
        "Is there a free trial and what does it include?",
        "What deployment options are available?",
        "How do I get started with the Python SDK?",
        "What is the weather like today?",  # out-of-scope question
    ]

    print("\n" + "=" * 60)
    print("Automated Q&A Demo")
    print("=" * 60)

    for q in demo_questions:
        result = rag.ask(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        src = f"  [{', '.join(result['sources'])} | {result['retrieved']} chunk(s)]"
        print(src)

    # Offer interactive mode
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Full stack: paragraph chunking + ChromaDB + Ollama + citations")
    print("  - Local LLM: llama3.2:3b — good quality, fast, free, private")
    print("  - MIN_CONFIDENCE threshold catches out-of-scope queries gracefully")
    print("  - Source citations on every answer — users can verify")
    print("  - To run interactively: call rag.interactive() after setup")
    print("=" * 60)

    print("\nTo start interactive mode, run:")
    print("  python 11_rag_with_ollama.py")
    print("  (then type 'interactive' at the prompt, or edit __main__ to call rag.interactive())")
