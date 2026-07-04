# Author: Mohan Raju Amuri
"""
document_qa/app.py — Production Document Q&A System with RAG

A complete, runnable Q&A system over your own documents.
Supports multiple files, multi-turn conversation, source citations.

Usage:
    python app.py                          # interactive mode (default docs)
    python app.py --docs path/to/file.txt  # custom document
    python app.py --docs *.txt             # multiple documents
    python app.py --query "What is pricing?" --docs ...  # one-shot query

Requirements:
    pip install chromadb sentence-transformers openai
    ollama pull llama3.2:3b  (free local LLM)
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# Install Ollama: https://ollama.ai
# Pull model:    ollama pull llama3.2:3b
# Start server:  ollama serve
# ============================================================

import sys
import argparse
import re
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ── Config (tune these for your use case) ────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
LLM_MODEL      = "llama3.2:3b"
CHUNK_SIZE     = 600
OVERLAP        = 100
TOP_K          = 4
MIN_CONFIDENCE = 0.30
PERSIST_DIR    = "/tmp/document_qa_system"
COLLECTION     = "document_qa"

SYSTEM_PROMPT = """You are a helpful document assistant. Answer questions based ONLY on the provided context.
Rules:
- Cite sources with [filename] after each claim
- If the answer is not in the context, say "I don't have that information in the provided documents."
- Be concise and accurate
- Do not add information beyond what the context says"""


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_document(filepath: str) -> list[dict]:
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    source = Path(filepath).name
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 20]

    chunks, current, idx = [], "", 0
    for para in paragraphs:
        if current and len(current) + len(para) > CHUNK_SIZE:
            chunks.append({"id": f"{Path(filepath).stem}_{idx}", "text": current,
                           "source": source, "i": idx})
            idx += 1
            current = current[-OVERLAP:] + "\n\n" + para if len(current) > OVERLAP else para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append({"id": f"{Path(filepath).stem}_{idx}", "text": current,
                       "source": source, "i": idx})
    return chunks


# ── Vector Store ──────────────────────────────────────────────────────────────
def build_store(doc_paths: list[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = client.create_collection(COLLECTION, embedding_function=ef,
                                    metadata={"hnsw:space": "cosine"})

    all_chunks = []
    for path in doc_paths:
        if Path(path).exists():
            all_chunks.extend(chunk_document(path))
        else:
            print(f"  Warning: file not found: {path}")

    if not all_chunks:
        raise ValueError("No documents loaded. Check your file paths.")

    col.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "chunk_i": c["i"]} for c in all_chunks],
        ids=[c["id"] for c in all_chunks],
    )
    return col


def retrieve(col: chromadb.Collection, query: str) -> list[dict]:
    results = col.query(query_texts=[query], n_results=TOP_K)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        sim = 1.0 - dist / 2.0
        if sim >= MIN_CONFIDENCE:
            hits.append({"text": doc, "source": meta["source"], "score": sim})
    return hits


# ── LLM ───────────────────────────────────────────────────────────────────────
def build_llm():
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        return client
    except Exception:
        return None


def generate(llm, messages: list[dict]) -> str:
    if llm is None:
        return "[Ollama not running — install at https://ollama.ai and run `ollama pull llama3.2:3b`]"
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=0.1, max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


# ── DocumentQA ────────────────────────────────────────────────────────────────
class DocumentQA:
    def __init__(self, doc_paths: list[str]):
        print(f"\nLoading {len(doc_paths)} document(s)...")
        self.col = build_store(doc_paths)
        print(f"  Indexed {self.col.count()} chunks")
        self.llm = build_llm()
        status = f"Ollama ({LLM_MODEL})" if self.llm else "Ollama NOT running"
        print(f"  LLM: {status}")
        self.history: list[dict] = []  # multi-turn conversation history

    def ask(self, question: str) -> dict:
        # Retrieve context
        chunks = retrieve(self.col, question)

        if not chunks:
            answer = "I don't have relevant information in the provided documents to answer that."
            return {"answer": answer, "sources": [], "chunks": []}

        # Build context
        context = "\n\n---\n\n".join(
            f"[{c['source']}] (relevance {c['score']:.0%})\n{c['text']}"
            for c in chunks
        )

        # Maintain conversation history (last 3 turns for context)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-6:]  # last 3 Q&A pairs
        messages.append({
            "role": "user",
            "content": f"Context documents:\n{context}\n\nQuestion: {question}"
        })

        answer = generate(self.llm, messages)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": list(set(c["source"] for c in chunks)),
            "chunks": chunks,
        }

    def run_interactive(self):
        sources_loaded = set()
        results = self.col.get(limit=1000)
        for meta in results.get("metadatas", []):
            sources_loaded.add(meta.get("source", "unknown"))

        print("\n" + "=" * 60)
        print("Document Q&A — Interactive Mode")
        print(f"Documents loaded: {', '.join(sorted(sources_loaded))}")
        print("Commands: 'quit' to exit | 'clear' to reset history | 'sources' to list docs")
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
            if question.lower() == "clear":
                self.history = []
                print("  Conversation history cleared.")
                continue
            if question.lower() == "sources":
                print(f"  Loaded: {', '.join(sorted(sources_loaded))}")
                continue

            result = self.ask(question)
            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                print(f"\n  [Sources: {', '.join(result['sources'])} | {len(result['chunks'])} chunks]")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Document Q&A with RAG + Ollama")
    parser.add_argument("--docs", nargs="+", help="Document file paths")
    parser.add_argument("--query", type=str, help="One-shot query (non-interactive)")
    args = parser.parse_args()

    # Default to the module's sample KB if no docs specified
    if not args.docs:
        default_kb = Path(__file__).parent.parent.parent / "sample_docs" / "techcorp_knowledge_base.txt"
        doc_paths = [str(default_kb)]
        print(f"No --docs specified, using default: {default_kb.name}")
    else:
        doc_paths = args.docs

    qa = DocumentQA(doc_paths)

    if args.query:
        result = qa.ask(args.query)
        print(f"\nQ: {args.query}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            print(f"\n[Sources: {', '.join(result['sources'])}]")
    else:
        qa.run_interactive()


if __name__ == "__main__":
    main()
