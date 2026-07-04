# Author: Mohan Raju Amuri
"""
01_document_loading.py — How RAG starts: loading raw documents

What to remember:
- RAG needs raw text before anything else: load → chunk → embed → retrieve → generate
- Documents can come from .txt, .pdf, .md, URLs, databases — loading is step 0
- Always track metadata (source, page, date) alongside content — you need it for citations

What NOT to do:
- Don't load one giant string and try to embed it whole — models have token limits
- Don't ignore encoding issues — always use encoding='utf-8', errors='replace'
- Don't lose the source path — you need it to cite where the answer came from

Interview one-liner:
  "Document loading is the ingestion step — parse raw files into text chunks with metadata."
"""

import os
import json
from pathlib import Path

# ── What is a Document in RAG? ────────────────────────────────────────────────
# A Document = text content + metadata dict
# Metadata tells you WHERE the text came from so you can cite it
class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"Document(chars={len(self.content)}, preview='{preview}...', metadata={self.metadata})"


# ── Loader 1: Plain Text File ─────────────────────────────────────────────────
def load_text_file(filepath: str) -> Document:
    """Load a .txt file into a Document with file metadata."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return Document(
        content=content,
        metadata={
            "source": str(path),
            "filename": path.name,
            "filetype": "text",
            "size_bytes": path.stat().st_size,
        }
    )


# ── Loader 2: Multiple Files from a Directory ─────────────────────────────────
def load_directory(dirpath: str, extensions: list = None) -> list[Document]:
    """Load all matching files from a directory."""
    if extensions is None:
        extensions = [".txt", ".md"]

    docs = []
    for path in Path(dirpath).rglob("*"):
        if path.suffix in extensions and path.is_file():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            docs.append(Document(
                content=content,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "filetype": path.suffix.lstrip("."),
                }
            ))
    return docs


# ── Loader 3: In-Memory Strings (useful for tests / web scraped content) ──────
def load_from_strings(texts: list[str], sources: list[str] = None) -> list[Document]:
    """Wrap raw strings as Documents — useful when you've already extracted text."""
    if sources is None:
        sources = [f"string_{i}" for i in range(len(texts))]
    return [
        Document(content=text, metadata={"source": src})
        for text, src in zip(texts, sources)
    ]


# ── Loader 4: JSON / JSONL ────────────────────────────────────────────────────
def load_json_file(filepath: str, content_field: str = "text") -> list[Document]:
    """
    Load a JSON array or JSONL file where each record has a text field.
    Common format for FAQ datasets, product catalogs, etc.
    """
    path = Path(filepath)
    docs = []

    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)
            if isinstance(records, dict):
                records = [records]

    for i, record in enumerate(records):
        content = record.get(content_field, str(record))
        metadata = {k: v for k, v in record.items() if k != content_field}
        metadata["source"] = str(path)
        metadata["record_index"] = i
        docs.append(Document(content=content, metadata=metadata))

    return docs


# ── Document Stats Helper ─────────────────────────────────────────────────────
def document_stats(docs: list[Document]) -> dict:
    """Quick summary of a document collection."""
    total_chars = sum(len(d.content) for d in docs)
    # Rough token estimate: ~4 chars per token for English
    return {
        "num_documents": len(docs),
        "total_characters": total_chars,
        "avg_characters": total_chars // len(docs) if docs else 0,
        "estimated_tokens": total_chars // 4,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Step 1: Document Loading")
    print("=" * 60)

    # 1. Load the sample knowledge base
    kb_path = Path(__file__).parent.parent / "sample_docs" / "techcorp_knowledge_base.txt"

    print("\n[1] Loading single text file")
    doc = load_text_file(str(kb_path))
    print(f"  {doc}")
    print(f"  Content length: {len(doc.content)} characters")
    print(f"  Metadata keys: {list(doc.metadata.keys())}")

    # 2. Show a content preview
    print("\n[2] Content preview (first 200 chars)")
    print(f"  {doc.content[:200].strip()!r}")

    # 3. Load from strings (simulating web scraping or API responses)
    print("\n[3] Loading from raw strings")
    raw_texts = [
        "TechCorp AI Platform supports Python and R workflows.",
        "Enterprise plan includes 24/7 support with dedicated engineer.",
        "All data is encrypted at rest using AES-256.",
    ]
    string_docs = load_from_strings(
        texts=raw_texts,
        sources=["techcorp_overview", "techcorp_support", "techcorp_security"]
    )
    for d in string_docs:
        print(f"  {d}")

    # 4. Simulate loading a fake FAQ JSON (in-memory)
    print("\n[4] FAQ JSON structure (how real FAQ KBs are loaded)")
    import tempfile, json
    faq_data = [
        {"text": "What is TechCorp AI Platform?", "answer": "A cloud-native ML platform.", "category": "product"},
        {"text": "How much does the starter plan cost?", "answer": "$99/month.", "category": "pricing"},
        {"text": "Is there a free trial?", "answer": "Yes, 14-day free trial.", "category": "pricing"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(faq_data, f)
        tmp_path = f.name

    faq_docs = load_json_file(tmp_path, content_field="text")
    for d in faq_docs:
        print(f"  Source: {d.metadata['source'].split('/')[-1]}  |  Content: {d.content!r}")
    os.unlink(tmp_path)

    # 5. Stats
    all_docs = [doc] + string_docs + faq_docs
    stats = document_stats(all_docs)
    print("\n[5] Collection stats")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Document = content + metadata (always keep the source!)")
    print("  - Support multiple input formats from day one")
    print("  - Metadata enables citations: 'According to techcorp_kb.txt...'")
    print("  - Next step: chunk these docs so they fit in embedding windows")
    print("=" * 60)
