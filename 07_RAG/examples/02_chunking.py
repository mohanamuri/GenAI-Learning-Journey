# Author: Mohan Raju Amuri
"""
02_chunking.py — Split documents into pieces the embedding model can handle

What to remember:
- Embedding models have token limits (~512 tokens for most sentence-transformers)
- Chunks need overlap so context is not cut in half at boundaries
- chunk_size and chunk_overlap are the two knobs to tune for your use case

What NOT to do:
- Don't make chunks too small (< 100 chars) — no context, hallucinations spike
- Don't make chunks too large (> 1000 chars) — noisy retrieval, model misses the answer
- Don't chunk mid-sentence — always try to split on paragraph or sentence boundaries

Golden rule:
  chunk_size: 200–500 tokens / 800–2000 chars  |  overlap: 10–20% of chunk_size

Interview one-liner:
  "Chunking splits long documents into retrievable units; overlap prevents context loss at boundaries."
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Chunk(chars={len(self.content)}, source={self.metadata.get('source','?')}, idx={self.metadata.get('chunk_index','?')})"


# ── Strategy 1: Fixed-Size Character Chunking ─────────────────────────────────
def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50,
                     metadata: dict = None) -> list[Chunk]:
    """
    Slide a window of chunk_size over the text with overlap.
    Simple, predictable, but may split mid-sentence.
    Use when: text is uniform (transcripts, logs, numeric data).
    """
    metadata = metadata or {}
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                content=chunk_text,
                metadata={**metadata, "chunk_index": chunk_idx, "strategy": "fixed_size",
                           "char_start": start, "char_end": end}
            ))
            chunk_idx += 1
        start += chunk_size - overlap  # step forward, keeping overlap

    return chunks


# ── Strategy 2: Paragraph-Aware Chunking ─────────────────────────────────────
def chunk_by_paragraphs(text: str, max_chunk_size: int = 800,
                         metadata: dict = None) -> list[Chunk]:
    """
    Split on blank lines first, then merge small paragraphs up to max_chunk_size.
    Preserves semantic boundaries — much better quality than fixed-size.
    Use when: documents have clear paragraph structure (articles, KB docs).
    """
    metadata = metadata or {}
    # Split on one or more blank lines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        # If adding this paragraph would exceed max, flush current
        if current and len(current) + len(para) + 2 > max_chunk_size:
            chunks.append(Chunk(
                content=current.strip(),
                metadata={**metadata, "chunk_index": chunk_idx, "strategy": "paragraph"}
            ))
            chunk_idx += 1
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para

    if current.strip():
        chunks.append(Chunk(
            content=current.strip(),
            metadata={**metadata, "chunk_index": chunk_idx, "strategy": "paragraph"}
        ))

    return chunks


# ── Strategy 3: Recursive / Sentence-Aware Chunking ──────────────────────────
def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 100,
                    metadata: dict = None) -> list[Chunk]:
    """
    Try paragraph → sentence → word splits in order.
    Falls back to finer splits only when needed.
    This is what LangChain's RecursiveCharacterTextSplitter does.
    Use when: mixed content (some sections short, some long).
    """
    metadata = metadata or {}

    def _split(text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep = separators[0]
        parts = text.split(sep)
        result = []
        current = ""
        for part in parts:
            if not part.strip():
                continue
            trial = (current + sep + part).strip() if current else part.strip()
            if len(trial) <= chunk_size:
                current = trial
            else:
                if current:
                    result.append(current)
                # If part itself is too big, recurse with finer separator
                if len(part) > chunk_size:
                    result.extend(_split(part, separators[1:]))
                else:
                    current = part.strip()
        if current:
            result.append(current)
        return result

    # Separator hierarchy: paragraph → sentence → word
    raw_chunks = _split(text, separators=["\n\n", ". ", " "])

    # Add overlap: each chunk gets the tail of the previous chunk prepended
    chunks = []
    for i, raw in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_tail = raw_chunks[i - 1][-overlap:]
            content = prev_tail + " " + raw
        else:
            content = raw
        chunks.append(Chunk(
            content=content.strip(),
            metadata={**metadata, "chunk_index": i, "strategy": "recursive"}
        ))

    return chunks


# ── Chunking Stats ────────────────────────────────────────────────────────────
def chunking_stats(chunks: list[Chunk], label: str = "") -> None:
    lengths = [len(c.content) for c in chunks]
    print(f"\n  [{label}]  {len(chunks)} chunks")
    print(f"    min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)//len(lengths)}  chars")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    print("=" * 60)
    print("RAG Step 2: Chunking Strategies")
    print("=" * 60)

    kb_path = Path(__file__).parent.parent / "sample_docs" / "techcorp_knowledge_base.txt"
    text = kb_path.read_text(encoding="utf-8")
    meta = {"source": "techcorp_knowledge_base.txt"}

    print(f"\nSource document: {len(text)} characters")

    # Strategy 1: Fixed-size
    fixed = chunk_fixed_size(text, chunk_size=300, overlap=50, metadata=meta)
    chunking_stats(fixed, "Fixed-size  chunk=300  overlap=50")
    print(f"    Sample chunk 0:\n      {fixed[0].content[:120]!r}")

    # Strategy 2: Paragraph-aware
    para = chunk_by_paragraphs(text, max_chunk_size=500, metadata=meta)
    chunking_stats(para, "Paragraph   max=500")
    print(f"    Sample chunk 0:\n      {para[0].content[:120]!r}")

    # Strategy 3: Recursive
    recursive = chunk_recursive(text, chunk_size=400, overlap=80, metadata=meta)
    chunking_stats(recursive, "Recursive   chunk=400  overlap=80")
    print(f"    Sample chunk 0:\n      {recursive[0].content[:120]!r}")

    # Side-by-side: how a pricing section gets chunked
    pricing_section = """
PRICING
--------
Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training
Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training
Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure
"""
    print("\n" + "=" * 60)
    print("Zooming in: how 'PRICING' section is chunked (size=100, overlap=20)")
    print("=" * 60)
    for strategy_name, fn, kwargs in [
        ("Fixed-size", chunk_fixed_size, {"chunk_size": 100, "overlap": 20}),
        ("Paragraph",  chunk_by_paragraphs, {"max_chunk_size": 100}),
        ("Recursive",  chunk_recursive, {"chunk_size": 100, "overlap": 20}),
    ]:
        result = fn(pricing_section, **kwargs)
        print(f"\n  {strategy_name} → {len(result)} chunk(s):")
        for c in result:
            print(f"    [{c.metadata['chunk_index']}] {c.content!r}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Fixed-size: simple but cuts mid-sentence — use for uniform text")
    print("  - Paragraph: best quality for KB docs — respects structure")
    print("  - Recursive: flexible — tries paragraph first, falls back to sentence")
    print("  - Overlap ~10-20% prevents losing context at chunk edges")
    print("  - Too small chunks → no context | Too large → noisy retrieval")
    print("=" * 60)
