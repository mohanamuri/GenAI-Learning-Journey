"""
Context Window
---------------
One-liner: The context window is how much text an LLM can "see" at once — input + output combined.

Remember:
- Context window = max tokens for input + output together
- Exceeding it: oldest tokens are dropped (sliding window) or it errors
- Larger window ≠ better performance — models degrade on very long contexts
  → "Lost in the middle" effect: info buried in the middle gets ignored
- Always count tokens BEFORE sending to the API — avoid surprise truncation
- Reserve headroom: if limit=128k, don't send 127k — leave room for output

Context window sizes (2024–2025):
  GPT-3.5-turbo  →  16,385 tokens  (~12,000 words)
  GPT-4o         → 128,000 tokens  (~96,000 words)
  Claude 3.5     → 200,000 tokens  (~150,000 words)
  LLaMA 3 8B     →   8,192 tokens  (~6,000 words)
  Gemini 1.5 Pro →   1,000,000 tokens

Don't:
- Don't assume your full document fits — always count tokens first
- Don't rely on info buried in the middle of a huge context — put key info at start or end
- Don't confuse context window with memory — LLMs have no persistent memory between calls
- Don't send the full conversation history forever — implement a windowing strategy

Chunking rule of thumb:
  chunk_size = 512 tokens, overlap = 50 tokens  (for RAG — Module 07)
"""

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def fits_in_context(text: str, model_limit: int, reserved_for_output: int = 500) -> bool:
    return count_tokens(text) <= (model_limit - reserved_for_output)


# ── 1. Token counting ─────────────────────────────────────────────────────
print("=== Token Counting ===")
samples = {
    "One sentence":     "The quick brown fox jumps over the lazy dog.",
    "Short paragraph":  "Machine learning is transforming how we build software. " * 5,
    "1-page document":  "This is a representative paragraph. " * 100,
    "10-page document": "This is a representative paragraph. " * 1000,
}
for label, text in samples.items():
    tokens = count_tokens(text)
    words = len(text.split())
    print(f"  {label:<20} {words:>6} words  →  {tokens:>6} tokens  (ratio: {tokens/words:.2f})")

# ── 2. Context limits by model ────────────────────────────────────────────
print("\n=== Context Limit Check ===")
models = {
    "GPT-3.5-turbo":  16_385,
    "GPT-4o":        128_000,
    "Claude 3.5":    200_000,
    "LLaMA 3 8B":      8_192,
}
test_text = "This is a sample paragraph. " * 500  # ~3500 tokens
tokens = count_tokens(test_text)
print(f"Document size: {tokens} tokens\n")
for model, limit in models.items():
    fits = "✓ fits" if tokens < limit - 500 else "✗ too long"
    pct = tokens / limit * 100
    print(f"  {model:<20} limit={limit:>7,}  used={pct:4.1f}%  {fits}")

# ── 3. Sliding window for chat history ────────────────────────────────────
print("\n=== Sliding Window for Chat History ===")

class ChatHistory:
    """Keep only the most recent messages that fit in the context window."""
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self):
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)  # remove oldest message

    def _total_tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.messages)

    def __len__(self) -> int:
        return len(self.messages)


history = ChatHistory(max_tokens=200)
for i in range(10):
    history.add("user", f"This is message {i}. " + "Some content here. " * 5)
    history.add("assistant", f"Reply to message {i}. " + "Some response content. " * 5)

print(f"Total messages added: 20")
print(f"Messages kept in window: {len(history.messages)}")
print(f"Total tokens in window: {history._total_tokens()}")

# ── 4. Chunking strategy preview (for RAG) ────────────────────────────────
print("\n=== Chunking Preview (for RAG in Module 07) ===")
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20) -> list[str]:
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        start += chunk_size - overlap  # overlap for context continuity
    return chunks

long_text = "Natural language processing is a field of AI. " * 50
chunks = chunk_text(long_text, chunk_size=100, overlap=10)
print(f"Document: {count_tokens(long_text)} tokens")
print(f"Chunks  : {len(chunks)} chunks of ~100 tokens each (10 token overlap)")
print(f"Chunk 1 : {chunks[0][:80]}...")
