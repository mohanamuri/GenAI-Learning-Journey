# Author: Mohan Raju Amuri
"""
Streaming
----------
One-liner: Stream tokens as they are generated instead of waiting for the full response — better UX for long outputs.

Remember:
- Streaming returns a generator — iterate with for chunk in stream
- OpenAI/Ollama: chunk.choices[0].delta.content (may be None for first/last chunk)
- Time-to-first-token (TTFT) is instant with streaming — user sees output immediately
- Total latency is the same — streaming just improves perceived speed
- For APIs/batch jobs: don't stream. For chat UIs: always stream.

Don't:
- Don't check chunk.choices[0].message — that's non-streaming. Use .delta for streaming.
- Don't collect all chunks then print — defeats the purpose
- Don't stream for programmatic use where you need the full response before processing
- Don't forget: chunk.choices[0].delta.content can be None — always guard it

Running on Ollama (local, free — no API key needed):
  ollama pull llama3.2:3b
"""

from openai import OpenAI

# ── Using Ollama (local LLM) — free, no API key required ──────────────────
# Ollama fully supports streaming — same behaviour as OpenAI cloud
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "llama3.2:3b"


# ── 1. Basic streaming ────────────────────────────────────────────────────
# Notice how output appears token-by-token — this is what makes chat UIs feel responsive
print("=== 1. Basic Streaming ===")
print("Response: ", end="", flush=True)

with client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
    stream=True,
    max_tokens=100,
) as stream:
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:       # guard: first/last chunks can be None
            print(delta, end="", flush=True)
print()


# ── 2. Collect streamed response ──────────────────────────────────────────
# In production you often need to: display tokens live AND store the full response.
# This pattern does both — stream to stdout + accumulate in a list.
print("\n=== 2. Stream + Collect ===")

def stream_and_collect(prompt: str) -> str:
    """Stream tokens to stdout while collecting the full response."""
    full_response = []
    print("Streaming: ", end="", flush=True)
    with client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=150,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                print(delta, end="", flush=True)
                full_response.append(delta)  # accumulate for later use
    print()
    return "".join(full_response)

full_text = stream_and_collect("List 3 benefits of Python in one sentence each.")
print(f"\nCollected ({len(full_text)} chars): {full_text[:60]}...")


# ── 3. When to stream vs not ─────────────────────────────────────────────
# Use streaming only when the user is watching — not in batch processing
print("\n=== When to Stream ===")
decisions = [
    ("Chat UI / user-facing",          "YES — improves perceived speed"),
    ("Long generation (>200 tokens)",  "YES — user doesn't stare at spinner"),
    ("Batch processing 1000 docs",     "NO  — streaming adds overhead with no benefit"),
    ("API that returns JSON",          "NO  — need full response before parsing"),
    ("Classification (short output)",  "NO  — response arrives fast anyway"),
]
for scenario, decision in decisions:
    print(f"  {scenario:<40} → {decision}")
