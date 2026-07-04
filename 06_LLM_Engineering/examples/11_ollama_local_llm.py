# Author: Mohan Raju Amuri
"""
Ollama — Local LLMs
---------------------
One-liner: Run LLMs on your own machine — no API key, no internet, no cost per token.

Remember:
- Ollama runs a local server at http://localhost:11434
- OpenAI-compatible API → same code works with local + cloud models
- Models are downloaded once and stored locally (~4-8 GB per model)
- Good for: dev/testing, privacy-sensitive data, offline use, cost control
- Speed depends on your RAM/GPU — CPU-only is slower but works

Popular models:
  llama3.2:3b    → 3B params, fast on CPU, good quality  (~2 GB)
  llama3.1:8b    → 8B params, best quality/speed balance (~5 GB)
  mistral:7b     → strong coding + reasoning             (~4 GB)
  phi3:mini      → Microsoft, very fast, small           (~2 GB)
  nomic-embed-text → for embeddings (no API cost)       (~275 MB)

Don't:
- Don't expect same quality as GPT-4 — local 7B ≈ GPT-3.5 quality
- Don't run on a machine with <8 GB RAM for 7B models
- Don't use for production customer-facing apps without benchmarking
- Don't forget: first model pull is slow (GB download)

Setup:
  1. Install Ollama: https://ollama.com/download
  2. Pull a model: ollama pull llama3.2:3b
  3. Ollama server starts automatically
"""

from openai import OpenAI  # works with Ollama's OpenAI-compatible API!

# Point to local Ollama server — no API key needed
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",        # required by SDK but ignored by Ollama
)


def ask_local(prompt: str, model: str = "llama3.2:3b") -> str:
    response = ollama_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content


def ask_local_stream(prompt: str, model: str = "llama3.2:3b") -> str:
    print("Response: ", end="", flush=True)
    full = []
    with ollama_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full.append(delta)
    print()
    return "".join(full)


# ── 1. Basic call ──────────────────────────────────────────────────────────
print("=== 1. Local LLM (Ollama) ===")
print("(Requires: ollama pull llama3.2:3b)\n")

try:
    answer = ask_local("What is Python? Answer in 2 sentences.")
    print(f"Answer: {answer}")

    # ── 2. Streaming ──────────────────────────────────────────────────────
    print("\n=== 2. Streaming ===")
    ask_local_stream("List 3 benefits of using local LLMs.")

    # ── 3. Local embeddings (nomic-embed-text) ─────────────────────────────
    print("\n=== 3. Local Embeddings ===")
    print("(Requires: ollama pull nomic-embed-text)\n")
    try:
        response = ollama_client.embeddings.create(
            model="nomic-embed-text",
            input="Machine learning is a subset of AI",
        )
        emb = response.data[0].embedding
        print(f"Embedding dims : {len(emb)}")
        print(f"First 5 values : {[round(v, 4) for v in emb[:5]]}")
        print("Use these for RAG without paying per embedding!")
    except Exception:
        print("nomic-embed-text not installed — run: ollama pull nomic-embed-text")

except Exception as e:
    print(f"Ollama not running. Start it with: ollama serve")
    print(f"Then pull a model: ollama pull llama3.2:3b")
    print(f"Error: {e}")


# ── 4. OpenAI ↔ Ollama swap pattern ──────────────────────────────────────
print("\n=== 4. Easy Cloud ↔ Local Swap ===")
print("""
# Cloud (OpenAI)
client = OpenAI()
# client.chat.completions.create(model="gpt-4o-mini", ...)

# Local (Ollama) — same code, different client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# client.chat.completions.create(model="llama3.2:3b", ...)

# Tip: use an env variable to switch
import os
if os.getenv("USE_LOCAL_LLM"):
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL = "llama3.2:3b"
else:
    client = OpenAI()
    MODEL = "gpt-4o-mini"
""")


# ── 5. Model selection guide ──────────────────────────────────────────────
print("=== 5. Local Model Guide ===")
models = [
    ("llama3.2:3b",        "~2 GB", "Fast, good for most tasks, CPU-friendly"),
    ("llama3.1:8b",        "~5 GB", "Best local quality, needs 8+ GB RAM"),
    ("mistral:7b",         "~4 GB", "Strong at coding and reasoning"),
    ("phi3:mini",          "~2 GB", "Microsoft, very fast on CPU"),
    ("nomic-embed-text",   "~275 MB","Embeddings only — free RAG"),
    ("codellama:7b",       "~4 GB", "Code generation and explanation"),
]
print(f"{'Model':<22} {'Size':>7}  Use case")
print("-" * 60)
for name, size, use in models:
    print(f"{name:<22} {size:>7}  {use}")
