# Ollama

**What it is:** Run open-source LLMs (LLaMA, Mistral, Phi) locally on your machine. Provides an OpenAI-compatible REST API at `localhost:11434`.

**Before Ollama (pre-2023):**
Running local LLMs meant: manually downloading GGUF weights, installing llama.cpp, compiling with CUDA flags, writing your own server. Painful. Ollama made it `ollama pull llama3` and done.

**Why we picked it here:**
Two reasons:
1. **Learning without cost** — practice API patterns without spending tokens
2. **The OpenAI-compatible interface** — same SDK code works for cloud and local. Understanding this swap is a key LLM engineering skill.

**When to use:**
- Development and testing (zero cost)
- Privacy-sensitive data (nothing leaves your machine)
- Offline environments
- Benchmarking open-source vs proprietary models
- Embedding generation without API cost (`nomic-embed-text`)

**When NOT to use:**
- Customer-facing production (quality gap vs GPT-4o)
- Machine with <8 GB RAM for 7B+ models
- When you need GPT-4 level reasoning

**The OpenAI swap pattern:**
```python
# Same OpenAI SDK — just change base_url and model name
from openai import OpenAI

# Cloud
client = OpenAI()
model = "gpt-4o-mini"

# Local
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
model = "llama3.2:3b"

# Identical call
response = client.chat.completions.create(model=model, messages=[...])
```

**Setup:**
```bash
# 1. Install
brew install ollama                 # macOS
# or: https://ollama.com/download   # Windows/Linux

# 2. Pull models
ollama pull llama3.2:3b             # ~2 GB, fast
ollama pull nomic-embed-text        # ~275 MB, embeddings

# 3. Server starts automatically (or: ollama serve)
```

**Recommended models:**
- `llama3.2:3b` → daily use, fast on CPU
- `llama3.1:8b` → best quality for 8 GB RAM
- `nomic-embed-text` → free embeddings for RAG

**Introduced in:** Module 06 — LLM Engineering (example 11)
