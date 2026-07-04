# OpenAI SDK

**What it is:** Official Python client for OpenAI's API. Wraps REST calls to GPT-4o, GPT-4o-mini, embeddings, DALL-E, and Whisper into clean Python objects.

**Before the SDK existed:**
Raw `requests` calls to the REST API. Manual JSON parsing, no type hints, no built-in retry logic. Error handling was entirely manual.

**Why we picked it here:**
GPT-4o-mini is the default workhorse for LLM engineering tasks. The SDK is the most documented, most widely used LLM client — knowing it means you can read 90% of LLM engineering code on GitHub.

**When to use:**
- GPT-4o / GPT-4o-mini for production tasks
- OpenAI embeddings (`text-embedding-3-small`)
- Structured outputs with Pydantic (`.parse()`)
- When you need proven, well-documented API

**When NOT to use:**
- When you want to avoid vendor lock-in → use LiteLLM (unified interface)
- For Anthropic/Gemini models → use their own SDKs
- For local models → use Ollama (same OpenAI interface, different base_url)

**Key API:**
```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY env var

# Chat
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
text = response.choices[0].message.content

# Structured output
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[...],
    response_format=MyPydanticModel,
)
obj = response.choices[0].message.parsed

# Embeddings
emb = client.embeddings.create(model="text-embedding-3-small", input="text")
vector = emb.data[0].embedding
```

**Built-in retry:** `OpenAI(max_retries=3, timeout=30.0)`

**Install:** `pip install openai`

**Env var:** `export OPENAI_API_KEY="sk-..."`

**Introduced in:** Module 06 — LLM Engineering (examples 01, 03–10)
