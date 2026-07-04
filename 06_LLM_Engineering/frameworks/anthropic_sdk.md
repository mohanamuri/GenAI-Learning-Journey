# Anthropic SDK

**What it is:** Official Python client for Anthropic's Claude models (Haiku, Sonnet, Opus).

**Before the SDK:**
Same story as OpenAI — raw requests, manual JSON, no type safety.

**Why we picked it here:**
Claude is a strong alternative to GPT — often better at following nuanced instructions, longer context (200k tokens), and more predictable on complex reasoning. Knowing both SDKs means you can benchmark and pick the right model per task.

**Key difference from OpenAI SDK:**

| | OpenAI | Anthropic |
|---|---|---|
| System prompt | Inside `messages` list | Top-level `system=` param |
| `max_tokens` | Optional | **Required** |
| Response text | `choices[0].message.content` | `content[0].text` |
| Token usage | `usage.prompt_tokens` | `usage.input_tokens` |
| Env var | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` |

**When to use:**
- When Claude's longer context (200k) matters
- When you want an alternative to OpenAI for benchmarking
- When nuanced instruction-following is critical
- Privacy: Anthropic has different data retention policies

**When NOT to use:**
- When you already have OpenAI infrastructure and Claude's quality gap doesn't justify switching
- For embeddings — Anthropic doesn't have an embeddings endpoint (use OpenAI or sentence-transformers)

**Key API:**
```python
from anthropic import Anthropic
client = Anthropic()  # reads ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=256,           # required!
    system="You are a tutor", # top-level, NOT in messages
    messages=[{"role": "user", "content": "Explain transformers"}],
)
text = message.content[0].text

# Streaming
with client.messages.stream(model=..., max_tokens=..., messages=[...]) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

**Model tiers:**
- `claude-haiku-4-5` → fastest, cheapest (use for dev)
- `claude-sonnet-4-5` → balanced
- `claude-opus-4-5` → most capable, most expensive

**Install:** `pip install anthropic`

**Env var:** `export ANTHROPIC_API_KEY="sk-ant-..."`

**Introduced in:** Module 06 — LLM Engineering (example 02)
