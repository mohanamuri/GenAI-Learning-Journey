# tiktoken

**What it is:** OpenAI's fast tokenizer library. Counts tokens exactly as GPT-3.5/GPT-4 does — essential for managing API costs, context windows, and chunking strategies.

**Before tiktoken:**
Developers estimated token count using the "1 token ≈ 4 chars" rule — which is wrong for code, non-English text, and edge cases. This caused silent context overflow bugs and unpredictable API costs.

**Why we picked it here:**
Once you start working with LLMs via API (Module 06), token counting is not optional — it's how you control cost, prevent truncation, and implement chunking for RAG. tiktoken is the standard tool for GPT models. We introduce it here because context window awareness is a Module 05 concept.

**When to use:**
- Counting tokens before sending to any OpenAI/GPT API call
- Chunking documents for RAG (Module 07)
- Estimating API cost before running a batch job
- Implementing sliding window for chat history

**When NOT to use:**
- For Anthropic Claude → use `anthropic` SDK's token counter
- For open-source models (LLaMA, Mistral) → use their own tokenizers via Hugging Face
- For BERT → use `bert-base-uncased` tokenizer (different vocabulary)

**Key API:**
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4, GPT-3.5
# enc = tiktoken.get_encoding("p50k_base")  # GPT-3 (davinci)

tokens = enc.encode("Hello, world!")        # → list of ints
count = len(tokens)                         # token count
text = enc.decode(tokens)                   # back to string
```

**Encoding names:**
- `cl100k_base` → GPT-4, GPT-3.5-turbo, text-embedding-3
- `p50k_base`   → GPT-3 (davinci, curie)
- `r50k_base`   → GPT-2

**Cost estimation:**
```python
# GPT-4o pricing example
tokens = len(enc.encode(your_text))
cost = tokens / 1_000_000 * 5.0  # $5 per 1M input tokens
```

**Install:** `pip install tiktoken`

**Introduced in:** Module 05 — Generative AI (examples 01, 07)
