# Author: Mohan Raju Amuri

# ============================================================
# ⚠️  API KEY REQUIRED — COSTS MONEY
# ============================================================
# This example calls the OpenAI API to demonstrate real cost
# comparisons between models. Needs a real API to show actual
# pricing — cannot be meaningfully demonstrated with Ollama
# (local models have no per-token cost).
#
# Setup:
#   export OPENAI_API_KEY="sk-..."
#   OR create a .env file with: OPENAI_API_KEY=sk-...
#
# Estimated cost to run this file: < $0.002
# (intentionally makes multiple calls to compare models)
#
# What you'll learn: how model choice and prompt length
# directly affect cost — the most important cost lever.
# ============================================================

"""
Cost Optimization
------------------
One-liner: LLM cost = tokens × price/token — optimize by reducing tokens and picking the right model.

Cost levers (in order of impact):
  1. Model selection     — gpt-4o-mini is 30× cheaper than gpt-4o for same quality on simple tasks
  2. Prompt compression  — shorter prompts = fewer input tokens every call
  3. Output length       — set max_tokens, ask for concise answers
  4. Caching             — same prompt = same cost every time (cache it!)
  5. Batching            — batch API (OpenAI) is 50% cheaper for non-realtime tasks
  6. Right-sizing        — don't use GPT-4 for classification, use GPT-4o-mini

Remember:
- Input tokens are cheaper than output tokens (usually 3-5× cheaper)
- System prompt tokens are charged EVERY call — trim ruthlessly
- A/B test: gpt-4o-mini first, only upgrade if quality is insufficient
- Semantic caching: cache by embedding similarity, not exact string match

Don't:
- Don't use gpt-4o for tasks gpt-4o-mini handles well
- Don't send full documents when you only need excerpts
- Don't repeat context that's already in the system prompt
- Don't skip caching for repeated/similar queries in production
"""

import hashlib
import time
from openai import OpenAI
import tiktoken

client = OpenAI()
enc = tiktoken.get_encoding("cl100k_base")

PRICES = {
    "gpt-4o-mini":   {"input": 0.15,  "output": 0.60},   # per 1M tokens
    "gpt-4o":        {"input": 5.00,  "output": 15.00},
    "gpt-3.5-turbo": {"input": 0.50,  "output": 1.50},
}


def estimate_cost(prompt: str, response: str, model: str) -> float:
    input_tokens  = len(enc.encode(prompt))
    output_tokens = len(enc.encode(response))
    p = PRICES[model]
    return (input_tokens / 1_000_000 * p["input"]) + (output_tokens / 1_000_000 * p["output"])


# ── 1. Model selection impact ─────────────────────────────────────────────
print("=== 1. Same task, different models ===\n")

task = "Classify this review as POSITIVE or NEGATIVE: 'Great product, fast shipping!'"

for model in ["gpt-4o-mini", "gpt-4o"]:
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": task}],
        max_tokens=5,
    )
    elapsed = time.time() - start
    answer = response.choices[0].message.content.strip()
    cost = estimate_cost(task, answer, model)
    print(f"  {model:<15} → {answer:<10} cost=${cost:.7f}  time={elapsed:.2f}s")


# ── 2. Prompt compression ─────────────────────────────────────────────────
print("\n=== 2. Prompt Compression ===\n")

verbose_prompt = """
I would like you to please help me with a task. What I need is for you to take a look at
the following piece of text and determine whether the sentiment expressed within it is of
a positive nature or a negative nature. Please respond with just the word POSITIVE or NEGATIVE.
The text that I would like you to analyze is as follows: 'I love this product!'
"""

concise_prompt = "Sentiment (POSITIVE/NEGATIVE): 'I love this product!'"

verbose_tokens  = len(enc.encode(verbose_prompt))
concise_tokens  = len(enc.encode(concise_prompt))

print(f"  Verbose prompt : {verbose_tokens} tokens")
print(f"  Concise prompt : {concise_tokens} tokens")
print(f"  Savings        : {verbose_tokens - concise_tokens} tokens ({(1 - concise_tokens/verbose_tokens)*100:.0f}% reduction)")
print(f"  At scale (10k calls, gpt-4o-mini): ${(verbose_tokens - concise_tokens) * 10000 / 1_000_000 * 0.15:.4f} saved")


# ── 3. Simple in-memory cache ────────────────────────────────────────────
print("\n=== 3. Prompt Caching ===\n")

_cache: dict[str, str] = {}

def cached_llm(prompt: str, model: str = "gpt-4o-mini") -> tuple[str, bool]:
    """Returns (response, cache_hit)."""
    key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
    if key in _cache:
        return _cache[key], True

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    result = response.choices[0].message.content
    _cache[key] = result
    return result, False

prompt = "What are the 3 pillars of object-oriented programming?"
for i in range(3):
    start = time.time()
    result, hit = cached_llm(prompt)
    elapsed = time.time() - start
    status = "CACHE HIT" if hit else "API CALL "
    print(f"  Call {i+1}: [{status}] {elapsed:.3f}s  {result[:40]}...")


# ── 4. Cost comparison table ─────────────────────────────────────────────
print("\n=== 4. Cost at Scale (10k calls, 500 input + 200 output tokens) ===\n")
calls = 10_000
input_tok, output_tok = 500, 200
print(f"{'Model':<15} {'Per call':>10} {'10k calls':>12}")
print("-" * 40)
for model, price in PRICES.items():
    per_call = (input_tok / 1_000_000 * price["input"]) + (output_tok / 1_000_000 * price["output"])
    total = per_call * calls
    print(f"{model:<15} ${per_call:>9.5f}  ${total:>10.2f}")
