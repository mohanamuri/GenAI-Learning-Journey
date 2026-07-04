# Author: Mohan Raju Amuri

# ============================================================
# ⚠️  API KEY REQUIRED — COSTS MONEY
# ============================================================
# This example calls the OpenAI cloud API.
# It will NOT run without a valid API key.
#
# Setup:
#   export OPENAI_API_KEY="sk-..."
#   OR create a .env file with: OPENAI_API_KEY=sk-...
#
# Estimated cost to run this file: < $0.001
# Model used: gpt-4o-mini (cheapest, use this for learning)
#
# To run for FREE instead → see 03_system_prompts.py (Ollama)
# ============================================================

"""
OpenAI SDK — Basics
---------------------
One-liner: The OpenAI Python SDK wraps the REST API — one call returns a chat completion.

Remember:
- client = OpenAI() reads OPENAI_API_KEY from environment automatically
- model="gpt-4o-mini" is cheapest + fast — use for learning/dev
- response.choices[0].message.content → your answer string
- response.usage → token counts (prompt + completion + total)
- Messages are a list of dicts: {"role": "...", "content": "..."}
- Roles: "system", "user", "assistant"

Pricing (approximate, check openai.com for current):
  gpt-4o-mini   $0.15 / 1M input,  $0.60 / 1M output  ← dev/learning default
  gpt-4o        $5.00 / 1M input, $15.00 / 1M output
  gpt-3.5-turbo $0.50 / 1M input,  $1.50 / 1M output

Don't:
- Don't hardcode API keys in code — use environment variables or .env
- Don't use gpt-4o for every task — gpt-4o-mini handles 80% of tasks at 10% cost
- Don't ignore response.usage — track tokens to understand cost
- Don't send a new client per request — instantiate once and reuse

Setup:
  pip install openai
  export OPENAI_API_KEY="sk-..."   # or use .env file
"""

import os
from openai import OpenAI

# Load from environment — never hardcode
# Set with: export OPENAI_API_KEY="sk-..."
client = OpenAI()  # reads OPENAI_API_KEY automatically


# ── 1. Simplest possible call ──────────────────────────────────────────────
def ask(question: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


print("=== 1. Basic call ===")
answer = ask("What is the difference between RAM and ROM? Answer in 2 sentences.")
print(answer)


# ── 2. System + user message ──────────────────────────────────────────────
def ask_with_system(system: str, user: str, model: str = "gpt-4o-mini") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content


print("\n=== 2. System + User ===")
answer = ask_with_system(
    system="You are a concise Python tutor. Answer in bullet points only.",
    user="Explain list comprehension.",
)
print(answer)


# ── 3. Token usage tracking ───────────────────────────────────────────────
print("\n=== 3. Token Usage ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Name 3 Python web frameworks."}],
)
usage = response.usage
print(f"Answer       : {response.choices[0].message.content}")
print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Output tokens: {usage.completion_tokens}")
print(f"Total tokens : {usage.total_tokens}")
cost = (usage.prompt_tokens / 1_000_000 * 0.15) + (usage.completion_tokens / 1_000_000 * 0.60)
print(f"Approx cost  : ${cost:.6f}")


# ── 4. Parameters: temperature, max_tokens ────────────────────────────────
print("\n=== 4. Key Parameters ===")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a one-line tagline for an AI startup."}],
    temperature=0.9,      # creative
    max_tokens=50,        # cap output length
    n=3,                  # return 3 completions
)
for i, choice in enumerate(response.choices, 1):
    print(f"  [{i}] {choice.message.content.strip()}")


# ── 5. Model comparison ───────────────────────────────────────────────────
print("\n=== 5. Model Selection Guide ===")
models = {
    "gpt-4o-mini":   "Dev, learning, 80% of tasks — USE THIS BY DEFAULT",
    "gpt-4o":        "Complex reasoning, vision, production",
    "gpt-3.5-turbo": "Legacy — prefer gpt-4o-mini instead",
    "o1-mini":       "Math, coding, multi-step reasoning",
}
for model, use in models.items():
    print(f"  {model:<20} → {use}")
