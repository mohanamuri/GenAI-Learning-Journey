# Author: Mohan Raju Amuri

# ============================================================
# ⚠️  API KEY REQUIRED — COSTS MONEY
# ============================================================
# This example calls the Anthropic Claude cloud API.
# It will NOT run without a valid API key.
#
# Setup:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   OR create a .env file with: ANTHROPIC_API_KEY=sk-ant-...
#
# Estimated cost to run this file: < $0.001
# Model used: claude-haiku-4-5 (cheapest Claude model)
#
# To run for FREE instead → see 03_system_prompts.py (Ollama)
# ============================================================

"""
Anthropic SDK — Basics
------------------------
One-liner: Anthropic's SDK calls Claude models — same pattern as OpenAI but different method names.

Key differences from OpenAI:
  anthropic.Anthropic()           vs  openai.OpenAI()
  client.messages.create()        vs  client.chat.completions.create()
  message.content[0].text         vs  response.choices[0].message.content
  system= is a top-level param    vs  system is inside messages list
  max_tokens is REQUIRED          vs  optional in OpenAI

Claude model tiers (2025):
  claude-haiku-4-5      → fastest, cheapest  ← dev/learning default
  claude-sonnet-4-5     → balanced speed + quality
  claude-opus-4-5       → most powerful, slowest, most expensive

Remember:
- system prompt is a separate parameter, NOT inside the messages list
- max_tokens is mandatory — no default
- response.usage.input_tokens / output_tokens (not prompt/completion like OpenAI)
- Claude tends to be more verbose by default — use system prompt to constrain

Don't:
- Don't put system message inside messages list (it's a top-level param)
- Don't forget max_tokens — call will fail without it
- Don't use claude-opus for every task — haiku handles most at a fraction of cost
- Don't hardcode API key — use ANTHROPIC_API_KEY env var

Setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY automatically


# ── 1. Basic call ──────────────────────────────────────────────────────────
def ask(user_message: str, model: str = "claude-haiku-4-5") -> str:
    message = client.messages.create(
        model=model,
        max_tokens=256,                          # required — no default
        messages=[
            {"role": "user", "content": user_message}
        ],
    )
    return message.content[0].text


print("=== 1. Basic call ===")
print(ask("What is the difference between RAM and ROM? Answer in 2 sentences."))


# ── 2. System prompt (top-level, NOT in messages) ─────────────────────────
def ask_with_system(system: str, user: str, model: str = "claude-haiku-4-5") -> str:
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,                           # top-level param, not in messages!
        messages=[
            {"role": "user", "content": user}
        ],
    )
    return message.content[0].text


print("\n=== 2. System prompt ===")
print(ask_with_system(
    system="You are a concise Python tutor. Use bullet points only. Max 5 bullets.",
    user="Explain decorators in Python.",
))


# ── 3. Token usage ────────────────────────────────────────────────────────
print("\n=== 3. Token usage ===")
message = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    messages=[{"role": "user", "content": "Name 3 Python web frameworks."}],
)
usage = message.usage
print(f"Answer        : {message.content[0].text.strip()}")
print(f"Input tokens  : {usage.input_tokens}")
print(f"Output tokens : {usage.output_tokens}")
# Haiku pricing: ~$0.25/1M input, $1.25/1M output
cost = (usage.input_tokens / 1_000_000 * 0.25) + (usage.output_tokens / 1_000_000 * 1.25)
print(f"Approx cost   : ${cost:.6f}")


# ── 4. OpenAI vs Anthropic: side-by-side ─────────────────────────────────
print("\n=== 4. OpenAI vs Anthropic: Key Differences ===")
diff = [
    ("Client",         "OpenAI()",                    "Anthropic()"),
    ("Method",         "chat.completions.create()",   "messages.create()"),
    ("System prompt",  "Inside messages list",        "Top-level system= param"),
    ("max_tokens",     "Optional",                    "REQUIRED"),
    ("Response text",  "choices[0].message.content",  "content[0].text"),
    ("Token usage",    "usage.prompt_tokens",         "usage.input_tokens"),
    ("Env var",        "OPENAI_API_KEY",               "ANTHROPIC_API_KEY"),
]
print(f"{'Aspect':<18} {'OpenAI':<35} {'Anthropic'}")
print("-" * 80)
for aspect, oai, ant in diff:
    print(f"{aspect:<18} {oai:<35} {ant}")
