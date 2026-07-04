# Author: Mohan Raju Amuri
"""
Error Handling & Retries
-------------------------
One-liner: LLM APIs fail — rate limits, timeouts, server errors. Build retry logic from day one.

Common errors:
  RateLimitError    → 429, too many requests — wait and retry with backoff
  APITimeoutError   → request took too long — retry
  APIConnectionError → network issue — retry
  AuthenticationError → bad API key — don't retry, fix the key
  BadRequestError   → invalid request (bad model name, token limit exceeded) — don't retry

Retry strategy:
  Exponential backoff: wait 1s → 2s → 4s → 8s ... (with jitter)
  Max retries: 3-5 for transient errors
  Never retry: auth errors, invalid requests

Remember:
- openai SDK has built-in retry: OpenAI(max_retries=3)
- Always log errors with enough context to debug later
- Set a timeout: OpenAI(timeout=30.0) — prevents hanging forever
- Rate limit = slow down, not give up — implement a queue for bulk processing

Don't:
- Don't retry on AuthenticationError — it will never succeed
- Don't retry immediately — always wait (exponential backoff)
- Don't retry more than 5 times for a single request
- Don't swallow exceptions silently — at minimum log them
- Don't forget to set request timeouts in production

Running on Ollama (local, free — no API key, no rate limits):
  ollama pull llama3.2:3b
  Note: With Ollama you won't hit RateLimitError, but APIConnectionError
  can still happen if the Ollama server isn't running.
"""

import time
import random
import logging
from openai import OpenAI, APIConnectionError, APITimeoutError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Using Ollama (local LLM) — free, no API key, no rate limits ───────────
# With Ollama you mainly worry about: APIConnectionError (server not running)
# and APITimeoutError (model too slow for large prompts).
# The retry patterns below apply to both local and cloud APIs.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    max_retries=3,       # built-in retry with exponential backoff
    timeout=60.0,        # local models can be slower — generous timeout
)
MODEL = "llama3.2:3b"


# ── 1. Manual retry with exponential backoff ──────────────────────────────
# Exponential backoff + jitter is the industry standard for retry logic.
# Jitter prevents "thundering herd" when many clients retry at once.
def call_with_retry(
    prompt: str,
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> str | None:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.choices[0].message.content

        except APITimeoutError:
            if attempt == max_retries:
                logger.error("Timeout: max retries reached")
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(f"Timeout (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s")
            time.sleep(delay)

        except APIConnectionError as e:
            if attempt == max_retries:
                logger.error(f"Connection error: {e}")
                raise
            delay = base_delay * attempt
            logger.warning(f"Connection error (attempt {attempt}). Is Ollama running? Retrying in {delay:.1f}s")
            time.sleep(delay)

        except Exception as e:
            # Unknown error — log and don't retry
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            raise

    return None


# ── 2. Fallback model chain ───────────────────────────────────────────────
# In production: try fast model first, fall back to larger one if it fails.
# With Ollama: try small model first, fall back to larger local model.
def call_with_fallback(prompt: str, models: list[str]) -> tuple[str, str] | None:
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            return response.choices[0].message.content, model
        except Exception as e:
            logger.warning(f"Model {model} failed: {type(e).__name__}. Trying next.")
    logger.error("All models failed")
    return None


print("=== 1. Normal call with retry ===")
try:
    result = call_with_retry("What is 2 + 2?")
    print(f"Result: {result}")
except Exception as e:
    print(f"Failed after retries: {e}")


print("\n=== 2. Fallback model chain ===")
# Falls back to next model if the first isn't available
result = call_with_fallback(
    "Name the capital of France in one word.",
    models=["llama3.2:3b", "llama3.1:8b"],  # try small first, fall back to larger
)
if result:
    answer, model_used = result
    print(f"Answer: {answer.strip()} (using {model_used})")


# ── 3. Token limit protection ─────────────────────────────────────────────
# Always validate input length before sending — avoids BadRequestError
# and prevents wasting time on requests that will fail anyway.
print("\n=== 3. Token limit protection ===")
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def safe_call(prompt: str, max_input_tokens: int = 3000) -> str:
    token_count = len(enc.encode(prompt))
    if token_count > max_input_tokens:
        # Truncate to fit — log warning so it's visible
        tokens = enc.encode(prompt)[:max_input_tokens]
        prompt = enc.decode(tokens)
        logger.warning(f"Prompt truncated from {token_count} to {max_input_tokens} tokens")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content

long_prompt = "Summarize this: " + ("word " * 5000)
result = safe_call(long_prompt)
print(f"Result: {result[:80]}...")


# ── 4. Error taxonomy ────────────────────────────────────────────────────
print("\n=== Error Taxonomy (Cloud + Local) ===")
errors = [
    ("RateLimitError (429)",      "Cloud only", "YES", "Exponential backoff + jitter"),
    ("APITimeoutError",           "Both",       "YES", "Retry with longer timeout"),
    ("APIConnectionError",        "Both",       "YES", "Check server, retry"),
    ("AuthenticationError (401)", "Cloud only", "NO",  "Fix API key, never retry"),
    ("BadRequestError (400)",     "Both",       "NO",  "Fix prompt/params, never retry"),
]
print(f"{'Error':<30} {'Where':<12} {'Retry?':>7}  {'Action'}")
print("-" * 72)
for err, where, retry, action in errors:
    print(f"{err:<30} {where:<12} {retry:>7}  {action}")
