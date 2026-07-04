"""
Temperature, Top-K, Top-P (Sampling Parameters)
-------------------------------------------------
One-liner: These three knobs control randomness in LLM output — how "creative" vs "predictable" it is.

Temperature:
  0.0       → deterministic (always picks highest prob token)
  0.1–0.5   → focused, factual, low creativity
  0.7–0.9   → balanced (good default for most tasks)
  1.0       → model's raw distribution
  >1.0      → chaotic, high randomness (rarely useful)

Top-K:
  Keep only the K highest probability tokens, sample from those.
  top_k=1   → greedy (same as temperature=0)
  top_k=50  → common default

Top-P (nucleus sampling):
  Keep smallest set of tokens whose cumulative probability >= P.
  top_p=0.9 → covers 90% of probability mass (adapts to context)
  top_p=1.0 → no filtering
  Preferred over top_k because it's adaptive.

Remember:
- Temperature + top_p together is the most common combo in production
- temperature=0 for factual tasks (extraction, classification, structured output)
- temperature=0.7–0.9 for conversational, creative tasks
- Never use temperature > 1 in production
- top_p=0.9, temperature=0.7 is a solid default

Don't:
- Don't set both top_k and top_p at the same time — pick one
- Don't use high temperature for code generation — you want precision
- Don't use temperature=0 for creative writing — it's boring
"""

from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2", pad_token_id=50256)
prompt = "The best way to learn programming is"

# ── 1. Temperature comparison ─────────────────────────────────────────────
print("=== Temperature Effect ===")
print(f"Prompt: '{prompt}'\n")

for temp in [0.1, 0.7, 1.2]:
    set_seed(42)
    result = generator(
        prompt,
        max_new_tokens=25,
        do_sample=True if temp > 0 else False,
        temperature=temp if temp > 0 else None,
        num_return_sequences=1,
    )
    output = result[0]["generated_text"].replace("\n", " ")
    print(f"  temp={temp}: {output}")

# ── 2. Top-P (nucleus) sampling ───────────────────────────────────────────
print("\n=== Top-P (Nucleus) Sampling ===")
for top_p in [0.5, 0.9, 1.0]:
    set_seed(42)
    result = generator(
        prompt,
        max_new_tokens=25,
        do_sample=True,
        top_p=top_p,
        temperature=1.0,
    )
    output = result[0]["generated_text"].replace("\n", " ")
    print(f"  top_p={top_p}: {output}")

# ── 3. Top-K sampling ─────────────────────────────────────────────────────
print("\n=== Top-K Sampling ===")
for top_k in [1, 10, 50]:
    set_seed(42)
    result = generator(
        prompt,
        max_new_tokens=25,
        do_sample=True if top_k > 1 else False,
        top_k=top_k,
    )
    output = result[0]["generated_text"].replace("\n", " ")
    label = "greedy" if top_k == 1 else f"top_k={top_k}"
    print(f"  {label}: {output}")

# ── 4. Production-recommended settings ────────────────────────────────────
print("\n=== Recommended Settings by Task ===")
settings = {
    "Factual Q&A / extraction":    {"temperature": 0.0, "top_p": 1.0,  "note": "deterministic"},
    "Code generation":             {"temperature": 0.2, "top_p": 0.95, "note": "precise"},
    "Chatbot / conversation":      {"temperature": 0.7, "top_p": 0.9,  "note": "natural"},
    "Creative writing":            {"temperature": 0.9, "top_p": 0.95, "note": "creative"},
    "Brainstorming / ideas":       {"temperature": 1.0, "top_p": 0.9,  "note": "diverse"},
}
print(f"{'Task':<35} {'Temp':>5}  {'Top-P':>6}  Note")
print("-" * 60)
for task, cfg in settings.items():
    print(f"{task:<35} {cfg['temperature']:>5}  {cfg['top_p']:>6}  {cfg['note']}")
