"""
Prompt Engineering
-------------------
One-liner: The quality of your prompt directly determines the quality of the output — it IS the engineering.

Techniques (in order of power):
  1. Zero-shot     — just ask
  2. Few-shot      — show examples in the prompt
  3. Chain-of-Thought (CoT) — tell it to "think step by step"
  4. Role prompting — "You are an expert..."
  5. Output format  — "respond as JSON", "use bullet points"

Remember:
- Be specific: vague prompts → vague outputs
- Few-shot examples must match your desired format exactly
- CoT dramatically improves reasoning — add "Think step by step"
- System prompt sets behavior; user prompt is the task
- Constraints reduce hallucination: "answer only from the context below"
- Temperature=0 for deterministic, reproducible prompt testing

Don't:
- Don't write prompts once and assume they generalise — test on edge cases
- Don't put instructions at the end — put them at the START or in system prompt
- Don't use long, complex sentences — clear and direct outperforms verbose
- Don't ask multiple questions in one prompt for structured tasks — split them
- Don't forget: longer prompts = more tokens = higher cost + slower
"""

# NOTE: These examples use a local GPT-2 to demonstrate the CONCEPT.
# In real use, replace with OpenAI/Anthropic API (Module 06).

from transformers import pipeline

gen = pipeline("text-generation", model="gpt2", pad_token_id=50256, max_new_tokens=60)


def generate(prompt: str) -> str:
    result = gen(prompt, do_sample=False, num_return_sequences=1)
    return result[0]["generated_text"][len(prompt):].strip().split("\n")[0]


# ── 1. Zero-shot ──────────────────────────────────────────────────────────
print("=== 1. Zero-Shot ===")
prompt = "Translate to French: 'Good morning, how are you?'\nFrench:"
print(f"Prompt: {prompt}")
print(f"Output: {generate(prompt)}\n")

# ── 2. Few-shot ───────────────────────────────────────────────────────────
print("=== 2. Few-Shot (examples guide format) ===")
prompt = """Classify sentiment as POSITIVE or NEGATIVE.

Review: "I love this product!" → POSITIVE
Review: "Terrible quality, broke in a day." → NEGATIVE
Review: "Best laptop I've ever owned!" → POSITIVE
Review: "Complete waste of money." →"""
print(f"Prompt:\n{prompt}")
print(f"Output: {generate(prompt)}\n")

# ── 3. Chain-of-Thought ───────────────────────────────────────────────────
print("=== 3. Chain-of-Thought ===")
# Without CoT
prompt_no_cot = "If a shirt costs $15 and you buy 3, how much do you spend? Answer:"
# With CoT
prompt_cot = "If a shirt costs $15 and you buy 3, how much do you spend? Think step by step. Answer:"

print(f"Without CoT: {generate(prompt_no_cot)}")
print(f"With CoT   : {generate(prompt_cot)}")

# ── 4. Role Prompting ─────────────────────────────────────────────────────
print("\n=== 4. Role Prompting ===")
prompt = "You are a senior Python engineer. Review this code and find bugs:\n\ndef divide(a, b):\n    return a / b\n\nBugs:"
print(f"Output: {generate(prompt)}\n")

# ── 5. Output Format Control ──────────────────────────────────────────────
print("=== 5. Format Control ===")
prompts = {
    "Bullet list":  "List 3 benefits of exercise:\n-",
    "JSON output":  'Extract: name and age from "Alice is 30 years old". Output as JSON: {"name":',
    "Step by step": "How to make tea:\nStep 1:",
}
for style, prompt in prompts.items():
    out = generate(prompt)
    print(f"  [{style}] {prompt.split(chr(10))[-1]}{out[:60]}")

# ── 6. Prompt anti-patterns ───────────────────────────────────────────────
print("\n=== Common Prompt Mistakes ===")
bad_good = [
    ("Tell me something about AI",
     "List 3 real-world applications of AI in healthcare. Be specific."),
    ("Is Python good?",
     "What are 2 advantages and 1 disadvantage of Python for data science?"),
    ("Summarize this long document...",
     "Summarize the following in 2 sentences, focusing on key findings: [text]"),
]
for bad, good in bad_good:
    print(f"  ✗ Bad : {bad}")
    print(f"  ✓ Good: {good}\n")
