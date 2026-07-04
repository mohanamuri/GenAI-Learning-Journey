# Author: Mohan Raju Amuri
"""
Hallucinations
---------------
One-liner: LLMs generate plausible-sounding text, not necessarily true text — they can confidently lie.

What causes hallucinations:
- Training on internet text — which contains errors, speculation, fiction
- Next-token prediction doesn't have a "truth" signal
- The model fills gaps with statistically likely (but wrong) content
- No grounding to external facts at inference time

Types:
  Factual     → wrong facts ("Eiffel Tower is in London")
  Source      → fake citations, invented paper authors
  Temporal    → outdated info presented as current
  Math/logic  → confident but wrong arithmetic
  Context     → answers questions not in the given context

Remember:
- Always ground LLMs with retrieved context (RAG) for factual tasks
- Ask the model to say "I don't know" explicitly in your prompt
- temperature=0 reduces (but doesn't eliminate) hallucinations
- Verify critical output — never ship LLM output to production without validation
- Structured output (JSON schema) reduces hallucination in constrained tasks

Don't:
- Don't use raw LLM for legal, medical, or financial facts without RAG + verification
- Don't assume the model knows its own training cutoff accurately
- Don't ask "is this answer correct?" to the same LLM that generated it
- Don't cite LLM output as a source
"""

# NOTE: Demonstrates hallucination detection patterns.
# Real mitigation requires RAG (Module 07) or API calls (Module 06).

# ── 1. What hallucination looks like ──────────────────────────────────────
print("=== Types of Hallucinations ===\n")

hallucination_examples = [
    {
        "type": "Factual",
        "prompt": "Who invented the telephone?",
        "correct": "Alexander Graham Bell (1876)",
        "hallucinated": "Thomas Edison invented the telephone in 1877",
        "why": "Model conflates Edison (lightbulb) with Bell (telephone)",
    },
    {
        "type": "Citation",
        "prompt": "Give me a research paper on transformers",
        "correct": "'Attention Is All You Need', Vaswani et al., 2017",
        "hallucinated": "'Deep Transformers for NLP', Smith et al., 2019, arXiv:1902.xxxxx",
        "why": "Model fabricates plausible-looking but nonexistent papers",
    },
    {
        "type": "Math",
        "prompt": "What is 17 × 24?",
        "correct": "408",
        "hallucinated": "17 × 24 = 412",
        "why": "LLMs are not calculators — always use tools for math",
    },
    {
        "type": "Context",
        "prompt": "Context: 'John went to Paris'. Q: What did John eat?",
        "correct": "The context doesn't mention what John ate.",
        "hallucinated": "John ate a croissant at a café near the Eiffel Tower.",
        "why": "Model fills gap with 'Paris' stereotypes instead of saying unknown",
    },
]

for ex in hallucination_examples:
    print(f"[{ex['type']}]")
    print(f"  Prompt     : {ex['prompt']}")
    print(f"  Correct    : {ex['correct']}")
    print(f"  Hallucinated: {ex['hallucinated']}")
    print(f"  Why        : {ex['why']}\n")

# ── 2. Mitigation techniques ──────────────────────────────────────────────
print("=== Mitigation Techniques ===\n")

mitigations = {
    "Grounding prompt": (
        "WRONG: 'What is the capital of France?'\n"
        "RIGHT: 'Based ONLY on the context below, answer. If not in context, say \"I don't know\".\n"
        "Context: [your document here]'"
    ),
    "Force uncertainty": (
        "Add to system prompt: 'If you are not certain, say \"I'm not sure\" rather than guessing.'"
    ),
    "Temperature=0": (
        "Use temperature=0 for factual tasks — reduces (not eliminates) hallucination."
    ),
    "RAG": (
        "Retrieve relevant documents first, inject into context → model answers from facts, not memory."
        " (covered in Module 07)"
    ),
    "Structured output": (
        "Use JSON schema / function calling → model can't add fields it wasn't asked for."
    ),
    "Self-consistency": (
        "Generate the same answer N times at temperature>0, take majority vote."
    ),
}

for technique, description in mitigations.items():
    print(f"  [{technique}]")
    print(f"  {description}\n")

# ── 3. Simple hallucination detector (context grounding check) ────────────
print("=== Simple Context Grounding Check ===")

def check_answer_grounded(answer: str, context: str, threshold: float = 0.3) -> bool:
    """
    Rough heuristic: does the answer use words from the context?
    Real implementation would use NLI model or embedding similarity.
    """
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    overlap = answer_words & context_words
    ratio = len(overlap) / max(len(answer_words), 1)
    return ratio >= threshold

context = "Python was created by Guido van Rossum and released in 1991."
grounded_answer = "Guido van Rossum created Python."
hallucinated_answer = "Python was invented by Dennis Ritchie at Bell Labs in 1972."

print(f"Context: {context}")
print(f"Grounded answer   : '{grounded_answer}' → grounded={check_answer_grounded(grounded_answer, context)}")
print(f"Hallucinated answer: '{hallucinated_answer}' → grounded={check_answer_grounded(hallucinated_answer, context)}")
print("\nNote: Production grounding uses NLI models or embedding similarity, not word overlap.")
