# Author: Mohan Raju Amuri
"""
Prompt Templates
-----------------
One-liner: Templates separate prompt structure from dynamic values — reusable, testable, versionable.

Remember:
- Never build prompts with manual string concatenation in production
- Templates make prompts: testable, versionable, reusable across models
- Variables in templates should be clearly named and validated before injection
- Prompt templates ARE code — treat them like functions, not strings

Don't:
- Don't use f-strings for complex prompts — hard to test and audit
- Don't put dynamic user input directly into system prompts — injection risk
- Don't let templates grow unbounded — keep them focused and short
- Don't duplicate similar prompts — parameterize instead

Running on Ollama (local, free — no API key needed):
  ollama pull llama3.2:3b
"""

from string import Template
from dataclasses import dataclass
from typing import Any
from openai import OpenAI

# ── Using Ollama (local LLM) — free, no API key required ──────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "llama3.2:3b"


# ── 1. Basic template with string.Template ────────────────────────────────
print("=== 1. Basic Template ===")

# string.Template uses $variable syntax — safe against injection (no code execution)
SUMMARIZE_TEMPLATE = Template("""
Summarize the following $content_type in $num_sentences sentences.
Focus on: $focus

Content:
$content
""".strip())

prompt = SUMMARIZE_TEMPLATE.substitute(
    content_type="article",
    num_sentences="2",
    focus="key findings and practical implications",
    content="Machine learning is transforming how companies process data. "
            "Models trained on large datasets can now automate complex decisions "
            "that previously required human expertise.",
)
print("Rendered prompt:")
print(prompt)
print("\nResponse:", client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=100,
).choices[0].message.content.strip())


# ── 2. Reusable prompt template class ─────────────────────────────────────
# This class adds validation — fails loudly if a required variable is missing.
# In production, treat this like a function with typed parameters.
print("\n=== 2. PromptTemplate class ===")

@dataclass
class PromptTemplate:
    """Reusable, validated prompt template."""
    template: str
    required_vars: list[str]

    def render(self, **kwargs: Any) -> str:
        # Validate before rendering — fail fast on missing inputs
        missing = [v for v in self.required_vars if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        return self.template.format(**kwargs)

    def __call__(self, **kwargs: Any) -> str:
        return self.render(**kwargs)


# Store templates as module-level constants — version control these
CLASSIFY_TEMPLATE = PromptTemplate(
    template="""Classify the following {content_type} into one of these categories: {categories}.

Text: {text}

Respond with only the category name.""",
    required_vars=["content_type", "categories", "text"],
)

EXTRACT_TEMPLATE = PromptTemplate(
    template="""Extract {fields} from the text below.
Return as JSON. Use null for missing values.

Text: {text}""",
    required_vars=["fields", "text"],
)


def llm(prompt: str) -> str:
    return client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    ).choices[0].message.content.strip()


# Templates make it easy to reuse the same structure across many inputs
texts = [
    "The stock market surged 2% today on strong earnings reports",
    "Scientists discover a new species of deep-sea fish near the Pacific",
    "The home team won the championship after an intense final match",
]
for text in texts:
    prompt = CLASSIFY_TEMPLATE(
        content_type="news headline",
        categories="Finance, Science, Sports, Technology, Politics",
        text=text,
    )
    print(f"  [{llm(prompt)}] {text[:55]}")


# ── 3. Few-shot template ───────────────────────────────────────────────────
# Few-shot examples in the template dramatically improve output consistency.
# The model learns the format from examples, not just instructions.
print("\n=== 3. Few-Shot Template ===")

FEW_SHOT_TEMPLATE = PromptTemplate(
    template="""Task: {task}

Examples:
{examples}

Now do the same for:
Input: {input}
Output:""",
    required_vars=["task", "examples", "input"],
)

examples = "\n".join([
    "Input: I love this! -> Sentiment: POSITIVE",
    "Input: Terrible service -> Sentiment: NEGATIVE",
    "Input: It's okay I guess -> Sentiment: NEUTRAL",
])

for test in ["Best product ever!", "Won't be buying again", "Arrived on time"]:
    prompt = FEW_SHOT_TEMPLATE(
        task="Classify sentiment as POSITIVE, NEGATIVE, or NEUTRAL",
        examples=examples,
        input=test,
    )
    print(f"  {test!r:<30} → {llm(prompt)}")


# ── 4. Template best practices ────────────────────────────────────────────
print("\n=== Best Practices ===")
tips = [
    "Store templates as constants at module level — not inside functions",
    "Use required_vars validation — fail fast on missing inputs",
    "Name variables descriptively: {user_query} not {q}",
    "Version your templates: SUMMARIZE_V2 = ...",
    "Test templates with edge cases: empty input, very long input, special chars",
]
for tip in tips:
    print(f"  • {tip}")
