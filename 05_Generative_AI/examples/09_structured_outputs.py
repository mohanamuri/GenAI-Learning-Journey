"""
Structured Outputs
-------------------
One-liner: Force the LLM to return valid JSON/schema instead of free text — makes output parseable and reliable.

Why it matters:
- Free text output requires fragile regex parsing
- JSON mode / function calling guarantees parseable structure
- Reduces hallucination for constrained fields
- Required for tool use, agent workflows, database writes

Methods (in order of reliability):
  1. Prompt instruction ("respond as JSON")     → brittle, often breaks
  2. JSON mode (OpenAI/Anthropic API)           → guaranteed valid JSON
  3. Function calling / tool use                → schema-enforced, most reliable
  4. Outlines / guidance (local models)         → token-level JSON enforcement

Remember:
- Always validate with json.loads() even with JSON mode — LLMs can still be wrong
- Pydantic models are the cleanest way to define + validate expected schema
- Few-shot examples in prompt dramatically improve JSON formatting compliance
- Keep schema simple — deep nesting confuses the model
- Use json.dumps(schema, indent=2) to show the model the expected format

Don't:
- Don't parse LLM JSON with regex — use json.loads() + try/except
- Don't ask for complex nested schemas without examples
- Don't assume JSON mode = correct values, just correct syntax
- Don't skip validation — even valid JSON can have wrong types
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

# ── 1. Prompt-based JSON extraction (brittle but common) ──────────────────
print("=== 1. Prompt-based JSON Extraction ===\n")

# This is what the prompt + model response looks like
simulated_responses = [
    # Good response
    '{"name": "Alice Johnson", "age": 28, "role": "Data Scientist", "skills": ["Python", "ML", "SQL"]}',
    # Bad response (model added extra text — common failure)
    'Here is the JSON:\n{"name": "Bob", "age": 35}\n\nLet me know if you need anything else!',
    # Malformed JSON
    "{'name': 'Charlie', 'age': 30}",  # single quotes — invalid JSON
]

def extract_json(response: str) -> Optional[dict]:
    """Robustly extract JSON from LLM response."""
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    # Try to find JSON block in mixed text
    import re
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

for i, resp in enumerate(simulated_responses, 1):
    result = extract_json(resp)
    print(f"  Response {i}: {result if result else 'FAILED TO PARSE'}")

# ── 2. Schema definition with Pydantic ────────────────────────────────────
print("\n=== 2. Pydantic Schema Validation ===\n")

try:
    from pydantic import BaseModel, ValidationError, field_validator

    class ExtractedPerson(BaseModel):
        name: str
        age: int
        role: str
        skills: list[str]
        email: Optional[str] = None

        @field_validator("age")
        @classmethod
        def age_must_be_positive(cls, v):
            if v < 0 or v > 120:
                raise ValueError("Invalid age")
            return v

    # Simulate validating LLM output
    test_cases = [
        '{"name": "Alice", "age": 28, "role": "Engineer", "skills": ["Python", "Go"]}',
        '{"name": "Bob", "age": -5, "role": "Designer", "skills": []}',       # invalid age
        '{"name": "Charlie", "role": "Manager", "skills": ["Excel"]}',         # missing age
    ]
    for raw in test_cases:
        try:
            data = ExtractedPerson(**json.loads(raw))
            print(f"  ✓ Valid: {data.name}, age={data.age}, role={data.role}")
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"  ✗ Invalid: {str(e).split(chr(10))[0]}")

except ImportError:
    print("  pip install pydantic")

# ── 3. Few-shot prompt for reliable JSON ──────────────────────────────────
print("\n=== 3. Few-Shot Prompt Template for JSON ===")

JSON_EXTRACTION_PROMPT = '''Extract person information as JSON matching this schema:
{{"name": "string", "age": integer, "role": "string", "skills": ["string"]}}

Examples:
Input: "Sarah is a 32-year-old ML engineer who knows Python and TensorFlow"
Output: {{"name": "Sarah", "age": 32, "role": "ML engineer", "skills": ["Python", "TensorFlow"]}}

Input: "Tom, 45, works as a product manager focused on AI products"
Output: {{"name": "Tom", "age": 45, "role": "product manager", "skills": ["AI products"]}}

Input: "{input_text}"
Output:'''

sample_input = "Maria is a 29-year-old data scientist specializing in NLP and transformers"
print(JSON_EXTRACTION_PROMPT.format(input_text=sample_input))

# ── 4. Retry pattern for JSON parsing ────────────────────────────────────
print("\n=== 4. Retry Pattern (production use) ===")

def call_llm_with_json_retry(prompt: str, max_retries: int = 3) -> Optional[dict]:
    """
    In production: call actual LLM API, retry on JSON parse failure.
    Here we simulate the pattern.
    """
    for attempt in range(1, max_retries + 1):
        # response = llm.call(prompt)  ← real API call goes here
        response = '{"name": "Alice", "age": 28}'  # simulated
        result = extract_json(response)
        if result:
            print(f"  Succeeded on attempt {attempt}")
            return result
        print(f"  Attempt {attempt} failed, retrying with stricter prompt...")
        prompt += "\nIMPORTANT: Respond with ONLY valid JSON. No explanation."
    return None

call_llm_with_json_retry("Extract person info from: Alice is 28 years old")
