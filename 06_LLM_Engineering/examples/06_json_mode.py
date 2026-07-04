# Author: Mohan Raju Amuri

# ============================================================
# ⚠️  API KEY REQUIRED — COSTS MONEY
# ============================================================
# This example uses OpenAI's JSON mode and structured outputs
# (.parse() with Pydantic) — features specific to the OpenAI API.
# It will NOT run without a valid API key.
#
# Setup:
#   export OPENAI_API_KEY="sk-..."
#   OR create a .env file with: OPENAI_API_KEY=sk-...
#
# Estimated cost to run this file: < $0.001
# Model used: gpt-4o-mini
#
# Why not Ollama? JSON mode and .parse() are OpenAI-specific.
# Local Ollama alternative: prompt the model to output JSON
# and parse manually with json.loads() + Pydantic validation.
# ============================================================

"""
JSON Mode & Structured Output
-------------------------------
One-liner: Force the API to return valid JSON — no more brittle regex parsing of free text.

Two approaches:
  1. JSON mode (response_format={"type": "json_object"})
     → Guarantees valid JSON syntax. You define the schema in the prompt.
  2. Structured outputs (response_format=model schema)  [OpenAI gpt-4o+]
     → Guarantees JSON matches your exact Pydantic schema.

Remember:
- JSON mode: valid JSON guaranteed, but field values can still be wrong/missing
- Structured outputs: schema enforced at token level — can't produce wrong structure
- Always validate with Pydantic even in JSON mode — types may not match
- Include the JSON schema in your prompt when using JSON mode
- temperature=0 + JSON mode = most deterministic structured extraction

Don't:
- Don't use JSON mode without telling the model what schema you want (in the prompt)
- Don't skip Pydantic validation — JSON mode only guarantees syntax, not correctness
- Don't parse with json.loads() without try/except
- Don't expect null for missing fields unless you say so in the prompt
"""

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Optional

client = OpenAI()


# ── 1. JSON mode (works with all gpt-4o-mini + gpt-4o) ───────────────────
print("=== 1. JSON Mode ===")

def extract_json(user_message: str, schema_description: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},   # JSON mode ON
        messages=[
            {"role": "system", "content": f"Extract information as JSON. Schema: {schema_description}"},
            {"role": "user",   "content": user_message},
        ],
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


texts = [
    "John Smith, age 34, works as a senior data scientist at Google since 2019",
    "Maria Garcia is a 28-year-old ML engineer at Anthropic",
    "Bob, 45, product manager",   # missing fields
]

schema = '{"name": string, "age": integer, "role": string, "company": string | null}'
for text in texts:
    result = extract_json(text, schema)
    print(f"  Input : {text[:55]}")
    print(f"  Output: {result}\n")


# ── 2. Structured outputs with Pydantic (gpt-4o + gpt-4o-mini) ────────────
print("=== 2. Structured Outputs (schema-enforced) ===")

class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    remote: bool
    required_skills: list[str]


def extract_job(text: str) -> JobPosting:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract job posting details."},
            {"role": "user",   "content": text},
        ],
        response_format=JobPosting,   # Pydantic model directly
    )
    return response.choices[0].message.parsed


job_text = """
Senior ML Engineer at Stripe — Remote OK
$160,000 - $200,000 annually
Requirements: Python, PyTorch, 5+ years ML experience, MLOps, Docker
"""

job = extract_job(job_text)
print(f"Title    : {job.title}")
print(f"Company  : {job.company}")
print(f"Remote   : {job.remote}")
print(f"Salary   : ${job.salary_min:,} – ${job.salary_max:,}")
print(f"Skills   : {job.required_skills}")


# ── 3. Validation layer ────────────────────────────────────────────────────
print("\n=== 3. Validation with Pydantic ===")

class ProductReview(BaseModel):
    product_name: str
    rating: int          # 1-5
    sentiment: str       # POSITIVE / NEGATIVE / NEUTRAL
    key_points: list[str]

def safe_extract(text: str) -> ProductReview | None:
    try:
        raw = extract_json(text, str(ProductReview.model_json_schema()))
        return ProductReview(**raw)
    except (ValidationError, KeyError) as e:
        print(f"  Validation failed: {e}")
        return None

review = safe_extract(
    "The AirPods Pro are fantastic! Great noise cancellation and battery life. Rating: 5 stars."
)
if review:
    print(f"Product  : {review.product_name}")
    print(f"Rating   : {review.rating}/5")
    print(f"Sentiment: {review.sentiment}")
    print(f"Points   : {review.key_points}")
