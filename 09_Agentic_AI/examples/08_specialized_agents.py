# Author: Mohan Raju Amuri
"""
08_specialized_agents.py — Role-based agent pipeline

Research → Write → Review → Edit

Each agent has a distinct role with a clear input/output contract:
  ResearchAgent  : topic → structured findings
  WriterAgent    : findings → draft article
  ReviewerAgent  : draft → score + specific feedback
  EditorAgent    : draft + feedback → polished final

This is the pattern used by AI writing tools (Jasper, Copy.ai, etc.)
and CrewAI's crew of specialized agents.

Runs with Ollama (llama3.2:3b) or falls back to mock content.
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Optional
# ============================================================
# Falls back to mock content if Ollama is not running.
# ============================================================

import json
from dataclasses import dataclass, field
from typing import Optional


# ── Shared data contract ──────────────────────────────────────────────────────

@dataclass
class ResearchFindings:
    topic:     str
    facts:     list[str]
    sources:   list[str] = field(default_factory=list)

@dataclass
class Draft:
    title:    str
    body:     str
    word_count: int = 0
    def __post_init__(self):
        self.word_count = len(self.body.split())

@dataclass
class Review:
    score:      int          # 0–10
    strengths:  list[str]
    weaknesses: list[str]
    passed:     bool = False
    def __post_init__(self):
        self.passed = self.score >= 7

@dataclass
class FinalArticle:
    title:    str
    body:     str
    metadata: dict = field(default_factory=dict)


# ── LLM helper ───────────────────────────────────────────────────────────────

def _init_llm():
    try:
        from openai import OpenAI
        c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        c.models.list()
        return c
    except Exception:
        return None


def llm_call(client, system: str, user: str, max_tokens: int = 400) -> str:
    if client is None:
        return ""
    try:
        r = client.chat.completions.create(
            model="llama3.2:3b",
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.4, max_tokens=max_tokens,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"    LLM error: {e}")
        return ""


# ── Agent 1: ResearchAgent ────────────────────────────────────────────────────

class ResearchAgent:
    """
    Gathers facts and structures them as ResearchFindings.
    In production: web search, KB lookup, document retrieval.
    """
    name = "ResearchAgent"

    KB = {
        "techcorp pricing": {
            "facts": [
                "Starter plan: $99/month, up to 5 users, 100 GB storage, CPU training",
                "Pro plan: $499/month, up to 25 users, 1 TB storage, GPU training",
                "Enterprise: custom pricing, unlimited users, dedicated infra",
                "14-day free trial with full Pro features — no credit card required",
            ],
            "sources": ["techcorp.com/pricing", "techcorp.com/faq"],
        },
        "techcorp security": {
            "facts": [
                "SOC 2 Type II certified",
                "AES-256 encryption at rest, TLS 1.3 in transit",
                "Annual third-party penetration testing",
                "GDPR and CCPA compliant",
            ],
            "sources": ["techcorp.com/security", "techcorp.com/compliance"],
        },
    }

    def __init__(self, llm=None):
        self.llm = llm

    def research(self, topic: str) -> ResearchFindings:
        print(f"  [{self.name}] researching: '{topic}'")
        key  = topic.lower()
        data = next((v for k, v in self.KB.items() if k in key or key in k), None)

        if data:
            facts   = data["facts"]
            sources = data["sources"]
        elif self.llm:
            raw   = llm_call(self.llm,
                system="You are a research assistant. Return 3-5 key facts as a JSON list of strings.",
                user=f"Research topic: {topic}",
                max_tokens=300)
            try:
                import re
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                facts = json.loads(m.group()) if m else [f"General information about {topic}"]
            except Exception:
                facts = [f"Fact about {topic}: information available on request"]
            sources = ["llm-generated"]
        else:
            facts   = [f"Key fact about {topic} — see official documentation"]
            sources = ["mock"]

        print(f"    Found {len(facts)} facts")
        return ResearchFindings(topic=topic, facts=facts, sources=sources)


# ── Agent 2: WriterAgent ───────────────────────────────────────────────────────

class WriterAgent:
    """
    Converts ResearchFindings into a structured draft.
    Format: title + intro + bullet points + conclusion.
    """
    name = "WriterAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def write(self, findings: ResearchFindings) -> Draft:
        print(f"  [{self.name}] drafting article on '{findings.topic}'")

        if self.llm:
            facts_str = "\n".join(f"- {f}" for f in findings.facts)
            raw = llm_call(self.llm,
                system="You are a technical writer. Write a concise, informative article (150-200 words).",
                user=f"Topic: {findings.topic}\n\nFacts:\n{facts_str}",
                max_tokens=400)
            if raw:
                title = findings.topic.title()
                return Draft(title=title, body=raw)

        # Mock draft
        bullets = "\n".join(f"• {f}" for f in findings.facts)
        body = (
            f"Overview of {findings.topic.title()}\n\n"
            f"Here are the key facts you need to know:\n\n"
            f"{bullets}\n\n"
            f"For more information, visit: {', '.join(findings.sources[:2])}."
        )
        draft = Draft(title=findings.topic.title(), body=body)
        print(f"    Draft: {draft.word_count} words")
        return draft


# ── Agent 3: ReviewerAgent ────────────────────────────────────────────────────

class ReviewerAgent:
    """
    Evaluates a draft on: completeness, clarity, accuracy cues, length.
    Returns a Review with score and specific feedback.
    """
    name = "ReviewerAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def review(self, draft: Draft, findings: ResearchFindings) -> Review:
        print(f"  [{self.name}] reviewing '{draft.title}'")

        score      = 0
        strengths  = []
        weaknesses = []

        # Length check
        if draft.word_count >= 100:
            score += 2; strengths.append("Good length (≥100 words)")
        else:
            weaknesses.append(f"Too short ({draft.word_count} words, need ≥100)")

        # Fact coverage
        covered = sum(1 for f in findings.facts if any(w in draft.body.lower()
                       for w in f.lower().split()[:3]))
        coverage = covered / max(len(findings.facts), 1)
        if coverage >= 0.6:
            score += 3; strengths.append(f"Good fact coverage ({covered}/{len(findings.facts)} facts)")
        else:
            weaknesses.append(f"Low fact coverage ({covered}/{len(findings.facts)} facts mentioned)")

        # Has numbers / specific data
        import re
        if re.search(r"\$\d+|\d+%|\d+\s*users", draft.body):
            score += 2; strengths.append("Contains specific figures")
        else:
            weaknesses.append("Missing specific numbers/prices")

        # Has sources
        if any(s in draft.body for s in findings.sources):
            score += 1; strengths.append("Sources cited")
        else:
            weaknesses.append("No sources cited")

        # Structure
        if "\n" in draft.body and ("•" in draft.body or ":" in draft.body):
            score += 2; strengths.append("Well structured with bullet points")
        else:
            weaknesses.append("Add bullet points or headers for clarity")

        r = Review(score=min(score, 10), strengths=strengths, weaknesses=weaknesses)
        verdict = "✓ Passed" if r.passed else "✗ Failed"
        print(f"    Score: {r.score}/10 — {verdict}")
        return r


# ── Agent 4: EditorAgent ──────────────────────────────────────────────────────

class EditorAgent:
    """
    Applies reviewer feedback to produce a polished final article.
    """
    name = "EditorAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def edit(self, draft: Draft, review: Review, findings: ResearchFindings) -> FinalArticle:
        print(f"  [{self.name}] editing based on {len(review.weaknesses)} weakness(es)")

        if self.llm and review.weaknesses:
            feedback = "\n".join(f"- {w}" for w in review.weaknesses)
            raw = llm_call(self.llm,
                system="You are an editor. Revise the draft based on the feedback provided.",
                user=f"Draft:\n{draft.body}\n\nFeedback:\n{feedback}",
                max_tokens=400)
            if raw:
                return FinalArticle(
                    title=draft.title,
                    body=raw,
                    metadata={"editor": self.name, "score": review.score, "revisions": len(review.weaknesses)}
                )

        # Mock: patch weaknesses
        body = draft.body
        if "No sources cited" in review.weaknesses:
            body += f"\n\nSources: {', '.join(findings.sources)}"
        if "Missing specific numbers" in " ".join(review.weaknesses):
            extra = " | ".join(findings.facts[:2])
            body += f"\n\nKey figures: {extra}"

        return FinalArticle(
            title=draft.title,
            body=body,
            metadata={"editor": self.name, "score": review.score, "revisions": len(review.weaknesses)}
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ContentPipeline:
    """
    Chains Research → Write → Review → Edit into one pipeline.
    If review passes, skips editing step.
    """

    def __init__(self):
        llm = _init_llm()
        status = "Ollama connected" if llm else "mock mode"
        print(f"  ContentPipeline: {status}")
        self.researcher = ResearchAgent(llm)
        self.writer     = WriterAgent(llm)
        self.reviewer   = ReviewerAgent(llm)
        self.editor     = EditorAgent(llm)

    def run(self, topic: str) -> FinalArticle:
        print(f"\n  Pipeline: '{topic}'")
        print("  " + "─" * 55)

        findings = self.researcher.research(topic)
        draft    = self.writer.write(findings)
        review   = self.reviewer.review(draft, findings)

        if review.passed:
            print(f"  [Pipeline] review passed — skipping edit step")
            article = FinalArticle(title=draft.title, body=draft.body,
                                   metadata={"score": review.score, "edited": False})
        else:
            article = self.editor.edit(draft, review, findings)

        return article


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Specialized Agent Pipeline — Research → Write → Review → Edit")
    print("=" * 65)

    pipeline = ContentPipeline()

    topics = [
        "TechCorp pricing",
        "TechCorp security",
    ]

    for topic in topics:
        article = pipeline.run(topic)
        print(f"\n  ── Final Article: '{article.title}' ──")
        print(f"  {article.body[:300]}")
        print(f"  Metadata: {article.metadata}")

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Each agent = one responsibility, clear input/output type")
    print("  • Data flows typed: ResearchFindings → Draft → Review → FinalArticle")
    print("  • Skip edit step if review already passes (efficiency)")
    print("  • This is exactly how CrewAI's 'crew with roles' works")
    print("  • LLM can replace each mock agent independently")
    print("  • Easy to add a new agent (e.g. SEO optimizer) at any position")


if __name__ == "__main__":
    main()
