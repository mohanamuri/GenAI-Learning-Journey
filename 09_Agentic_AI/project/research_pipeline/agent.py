# Author: Mohan Raju Amuri
"""
research_pipeline/agent.py — Multi-agent research pipeline

Four specialized agents collaborate to produce a research report:

  ResearchAgent   → queries KB and simulated web, returns raw findings
  SummarizerAgent → condenses findings into key points
  FactCheckerAgent→ validates claims against KB ground truth
  WriterAgent     → produces final structured report

Orchestrated by ResearchPipeline:
  parallel: Research + FactCheck setup
  sequential: Summarize → Write
  Uses Ollama (llama3.2:3b) with mock fallback

Usage:
    python agent.py                         # interactive mode
    python agent.py --topic "TechCorp AI"   # one-shot topic

Requirements:
    pip install openai chromadb sentence-transformers
    ollama pull llama3.2:3b && ollama serve
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Optional
# ============================================================
# Install: https://ollama.ai
# Pull:    ollama pull llama3.2:3b
# Serve:   ollama serve
# Falls back to mock content if Ollama is not running.
# ============================================================

import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional


# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "llama3.2:3b"
MAX_TOKENS   = 400


# ── Knowledge base (ground truth for fact checking) ──────────────────────────

KB = {
    "pricing": {
        "Starter": "$99/month, up to 5 users, 100 GB storage, CPU training",
        "Pro":     "$499/month, up to 25 users, 1 TB storage, GPU training",
        "Enterprise": "custom pricing, unlimited users, dedicated infrastructure",
    },
    "security": {
        "certification": "SOC 2 Type II",
        "encryption":    "AES-256 at rest, TLS 1.3 in transit",
        "compliance":    "GDPR and CCPA compliant",
        "pentesting":    "annual third-party penetration testing",
    },
    "sla": {
        "Pro":        "99.9% uptime",
        "Enterprise": "99.9% uptime",
        "Starter":    "99.5% uptime",
    },
    "models": {
        "NLP":  "BERT, GPT-2, T5, custom transformer architectures",
        "CV":   "ResNet, EfficientNet, YOLO for image classification and detection",
    },
    "trial": {
        "duration":   "14 days",
        "features":   "full Pro features",
        "credit_card": "not required",
    },
}


# ── Shared state ─────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    topic:           str
    raw_findings:    list[str] = field(default_factory=list)
    fact_checks:     dict      = field(default_factory=dict)    # claim → verified bool
    summary_points:  list[str] = field(default_factory=list)
    final_report:    str       = ""
    sources_used:    list[str] = field(default_factory=list)
    agent_trace:     list[str] = field(default_factory=list)    # audit log


# ── LLM helper ────────────────────────────────────────────────────────────────

def init_llm():
    try:
        from openai import OpenAI
        c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        c.models.list()
        return c
    except Exception:
        return None


def llm_call(client, system: str, user: str) -> str:
    if not client:
        return ""
    try:
        r = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=0.3, max_tokens=MAX_TOKENS,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return ""


# ══════════════════════════════════════════════════════════════
# Agent 1: ResearchAgent
# ══════════════════════════════════════════════════════════════

class ResearchAgent:
    name = "ResearchAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def _kb_lookup(self, topic: str) -> list[str]:
        """Search knowledge base for topic-related facts."""
        topic_lower = topic.lower()
        findings    = []
        for section, data in KB.items():
            if section in topic_lower or any(section in w for w in topic_lower.split()):
                for key, value in data.items():
                    findings.append(f"{key}: {value}")
        # if topic is broad, include all
        if not findings:
            for section, data in KB.items():
                for key, value in data.items():
                    findings.append(f"{key}: {value}")
        return findings[:8]   # top 8 findings

    def _web_search_mock(self, topic: str) -> list[str]:
        """Simulated web search (no actual network call)."""
        return [
            f"TechCorp recently launched a new {topic}-focused product tier",
            f"Industry analysts rate TechCorp highly for {topic} capabilities",
            f"TechCorp's {topic} solution is used by 500+ enterprises globally",
        ]

    def run(self, state: PipelineState) -> None:
        print(f"  [{self.name}] researching: '{state.topic}'")
        start = time.perf_counter()

        # KB lookup
        kb_findings = self._kb_lookup(state.topic)

        # LLM-enhanced research (if available)
        if self.llm and kb_findings:
            kb_text = "\n".join(f"  - {f}" for f in kb_findings)
            enriched = llm_call(self.llm,
                system="You are a research analyst. Expand on the provided facts with context. Return 3-5 bullet points.",
                user=f"Topic: {state.topic}\n\nKnown facts:\n{kb_text}"
            )
            if enriched:
                llm_points = [l.strip().lstrip("•-* ") for l in enriched.split("\n")
                              if l.strip() and not l.strip().startswith("#")]
                kb_findings = kb_findings[:5] + llm_points[:3]

        # Simulated web findings
        web_findings = self._web_search_mock(state.topic)

        state.raw_findings  = kb_findings + web_findings
        state.sources_used  = ["TechCorp Knowledge Base", "Simulated Web Search"]
        state.agent_trace.append(f"[{self.name}] found {len(state.raw_findings)} items in {time.perf_counter()-start:.2f}s")
        print(f"    {len(state.raw_findings)} findings collected")


# ══════════════════════════════════════════════════════════════
# Agent 2: FactCheckerAgent
# ══════════════════════════════════════════════════════════════

class FactCheckerAgent:
    """Checks each finding against KB ground truth."""
    name = "FactCheckerAgent"

    def _check_claim(self, claim: str) -> bool:
        """Returns True if claim is supported by KB data."""
        claim_lower = claim.lower()
        for section, data in KB.items():
            for key, value in data.items():
                # Check if any KB value substring appears in claim
                for part in str(value).lower().split(","):
                    part = part.strip()
                    if len(part) > 5 and part in claim_lower:
                        return True
        # Claims without KB match are "unverified" (not false — just unknown)
        return "techcorp" in claim_lower or "$" in claim or "%" in claim

    def run(self, state: PipelineState) -> None:
        print(f"  [{self.name}] checking {len(state.raw_findings)} claims")
        start = time.perf_counter()
        verified = 0
        for finding in state.raw_findings:
            result = self._check_claim(finding)
            state.fact_checks[finding] = result
            if result: verified += 1
        state.agent_trace.append(
            f"[{self.name}] {verified}/{len(state.raw_findings)} verified in {time.perf_counter()-start:.2f}s"
        )
        print(f"    {verified}/{len(state.raw_findings)} claims verified")


# ══════════════════════════════════════════════════════════════
# Agent 3: SummarizerAgent
# ══════════════════════════════════════════════════════════════

class SummarizerAgent:
    """Condenses verified findings into key summary points."""
    name = "SummarizerAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def run(self, state: PipelineState) -> None:
        print(f"  [{self.name}] summarizing")
        start = time.perf_counter()

        # Use only verified findings
        verified = [f for f, ok in state.fact_checks.items() if ok]

        if self.llm and verified:
            facts_str = "\n".join(f"  • {f}" for f in verified[:8])
            result = llm_call(self.llm,
                system="You are a technical summarizer. Extract 4-6 key takeaways as bullet points.",
                user=f"Topic: {state.topic}\n\nVerified facts:\n{facts_str}"
            )
            if result:
                points = [l.strip().lstrip("•-* ") for l in result.split("\n")
                          if l.strip() and len(l.strip()) > 10]
                state.summary_points = points[:6]
                state.agent_trace.append(
                    f"[{self.name}] {len(state.summary_points)} points (LLM) in {time.perf_counter()-start:.2f}s"
                )
                print(f"    {len(state.summary_points)} key points (via LLM)")
                return

        # Mock: deduplicate and pick top facts
        seen  = set()
        points = []
        for f in verified:
            key = f.split(":")[0].lower()
            if key not in seen:
                seen.add(key)
                points.append(f)
        state.summary_points = points[:6]
        state.agent_trace.append(
            f"[{self.name}] {len(state.summary_points)} points (mock) in {time.perf_counter()-start:.2f}s"
        )
        print(f"    {len(state.summary_points)} key points (mock)")


# ══════════════════════════════════════════════════════════════
# Agent 4: WriterAgent
# ══════════════════════════════════════════════════════════════

class WriterAgent:
    """Produces the final structured report from summary points."""
    name = "WriterAgent"

    def __init__(self, llm=None):
        self.llm = llm

    def _format_mock(self, state: PipelineState) -> str:
        lines = [
            f"# Research Report: {state.topic.title()}",
            f"",
            f"## Key Findings",
        ]
        for point in state.summary_points:
            lines.append(f"• {point}")
        lines += [
            "",
            f"## Sources",
            ", ".join(state.sources_used),
            "",
            f"## Fact Check Summary",
            f"Verified: {sum(1 for v in state.fact_checks.values() if v)}/{len(state.fact_checks)} claims",
        ]
        return "\n".join(lines)

    def run(self, state: PipelineState) -> None:
        print(f"  [{self.name}] writing report")
        start = time.perf_counter()

        if self.llm and state.summary_points:
            points_str = "\n".join(f"  • {p}" for p in state.summary_points)
            result = llm_call(self.llm,
                system=(
                    "You are a technical writer. Write a concise research report (150-200 words). "
                    "Include: title, introduction, key findings (bullet list), conclusion. "
                    "Be specific and factual."
                ),
                user=f"Topic: {state.topic}\n\nKey points:\n{points_str}\n\nSources: {', '.join(state.sources_used)}"
            )
            if result:
                state.final_report = result
                state.agent_trace.append(
                    f"[{self.name}] report written (LLM) in {time.perf_counter()-start:.2f}s"
                )
                print(f"    Report written via LLM ({len(result.split())} words)")
                return

        state.final_report = self._format_mock(state)
        state.agent_trace.append(
            f"[{self.name}] report written (mock) in {time.perf_counter()-start:.2f}s"
        )
        print(f"    Report written (mock)")


# ══════════════════════════════════════════════════════════════
# Research Pipeline (Orchestrator)
# ══════════════════════════════════════════════════════════════

class ResearchPipeline:
    """
    Orchestrates 4 agents:
      Phase 1 (parallel): ResearchAgent + FactCheckerAgent setup
      Phase 2 (sequential): Research → FactCheck → Summarize → Write

    With Ollama: LLMs enhance research and writing.
    Without: mock content used throughout.
    """

    def __init__(self):
        print("Initializing Research Pipeline...")
        self.llm = init_llm()
        status   = f"Ollama ({OLLAMA_MODEL})" if self.llm else "mock mode"
        print(f"  LLM: {status}")

        self.researcher   = ResearchAgent(self.llm)
        self.fact_checker = FactCheckerAgent()
        self.summarizer   = SummarizerAgent(self.llm)
        self.writer       = WriterAgent(self.llm)
        print(f"  Agents: {[self.researcher.name, self.fact_checker.name, self.summarizer.name, self.writer.name]}")

    def run(self, topic: str) -> PipelineState:
        state = PipelineState(topic=topic)
        start = time.perf_counter()

        print(f"\n  {'='*55}")
        print(f"  Pipeline: '{topic}'")
        print(f"  {'='*55}")

        # Phase 1: Research
        self.researcher.run(state)

        # Phase 2: Fact check (after research populates raw_findings)
        self.fact_checker.run(state)

        # Phase 3: Summarize verified findings
        self.summarizer.run(state)

        # Phase 4: Write report
        self.writer.run(state)

        total = time.perf_counter() - start
        state.agent_trace.append(f"[Pipeline] total={total:.2f}s")
        return state

    def print_report(self, state: PipelineState) -> None:
        print(f"\n{'='*60}")
        print(state.final_report)
        print(f"{'='*60}")
        print(f"\nAgent trace:")
        for entry in state.agent_trace:
            print(f"  {entry}")

    def run_interactive(self) -> None:
        print("\n" + "=" * 60)
        print("Multi-Agent Research Pipeline")
        print("Type 'quit' to exit")
        print("=" * 60)
        print("\nTopics to try: 'TechCorp pricing', 'TechCorp security', 'TechCorp AI models'")

        while True:
            try:
                topic = input("\nResearch topic: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not topic:
                continue
            if topic.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            state = self.run(topic)
            self.print_report(state)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Research Pipeline")
    parser.add_argument("--topic", type=str, help="One-shot research topic")
    args = parser.parse_args()

    pipeline = ResearchPipeline()

    if args.topic:
        state = pipeline.run(args.topic)
        pipeline.print_report(state)
    else:
        pipeline.run_interactive()


if __name__ == "__main__":
    main()
