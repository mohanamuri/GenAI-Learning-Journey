# Author: Mohan Raju Amuri
"""
01_multi_agent_basics.py — Why multi-agent? What changes?

Single agent: one LLM handles everything → bottleneck, context gets polluted
Multi-agent: divide by ROLE (Researcher, Writer, Reviewer) or TASK (run subtasks in parallel)

Key ideas:
  - Each agent has a role + specialty
  - Agents pass outputs to each other
  - The pipeline is more modular and debuggable than one giant prompt

No Ollama required — demonstrates concepts with simulated outputs.
"""

from dataclasses import dataclass, field
from typing import Any


# ── Agent anatomy ─────────────────────────────────────────────────────────────

@dataclass
class Message:
    sender: str
    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"[{self.sender}] {self.content[:80]}"


@dataclass
class AgentResult:
    agent: str
    output: str
    success: bool = True
    steps: int = 1


class BaseAgent:
    """Minimal agent: role, instructions, process()."""

    def __init__(self, name: str, role: str, instructions: str):
        self.name        = name
        self.role        = role
        self.instructions = instructions

    def process(self, message: Message) -> AgentResult:
        raise NotImplementedError

    def __repr__(self):
        return f"Agent({self.name}, role={self.role})"


# ── Specialized agents ────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """Finds relevant facts for a topic."""

    KB = {
        "climate": [
            "Global temperature rose 1.1°C above pre-industrial levels by 2020.",
            "CO₂ concentration exceeded 420 ppm in 2023.",
            "Arctic sea ice extent is declining ~13% per decade.",
        ],
        "ai": [
            "GPT-4 has an estimated 1.76 trillion parameters.",
            "LLM inference costs dropped ~10× between 2022–2024.",
            "Multimodal models now handle text, image, audio in one model.",
        ],
        "default": [
            "No specific KB entry found. Returning general context.",
        ],
    }

    def process(self, message: Message) -> AgentResult:
        topic = message.content.lower()
        key   = next((k for k in self.KB if k in topic), "default")
        facts = self.KB[key]
        output = f"Research findings for '{message.content}':\n" + "\n".join(f"  • {f}" for f in facts)
        print(f"  [{self.name}] found {len(facts)} facts about '{key}'")
        return AgentResult(agent=self.name, output=output, steps=1)


class SummarizerAgent(BaseAgent):
    """Condenses research findings into 2-3 bullet points."""

    def process(self, message: Message) -> AgentResult:
        lines = [l.strip() for l in message.content.split("\n") if l.strip().startswith("•")]
        if not lines:
            lines = [message.content[:200]]
        summary = "Key takeaways:\n" + "\n".join(f"  → {l.lstrip('• ')}" for l in lines[:3])
        print(f"  [{self.name}] condensed to {len(lines[:3])} points")
        return AgentResult(agent=self.name, output=summary, steps=1)


class ReviewerAgent(BaseAgent):
    """Checks quality of the summary and scores it."""

    def process(self, message: Message) -> AgentResult:
        content = message.content
        score   = 0
        issues  = []

        if "Key takeaways" in content:     score += 3
        if len(content) > 50:              score += 2
        if content.count("→") >= 2:        score += 3
        if "error" not in content.lower(): score += 2
        total   = min(score, 10)

        verdict = "✓ Approved" if total >= 7 else "✗ Needs revision"
        output  = f"Review score: {total}/10 — {verdict}\n{content}"
        print(f"  [{self.name}] score={total}/10 → {verdict}")
        return AgentResult(agent=self.name, output=output, success=(total >= 7), steps=1)


# ── Sequential pipeline ────────────────────────────────────────────────────────

class SequentialPipeline:
    """
    Chains agents so each one's output becomes the next one's input.

    Agent A → Agent B → Agent C
    """

    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents

    def run(self, initial_input: str) -> list[AgentResult]:
        print(f"\nPipeline: {' → '.join(a.name for a in self.agents)}")
        print(f"Input: '{initial_input}'\n")

        results = []
        current = Message(sender="user", content=initial_input)

        for agent in self.agents:
            result  = agent.process(current)
            results.append(result)
            current = Message(sender=agent.name, content=result.output)  # chain output → next input

        return results


# ── Why multi-agent? comparison ───────────────────────────────────────────────

def single_agent_demo(topic: str) -> str:
    """Simulates a single agent trying to do everything in one prompt."""
    # One agent gets a huge context: research + summarize + review
    # Easy to get confused, harder to debug, can't parallelize
    return (
        f"[SingleAgent] Processing '{topic}'...\n"
        "  ⚠ All logic in one context → harder to debug\n"
        "  ⚠ Can't swap out just the summarizer\n"
        "  ⚠ Context window fills up with mixed concerns"
    )


def multi_agent_demo(topic: str) -> None:
    """Runs the same task through a specialized 3-agent pipeline."""
    pipeline = SequentialPipeline(agents=[
        ResearchAgent(
            name="Researcher",
            role="research",
            instructions="Find factual information about the topic."
        ),
        SummarizerAgent(
            name="Summarizer",
            role="summarize",
            instructions="Condense findings into key takeaways."
        ),
        ReviewerAgent(
            name="Reviewer",
            role="review",
            instructions="Score quality 0–10 and approve or flag for revision."
        ),
    ])
    results = pipeline.run(topic)
    print("\nFinal output:")
    print(results[-1].output)


# ── Anatomy comparison table ───────────────────────────────────────────────────

def print_comparison():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          Single Agent vs Multi-Agent                        ║
╠══════════════════╦═══════════════════╦═══════════════════════╣
║ Dimension        ║ Single Agent      ║ Multi-Agent           ║
╠══════════════════╬═══════════════════╬═══════════════════════╣
║ Context          ║ One big context   ║ Scoped per agent      ║
║ Specialization   ║ General-purpose   ║ Role-specific         ║
║ Debuggability    ║ Hard (one blob)   ║ Easy (per-agent trace)║
║ Parallelism      ║ None              ║ Independent tasks     ║
║ Failure scope    ║ Full failure      ║ Isolated failures     ║
║ Swap components  ║ Rewrite everything║ Replace one agent     ║
╚══════════════════╩═══════════════════╩═══════════════════════╝
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Multi-Agent Basics — Role-based Agent Pipeline")
    print("=" * 65)

    print_comparison()

    print("── Single agent approach (one context, everything mixed) ──")
    print(single_agent_demo("climate change"))

    print("\n── Multi-agent approach (specialized, chainable) ──")
    multi_agent_demo("climate change")

    print("\n── Second run — different topic ──")
    multi_agent_demo("AI technology trends")

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Multi-agent = divide by role, not by prompt complexity")
    print("  • Each agent has one job → easier to swap, test, debug")
    print("  • Output of agent N becomes input of agent N+1")
    print("  • NOT multi-agent: calling the same LLM twice in a loop")
    print("  • IS multi-agent: different roles, different contexts, chained")


if __name__ == "__main__":
    main()
