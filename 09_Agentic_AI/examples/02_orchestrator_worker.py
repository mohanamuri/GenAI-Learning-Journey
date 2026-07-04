# Author: Mohan Raju Amuri
"""
02_orchestrator_worker.py — Orchestrator/Worker pattern

Orchestrator: receives high-level task → uses LLM to decompose into subtasks
Worker:       receives one subtask → executes it → returns result
Orchestrator: collects all worker results → synthesizes final answer

Used in: LangGraph, AutoGen, CrewAI, GPT-4 Code Interpreter

Why this pattern?
  - Orchestrator stays high-level, never does grunt work
  - Workers are specialized and swappable
  - Tasks can be distributed / parallelized across workers

Runs with Ollama (llama3.2:3b) or falls back to mock decomposition.
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Optional
# ============================================================
# Install: https://ollama.ai
# Pull:    ollama pull llama3.2:3b
# Serve:   ollama serve
# Falls back to mock if Ollama is not running.
# ============================================================

import json
import re
from dataclasses import dataclass, field
from typing import Callable


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    id:          int
    description: str
    worker_type: str          # which worker class handles this
    result:      str  = ""
    done:        bool = False


@dataclass
class OrchestratorPlan:
    goal:     str
    subtasks: list[SubTask] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", f"Subtasks ({len(self.subtasks)}):"]
        for st in self.subtasks:
            status = "✓" if st.done else "○"
            lines.append(f"  {status} [{st.id}] [{st.worker_type}] {st.description}")
        return "\n".join(lines)


# ── Worker agents ─────────────────────────────────────────────────────────────

class ResearchWorker:
    """Looks up factual information."""
    name = "research"

    FACTS = {
        "pricing":    "Starter $99/mo, Pro $499/mo, Enterprise custom.",
        "security":   "SOC 2 Type II, AES-256 at rest, TLS 1.3 in transit.",
        "sla":        "99.9% uptime for Pro/Enterprise; 99.5% for Starter.",
        "support":    "Pro: email + chat, 24h SLA. Enterprise: 24/7 Slack + phone.",
        "trial":      "14-day free trial, full Pro features, no credit card.",
    }

    def run(self, task: str) -> str:
        key = next((k for k in self.FACTS if k in task.lower()), None)
        if key:
            return self.FACTS[key]
        return f"Research result: general information about '{task}'"


class CalculatorWorker:
    """Performs numeric calculations."""
    name = "calculator"

    import math as _math

    def run(self, task: str) -> str:
        import math
        nums = re.findall(r"\d+(?:\.\d+)?", task)
        if len(nums) >= 2:
            a, b = float(nums[0]), float(nums[1])
            if "annual" in task.lower() or "year" in task.lower():
                return f"{a} × 12 = {a * 12:,.2f} per year"
            if "%" in task or "percent" in task.lower():
                return f"{a}% of {b} = {a * b / 100:.2f}"
            if "×" in task or "multiply" in task.lower() or "*" in task:
                return f"{a} × {b} = {a * b:,.2f}"
            return f"{a} + {b} = {a + b:,.2f}"
        return f"Could not parse numbers from: '{task}'"


class SummaryWorker:
    """Writes a concise summary from collected information."""
    name = "summary"

    def run(self, task: str) -> str:
        # In production: call LLM with gathered context
        return (
            f"Summary of '{task[:50]}':\n"
            "  Based on the research and calculations above, here are the key points "
            "the user needs. Full details are available in the individual results."
        )


class FormatterWorker:
    """Formats results for presentation."""
    name = "formatter"

    def run(self, task: str) -> str:
        return f"[Formatted] {task.strip()}"


WORKER_REGISTRY: dict[str, object] = {
    "research":   ResearchWorker(),
    "calculator": CalculatorWorker(),
    "summary":    SummaryWorker(),
    "formatter":  FormatterWorker(),
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Decomposes a high-level goal into subtasks and dispatches to workers.

    With Ollama: LLM generates the decomposition plan as JSON.
    Without Ollama: mock decomposition for demo.
    """

    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        try:
            from openai import OpenAI
            c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            c.models.list()
            print("  Orchestrator: connected to Ollama")
            return c
        except Exception:
            print("  Orchestrator: Ollama not running → using mock decomposition")
            return None

    def _decompose_with_llm(self, goal: str) -> list[SubTask]:
        """Ask LLM to break the goal into typed subtasks."""
        system = (
            "You are a task orchestrator. Given a user goal, decompose it into subtasks.\n"
            "Available worker types: research, calculator, summary, formatter.\n"
            "Return a JSON array of objects: [{\"id\": 1, \"description\": \"...\", \"worker_type\": \"...\"}]\n"
            "Return ONLY the JSON array, no other text."
        )
        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Goal: {goal}"},
                ],
                temperature=0.1, max_tokens=300,
            )
            raw = resp.choices[0].message.content.strip()
            # extract JSON array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                items = json.loads(m.group())
                return [SubTask(id=i["id"], description=i["description"], worker_type=i.get("worker_type","research"))
                        for i in items]
        except Exception as e:
            print(f"  LLM decomposition failed: {e} → using mock")
        return []

    def _decompose_mock(self, goal: str) -> list[SubTask]:
        """Deterministic decomposition for common goals (no LLM needed)."""
        g = goal.lower()
        if "pricing" in g or "plan" in g or "cost" in g:
            return [
                SubTask(1, "Find TechCorp pricing plans",        "research"),
                SubTask(2, "Calculate annual cost of Pro plan",   "calculator"),
                SubTask(3, "Summarize pricing options for user",  "summary"),
            ]
        if "security" in g or "compliance" in g:
            return [
                SubTask(1, "Look up security certifications",     "research"),
                SubTask(2, "Look up SLA guarantees",              "research"),
                SubTask(3, "Summarize security and compliance",   "summary"),
            ]
        if "trial" in g or "try" in g or "free" in g:
            return [
                SubTask(1, "Find trial information",              "research"),
                SubTask(2, "Summarize how to get started",        "summary"),
            ]
        # default
        return [
            SubTask(1, f"Research: {goal}",                       "research"),
            SubTask(2, f"Summarize findings for: {goal}",         "summary"),
        ]

    def decompose(self, goal: str) -> OrchestratorPlan:
        print(f"\n[Orchestrator] Decomposing: '{goal}'")
        if self.llm:
            subtasks = self._decompose_with_llm(goal)
        else:
            subtasks = []

        if not subtasks:
            subtasks = self._decompose_mock(goal)

        plan = OrchestratorPlan(goal=goal, subtasks=subtasks)
        print(f"[Orchestrator] Plan created: {len(subtasks)} subtasks")
        return plan

    def dispatch(self, plan: OrchestratorPlan) -> OrchestratorPlan:
        """Send each subtask to the appropriate worker."""
        print("[Orchestrator] Dispatching subtasks to workers...")
        context = {}   # accumulate results for downstream tasks

        for st in plan.subtasks:
            worker = WORKER_REGISTRY.get(st.worker_type)
            if not worker:
                st.result = f"No worker for type '{st.worker_type}'"
                st.done   = True
                continue

            # inject prior context into task description if available
            task_input = st.description
            if context:
                task_input += " | Context: " + " | ".join(context.values())

            st.result = worker.run(task_input)
            st.done   = True
            context[str(st.id)] = st.result[:100]
            print(f"  [Worker:{st.worker_type}] subtask {st.id} → done")

        return plan

    def synthesize(self, plan: OrchestratorPlan) -> str:
        """Combine all worker results into a final answer."""
        if self.llm:
            results_text = "\n".join(
                f"Subtask {st.id} ({st.worker_type}): {st.result}"
                for st in plan.subtasks
            )
            try:
                resp = self.llm.chat.completions.create(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Synthesize worker results into a clear final answer."},
                        {"role": "user",   "content": f"Goal: {plan.goal}\n\nWorker results:\n{results_text}\n\nFinal answer:"},
                    ],
                    temperature=0.3, max_tokens=300,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        # Mock synthesis
        lines = [f"Answer to: '{plan.goal}'"]
        for st in plan.subtasks:
            if st.result and "summary" not in st.worker_type:
                lines.append(f"  • {st.result}")
        return "\n".join(lines)

    def run(self, goal: str) -> str:
        plan = self.decompose(goal)
        print(f"\n{plan.summary()}")

        plan = self.dispatch(plan)
        print(f"\n{plan.summary()}")

        answer = self.synthesize(plan)
        return answer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Orchestrator/Worker Pattern")
    print("=" * 65)

    orchestrator = OrchestratorAgent()

    queries = [
        "What are TechCorp's pricing plans and what does the Pro plan cost annually?",
        "Is TechCorp secure and what SLA do they guarantee?",
        "How can I try TechCorp for free?",
    ]

    for query in queries:
        print(f"\n{'─'*65}")
        answer = orchestrator.run(query)
        print(f"\n[Final Answer]\n{answer}")

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Orchestrator: plans and routes — never executes directly")
    print("  • Workers: execute one task type — never plan")
    print("  • Context flows: worker N result feeds worker N+1")
    print("  • Swapping a worker doesn't require changing the orchestrator")
    print("  • LLM used for decomposition + synthesis; workers can be non-LLM")


if __name__ == "__main__":
    main()
