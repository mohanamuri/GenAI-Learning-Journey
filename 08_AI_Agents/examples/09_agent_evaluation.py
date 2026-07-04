# Author: Mohan Raju Amuri
"""
09_agent_evaluation.py — Measure agent performance: task completion, tool accuracy, efficiency

What to remember:
- Evaluate agents on: task success, tool selection accuracy, step efficiency, final answer quality
- Build a test suite of tasks with expected tool sequences and expected answers
- Three failure modes: wrong tool selected, correct tool + wrong input, correct tools + hallucinated synthesis
- Trajectory evaluation: did the agent take the right path, not just get the right answer?

Metrics:
  Task Success Rate   — did the agent complete the task (final answer reached)?
  Tool Accuracy       — did the agent select the correct tools in the right order?
  Step Efficiency     — did the agent solve in minimal steps (no unnecessary calls)?
  Answer Quality      — is the final answer correct/faithful to tool outputs?

What NOT to do:
- Don't evaluate only the final answer — an agent can get the right answer via wrong reasoning
- Don't skip negative tests — agents must handle out-of-scope queries gracefully
- Don't evaluate on training-like data — use held-out test cases

Interview one-liner:
  "Agent evaluation checks tool selection, step count, and answer correctness — not just output quality."
"""

import json
from dataclasses import dataclass, field
from typing import Optional
import math
import re

# ── Tool implementations (simplified) ────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(round(eval(expression, {"__builtins__": {}}, allowed), 4))
    except Exception as e:
        return f"Error: {e}"

def search_kb(query: str) -> str:
    data = {
        "pricing":   "Starter $99/mo, Pro $499/mo, Enterprise custom",
        "sla":       "99.9% uptime Pro/Enterprise, 99.5% Starter",
        "security":  "SOC 2 Type II, AES-256, TLS 1.3",
        "trial":     "14-day free trial, full Professional, no credit card",
        "support":   "Enterprise: 24/7; Pro: 8h Slack; Starter: 48h email",
    }
    for k, v in data.items():
        if k in query.lower():
            return v
    return "No results"

def get_weather(city: str) -> str:
    return {"london": "14°C cloudy", "tokyo": "21°C sunny"}.get(city.lower(), "Unknown")

TOOLS = {
    "calculator": calculator,
    "search_kb":  search_kb,
    "get_weather": get_weather,
}


# ── Agent Trajectory (what the agent actually did) ────────────────────────────
@dataclass
class ToolCall:
    tool: str
    args: dict
    result: str

@dataclass
class AgentTrajectory:
    task: str
    steps: list[ToolCall] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False

    def add_step(self, tool: str, args: dict, result: str):
        self.steps.append(ToolCall(tool=tool, args=args, result=result))

    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


# ── Test Cases ────────────────────────────────────────────────────────────────
@dataclass
class AgentTestCase:
    name: str
    task: str
    expected_tools: list[str]          # tools that MUST be called (in any order)
    expected_answer_contains: list[str] # strings that must appear in the answer
    max_steps: int = 4
    should_succeed: bool = True         # False for out-of-scope queries


TEST_SUITE = [
    AgentTestCase(
        name="Pricing query",
        task="How much does the Professional plan cost per month?",
        expected_tools=["search_kb"],
        expected_answer_contains=["499", "professional"],
    ),
    AgentTestCase(
        name="Pricing + calculation",
        task="What is TechCorp Professional plan pricing and what is the annual cost?",
        expected_tools=["search_kb", "calculator"],
        expected_answer_contains=["499", "5988"],
    ),
    AgentTestCase(
        name="Security query",
        task="Is TechCorp SOC 2 certified?",
        expected_tools=["search_kb"],
        expected_answer_contains=["soc 2", "aes"],
    ),
    AgentTestCase(
        name="Math only",
        task="Calculate 15% of 2400",
        expected_tools=["calculator"],
        expected_answer_contains=["360"],
    ),
    AgentTestCase(
        name="Weather query",
        task="What is the weather in Tokyo?",
        expected_tools=["get_weather"],
        expected_answer_contains=["tokyo", "°c"],
    ),
    AgentTestCase(
        name="Out-of-scope — should not hallucinate",
        task="What is the stock price of TechCorp today?",
        expected_tools=[],  # no relevant tool should be called
        expected_answer_contains=["don't", "not", "unable", "no information"],
        should_succeed=True,  # succeeds by gracefully declining
    ),
]


# ── Evaluation Metrics ────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    test_name: str
    task_success: bool
    tool_precision: float   # correct tools / tools called
    tool_recall: float      # expected tools found / expected tools
    step_efficiency: float  # expected_steps / actual_steps (capped at 1.0)
    answer_quality: float   # fraction of expected strings found in answer
    notes: str = ""


def evaluate(test: AgentTestCase, trajectory: AgentTrajectory) -> EvalResult:
    """Score a single agent run against its test case."""
    actual_tools = trajectory.tools_used()
    expected = set(t.lower() for t in test.expected_tools)
    actual   = set(t.lower() for t in actual_tools)

    # Tool precision: of tools called, how many were correct
    if actual:
        tool_precision = len(expected & actual) / len(actual)
    else:
        tool_precision = 1.0 if not expected else 0.0

    # Tool recall: of expected tools, how many were called
    tool_recall = len(expected & actual) / len(expected) if expected else 1.0

    # Step efficiency: penalize extra steps
    expected_steps = max(len(test.expected_tools), 1)
    actual_steps   = max(len(trajectory.steps), 1)
    step_efficiency = min(1.0, expected_steps / actual_steps)

    # Answer quality: check for expected strings in final answer
    answer_lower = trajectory.final_answer.lower()
    found = sum(1 for s in test.expected_answer_contains if s.lower() in answer_lower)
    answer_quality = found / len(test.expected_answer_contains) if test.expected_answer_contains else 1.0

    task_success = trajectory.success and answer_quality >= 0.5

    notes = ""
    if tool_recall < 1.0:
        missing = expected - actual
        notes += f"Missing tools: {missing}. "
    if step_efficiency < 0.8:
        notes += f"Used {actual_steps} steps, expected ~{expected_steps}. "
    if answer_quality < 1.0:
        missing_ans = [s for s in test.expected_answer_contains if s.lower() not in answer_lower]
        notes += f"Answer missing: {missing_ans}."

    return EvalResult(
        test_name=test.name,
        task_success=task_success,
        tool_precision=round(tool_precision, 2),
        tool_recall=round(tool_recall, 2),
        step_efficiency=round(step_efficiency, 2),
        answer_quality=round(answer_quality, 2),
        notes=notes.strip(),
    )


# ── Mock Agent (simulates agent runs without Ollama) ─────────────────────────
def mock_agent_run(task: str) -> AgentTrajectory:
    """Rule-based agent for evaluation demo — mimics what a real agent would do."""
    t = task.lower()
    traj = AgentTrajectory(task=task)

    if "professional" in t and ("annual" in t or "year" in t):
        result1 = search_kb("pricing")
        traj.add_step("search_kb", {"query": "pricing"}, result1)
        result2 = calculator("499 * 12")
        traj.add_step("calculator", {"expression": "499 * 12"}, result2)
        traj.final_answer = f"Professional plan is $499/month. Annual cost: ${result2}. ({result1})"
        traj.success = True

    elif "professional" in t and ("cost" in t or "price" in t or "how much" in t):
        result = search_kb("pricing")
        traj.add_step("search_kb", {"query": "pricing"}, result)
        traj.final_answer = f"The Professional plan costs $499/month. ({result})"
        traj.success = True

    elif any(w in t for w in ["soc 2", "security", "certified", "encrypt"]):
        result = search_kb("security")
        traj.add_step("search_kb", {"query": "security"}, result)
        traj.final_answer = f"Yes, TechCorp is SOC 2 Type II certified. {result}"
        traj.success = True

    elif "15%" in t or "calculate" in t or re.search(r"\d+\s*[%\+\-\*\/]", t):
        result = calculator("0.15 * 2400")
        traj.add_step("calculator", {"expression": "0.15 * 2400"}, result)
        traj.final_answer = f"15% of 2400 = {result}"
        traj.success = True

    elif "weather" in t:
        city = re.search(r"in (\w+)", t)
        city = city.group(1) if city else "london"
        result = get_weather(city)
        traj.add_step("get_weather", {"city": city}, result)
        traj.final_answer = f"Weather in {city}: {result}"
        traj.success = True

    elif "stock price" in t:
        # Out-of-scope: agent should not hallucinate
        traj.final_answer = "I'm sorry, I don't have access to real-time stock prices. I'm not able to provide that information."
        traj.success = True  # succeeds by gracefully declining

    else:
        traj.final_answer = "I don't have enough information to answer this."
        traj.success = False

    return traj


# ── Evaluation Runner ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Evaluation")
    print("=" * 60)

    results = []
    trajectories = []

    print(f"\nRunning {len(TEST_SUITE)} test cases...\n")

    for test in TEST_SUITE:
        traj = mock_agent_run(test.task)
        result = evaluate(test, traj)
        results.append(result)
        trajectories.append((test, traj))
        status = "✓" if result.task_success else "✗"
        print(f"  {status} {test.name:<35} P={result.tool_precision:.2f} R={result.tool_recall:.2f} "
              f"Eff={result.step_efficiency:.2f} Ans={result.answer_quality:.2f}")
        if result.notes:
            print(f"      Note: {result.notes}")

    # Aggregate
    n = len(results)
    print(f"\n{'─'*60}")
    print(f"  Task success:       {sum(r.task_success for r in results)}/{n} = "
          f"{sum(r.task_success for r in results)/n:.0%}")
    print(f"  Avg tool precision: {sum(r.tool_precision for r in results)/n:.2f}")
    print(f"  Avg tool recall:    {sum(r.tool_recall for r in results)/n:.2f}")
    print(f"  Avg step efficiency:{sum(r.step_efficiency for r in results)/n:.2f}")
    print(f"  Avg answer quality: {sum(r.answer_quality for r in results)/n:.2f}")

    # Detailed view of one run
    print("\n" + "=" * 60)
    print("Detailed trajectory — 'Pricing + calculation' test")
    print("=" * 60)
    test, traj = trajectories[1]
    print(f"\nTask: {traj.task}")
    for i, step in enumerate(traj.steps, 1):
        print(f"  Step {i}: {step.tool}({step.args}) → {step.result}")
    print(f"  Final answer: {traj.final_answer}")
    r = results[1]
    print(f"\n  Scores: precision={r.tool_precision} recall={r.tool_recall} "
          f"efficiency={r.step_efficiency} answer={r.answer_quality}")

    # Evaluation insights
    print("\n" + "=" * 60)
    print("What to fix based on evaluation results")
    print("=" * 60)
    print("""
  Low tool recall (missing tools):
    → Add more specific tool descriptions
    → Add examples in tool description: "e.g., search_kb('pricing')"

  Low tool precision (calling wrong tools):
    → Make tool descriptions non-overlapping
    → Reduce number of tools (fewer = clearer choice)

  Low step efficiency (too many steps):
    → Improve system prompt to discourage redundant tool calls
    → Add "you should only call each tool once" to system prompt

  Low answer quality (answer misses key facts):
    → Increase context window / send more tool results to LLM
    → Add "always include numbers in your answer" to system prompt
    """)

    print("=" * 60)
    print("Key Takeaways:")
    print("  - Test with expected tool sequences, not just final answers")
    print("  - 4 metrics: success rate, tool precision, recall, step efficiency")
    print("  - Out-of-scope tests are critical — check for graceful failure")
    print("  - Build eval set before tuning any agent parameter")
    print("=" * 60)
