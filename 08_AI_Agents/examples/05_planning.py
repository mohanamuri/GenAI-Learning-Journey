# Author: Mohan Raju Amuri
"""
05_planning.py — Task decomposition and plan-and-execute agents

What to remember:
- Planning agents separate "what to do" from "how to do it"
- Plan-and-Execute: first generate the full plan (list of steps), then execute each step
- Better than pure ReAct for complex tasks — reduces mid-task confusion
- Two LLM calls: one to plan (high-level), one per step (execution)

Planning patterns:
  Sequential plan    — steps run one after another (most common)
  DAG / parallel     — independent steps run at the same time
  Hierarchical plan  — planner creates sub-planners for complex subtasks
  Tree of Thoughts   — explore multiple plan branches, pick the best

What NOT to do:
- Don't mix planning and execution in the same prompt — planning quality degrades
- Don't let plans have more than 5-7 steps — agents lose track beyond that
- Don't hardcode the plan — always let the LLM generate it from the task

Plan-and-Execute vs ReAct:
  ReAct:            reactive, one tool at a time, replans on the fly
  Plan-and-Execute: proactive, creates full roadmap first, then executes
  Use Plan-and-Execute for: long tasks, research tasks, report generation

Interview one-liner:
  "Plan-and-Execute splits thinking from doing — the planner creates steps, executor runs each one."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# ollama pull llama3.2:3b && ollama serve
# Falls back to mock plans if Ollama is not running.
# ============================================================

import json
import math
import re
from datetime import datetime

LLM_MODEL = "llama3.2:3b"

# ── Tool registry ─────────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(round(eval(expression, {"__builtins__": {}}, allowed), 4))
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    """Mock web search."""
    results = {
        "python popularity":  "Python ranked #1 in TIOBE index 2024. 30% of developers use it.",
        "llm market size":    "LLM market expected to reach $259B by 2030, CAGR 79%.",
        "pytorch vs tensorflow": "PyTorch leads in research (85% of papers); TensorFlow leads in enterprise.",
        "rag definition":     "RAG combines retrieval with generation to ground LLM answers in documents.",
        "agent frameworks":   "LangChain, LlamaIndex, AutoGen, CrewAI are popular agent frameworks.",
    }
    for key, val in results.items():
        if any(word in query.lower() for word in key.split()):
            return val
    return f"Search results for '{query}': found 3 relevant articles about AI and technology."

def write_report_section(title: str, content: str) -> str:
    """Mock report writing."""
    return f"## {title}\n\n{content}\n\n[Section written: {len(content)} chars]"

def summarize(text: str) -> str:
    """Mock summarizer."""
    sentences = text.split(". ")
    return ". ".join(sentences[:2]) + "." if sentences else text

TOOLS = {
    "calculator":     {"fn": calculator,          "desc": "Evaluate math expressions"},
    "search_web":     {"fn": search_web,           "desc": "Search the web for information"},
    "write_section":  {"fn": write_report_section, "desc": "Write a section of a report"},
    "summarize":      {"fn": summarize,            "desc": "Summarize a piece of text"},
}


# ── LLM Setup ─────────────────────────────────────────────────────────────────
def init_llm():
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        return client
    except Exception:
        return None


def call_llm(llm, prompt: str, max_tokens: int = 400) -> str | None:
    if llm is None:
        return None
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ── Planner ───────────────────────────────────────────────────────────────────
PLANNER_PROMPT = """You are a planning assistant. Break the task into clear numbered steps.
Each step should be one concrete action using one of these tools: {tools}

Format each step as:
1. [tool_name]: what to do

Task: {task}

Steps (maximum 5):"""

def generate_plan(task: str, llm) -> list[dict]:
    """Generate a plan as a list of steps."""
    tool_list = ", ".join(TOOLS.keys())
    prompt = PLANNER_PROMPT.format(tools=tool_list, task=task)
    response = call_llm(llm, prompt, max_tokens=200)

    if response is None:
        return None  # fall back to mock

    steps = []
    for line in response.split("\n"):
        # Parse "1. [tool_name]: description" or "1. tool_name: description"
        match = re.match(r"\d+\.\s*\[?(\w+)\]?:\s*(.*)", line.strip())
        if match:
            tool, desc = match.group(1), match.group(2).strip()
            steps.append({"tool": tool if tool in TOOLS else "search_web", "description": desc})
    return steps if steps else None


# ── Executor ──────────────────────────────────────────────────────────────────
EXECUTOR_PROMPT = """You are executing step {step_num} of a plan.

Original task: {task}
Current step: {step_description}
Available tool: {tool_name} — {tool_desc}
Previous results: {previous_results}

Generate the exact input to pass to the tool '{tool_name}'.
Output ONLY the tool input string, nothing else."""

def execute_step(task: str, step: dict, step_num: int, previous: list, llm) -> str:
    """Execute a single plan step."""
    tool_name = step["tool"]
    if tool_name not in TOOLS:
        tool_name = "search_web"

    tool_fn = TOOLS[tool_name]["fn"]
    tool_desc = TOOLS[tool_name]["desc"]
    prev_str = " | ".join(previous[-3:]) if previous else "none"

    # Ask LLM to generate the tool input
    prompt = EXECUTOR_PROMPT.format(
        step_num=step_num,
        task=task,
        step_description=step["description"],
        tool_name=tool_name,
        tool_desc=tool_desc,
        previous_results=prev_str,
    )
    tool_input = call_llm(llm, prompt, max_tokens=100)
    if tool_input is None:
        tool_input = step["description"]  # fall back to using description directly

    # Execute
    result = tool_fn(tool_input.strip().strip("'\""))
    return result


# ── Plan-and-Execute Agent ────────────────────────────────────────────────────
class PlanAndExecuteAgent:
    def __init__(self, llm=None, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max_steps

    def run(self, task: str, mock_plan: list = None) -> dict:
        print(f"\nTask: {task}")
        print("─" * 50)

        # Phase 1: Plan
        print("\n[Phase 1: Planning]")
        plan = generate_plan(task, self.llm)
        if plan is None:
            plan = mock_plan
            print("  (using mock plan)")
        if not plan:
            return {"task": task, "plan": [], "results": [], "answer": "Could not generate a plan."}

        print(f"  Generated {len(plan)} steps:")
        for i, step in enumerate(plan[:self.max_steps], 1):
            print(f"    {i}. [{step['tool']}] {step['description']}")

        # Phase 2: Execute
        print("\n[Phase 2: Execution]")
        results = []
        for i, step in enumerate(plan[:self.max_steps], 1):
            result = execute_step(task, step, i, results, self.llm)
            results.append(result)
            print(f"  Step {i} [{step['tool']}]: {result[:80]}")

        # Phase 3: Synthesize
        print("\n[Phase 3: Synthesis]")
        synthesis = f"Completed {len(results)} steps. Key findings: " + " | ".join(results[:3])
        print(f"  {synthesis}")

        return {"task": task, "plan": plan, "results": results, "answer": synthesis}


# ── Mock plans (shown when Ollama is not running) ─────────────────────────────
mock_research_plan = [
    {"tool": "search_web",    "description": "research LLM market size and growth"},
    {"tool": "search_web",    "description": "research popular agent frameworks"},
    {"tool": "calculator",    "description": "calculate percentage growth rate"},
    {"tool": "write_section", "description": "write market overview section"},
    {"tool": "summarize",     "description": "create executive summary"},
]

mock_analysis_plan = [
    {"tool": "search_web",    "description": "search pytorch vs tensorflow comparison"},
    {"tool": "search_web",    "description": "search python popularity statistics"},
    {"tool": "calculator",    "description": "85 / 100 * 259 — estimate PyTorch share of LLM market"},
    {"tool": "write_section", "description": "write framework comparison analysis"},
]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Planning")
    print("=" * 60)

    llm = init_llm()
    print(f"\nOllama: {'connected' if llm else 'not running (using mock plans)'}")

    agent = PlanAndExecuteAgent(llm=llm, max_steps=5)

    print("\n" + "=" * 60)
    print("Task 1: Research Report")
    print("=" * 60)
    agent.run(
        "Write a short research report on the LLM and AI agent market",
        mock_plan=mock_research_plan
    )

    print("\n" + "=" * 60)
    print("Task 2: Technology Analysis")
    print("=" * 60)
    agent.run(
        "Compare PyTorch vs TensorFlow and Python's market position in AI",
        mock_plan=mock_analysis_plan
    )

    # Show the conceptual difference between ReAct and Plan-and-Execute
    print("\n" + "=" * 60)
    print("ReAct vs Plan-and-Execute")
    print("=" * 60)
    print("""
  ReAct (reactive):
    Task → Think → Act → Observe → Think → Act → ...
    Replans after every observation. Good for dynamic tasks.
    Risk: gets stuck in loops, may take wrong detours.

  Plan-and-Execute (proactive):
    Task → [Planner] → Full plan → [Executor] step 1 → step 2 → ...
    Creates roadmap upfront. Good for structured, predictable tasks.
    Risk: plan may be wrong if task changes mid-way.

  When to use:
    ReAct            → open-ended exploration, web research, debugging
    Plan-and-Execute → report generation, data pipelines, multi-step analysis
    """)

    print("=" * 60)
    print("Key Takeaways:")
    print("  - Planning = generate all steps first, then execute in order")
    print("  - Two LLM calls: planner (high-level) + executor (per step)")
    print("  - Separate planner and executor prompts → better output per role")
    print("  - Max 5-7 steps per plan — agents drift on longer plans")
    print("  - Add a re-planner if a step fails — let the agent self-correct")
    print("=" * 60)
