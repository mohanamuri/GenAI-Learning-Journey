# Author: Mohan Raju Amuri
"""
01_what_is_an_agent.py — The agent loop: observe → think → act → repeat

What to remember:
- An AI Agent = LLM + tools + loop: it takes actions, observes results, and keeps going
- Unlike a single LLM call, an agent runs multiple steps to complete a task
- The agent decides WHEN to stop — it checks if the goal is achieved after each action
- Three parts: Perception (input), Reasoning (LLM), Action (tool calls)

What NOT to do:
- Don't confuse "agent" with a simple LLM call — agents are multi-step and stateful
- Don't let agents run without a max_steps limit — infinite loops are real
- Don't skip the "thought" step — agents need to reason before acting

Agent vs Chain vs LLM call:
  LLM call:  single input → LLM → single output (no state, no loop)
  Chain:     fixed sequence of LLM calls (predefined steps)
  Agent:     LLM decides which tool to call next (dynamic, self-directed)

Interview one-liner:
  "An agent is an LLM in a loop: it observes, reasons, acts, observes again — until the task is done."
"""

import json
from datetime import datetime

# ── What makes something an Agent? ───────────────────────────────────────────
# The core loop: think → act → observe → think → act → ...
# Each iteration is called a "step" or "turn"

class AgentStep:
    """A single step in the agent's reasoning loop."""
    def __init__(self, thought: str, action: str, action_input: dict, observation: str):
        self.thought = thought          # what the agent is thinking
        self.action = action            # which tool to call
        self.action_input = action_input # what to pass to the tool
        self.observation = observation  # what the tool returned
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        return (
            f"\nThought:     {self.thought}\n"
            f"Action:      {self.action}({json.dumps(self.action_input)})\n"
            f"Observation: {self.observation}"
        )


# ── Minimal toy tools ─────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # restricted eval
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"


def get_current_date(_: str = "") -> str:
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")


def word_count(text: str) -> str:
    """Count words in text."""
    return f"{len(text.split())} words"


TOOLS = {
    "calculator":      {"fn": calculator,       "desc": "Evaluate math: calculator('2 + 2 * 3')"},
    "get_current_date":{"fn": get_current_date,  "desc": "Get today's date: get_current_date()"},
    "word_count":      {"fn": word_count,        "desc": "Count words: word_count('some text here')"},
}


# ── Simulated Agent Trace ─────────────────────────────────────────────────────
# This shows what a real agent's reasoning trace looks like.
# In production, the LLM generates the thought and action; here it's hard-coded
# so you can see the structure without needing Ollama running.

def simulate_agent_trace(task: str) -> list[AgentStep]:
    """
    Simulate how an agent would reason through a multi-step task.
    Replace hard-coded thoughts with an LLM call in production.
    """
    steps = []

    if "date" in task.lower() and ("day" in task.lower() or "today" in task.lower()):
        # Step 1: get the date
        obs = TOOLS["get_current_date"]["fn"]("")
        steps.append(AgentStep(
            thought="I need to find today's date. I'll use the get_current_date tool.",
            action="get_current_date",
            action_input={},
            observation=obs
        ))
        # Step 2: calculate days (simplified)
        obs2 = TOOLS["calculator"]["fn"]("365 - 184")
        steps.append(AgentStep(
            thought=f"Today is {obs}. Now I need to calculate how many days are left in the year.",
            action="calculator",
            action_input={"expression": "365 - 184"},
            observation=obs2
        ))

    elif "calculate" in task.lower() or any(op in task for op in ["+", "-", "*", "/"]):
        expr = task.split("calculate")[-1].strip().strip("?").strip()
        obs = TOOLS["calculator"]["fn"](expr or "15 * 7 + 23")
        steps.append(AgentStep(
            thought=f"The user wants me to calculate something. I'll use the calculator tool.",
            action="calculator",
            action_input={"expression": expr or "15 * 7 + 23"},
            observation=obs
        ))

    else:
        obs = TOOLS["word_count"]["fn"](task)
        steps.append(AgentStep(
            thought="I'm not sure what specific action is needed. Let me analyze the input.",
            action="word_count",
            action_input={"text": task},
            observation=obs
        ))

    return steps


# ── The Agent Loop ─────────────────────────────────────────────────────────────
class SimpleAgent:
    """
    Demonstrates the fundamental agent loop structure.
    The `think` method would be replaced by an LLM call in production.
    """
    def __init__(self, tools: dict, max_steps: int = 5):
        self.tools = tools
        self.max_steps = max_steps   # always set a limit — prevents infinite loops
        self.history: list[AgentStep] = []

    def run(self, task: str) -> str:
        print(f"\n[Agent] Task: {task}")
        print(f"[Agent] Available tools: {list(self.tools.keys())}")
        print(f"[Agent] Max steps: {self.max_steps}\n")

        # Simulate steps (in production: LLM decides each step)
        steps = simulate_agent_trace(task)

        for i, step in enumerate(steps[:self.max_steps]):
            print(f"--- Step {i+1} ---{step}")
            self.history.append(step)

        # Final answer (in production: LLM synthesizes from history)
        final = self._synthesize(task, self.history)
        print(f"\n[Agent] Final Answer: {final}")
        return final

    def _synthesize(self, task: str, history: list[AgentStep]) -> str:
        """Summarize observations into a final answer."""
        if not history:
            return "I could not complete the task."
        last_obs = history[-1].observation
        return f"Based on the tools I used: {last_obs}"


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — What is an Agent?")
    print("=" * 60)

    agent = SimpleAgent(tools=TOOLS, max_steps=5)

    # Task 1: single tool
    print("\n" + "─" * 60)
    print("Task 1: Single tool use")
    agent.run("calculate 15 * 7 + 23")

    # Task 2: multi-step
    print("\n" + "─" * 60)
    print("Task 2: Multi-step reasoning")
    agent = SimpleAgent(tools=TOOLS, max_steps=5)
    agent.run("What is today's date and how many days are left in the year?")

    # Show the anatomy of an agent
    print("\n" + "=" * 60)
    print("Anatomy of an AI Agent")
    print("=" * 60)
    components = [
        ("Perception",  "Receives the task/goal from the user"),
        ("Memory",      "Stores conversation history and past observations"),
        ("Reasoning",   "LLM decides which tool to call and with what input"),
        ("Action",      "Calls a tool (function, API, web search, code exec)"),
        ("Observation", "Reads the tool's output and feeds it back to the LLM"),
        ("Loop",        "Repeats until task is done or max_steps reached"),
    ]
    for name, desc in components:
        print(f"  {name:<14} — {desc}")

    print("\nAgent loop in pseudocode:")
    loop = """
  task = "user's goal"
  history = []
  for step in range(max_steps):
      thought, action, action_input = llm.think(task, history)
      if action == "FINISH":
          return llm.synthesize(history)
      observation = tools[action](action_input)
      history.append((thought, action, observation))
    """
    print(loop)

    print("=" * 60)
    print("Key Takeaways:")
    print("  - Agent = LLM + loop + tools (not just a single LLM call)")
    print("  - The LLM decides what to do next at every step")
    print("  - Always set max_steps — agents can loop indefinitely")
    print("  - History carries the context between steps")
    print("  - Next: the ReAct pattern formalizes thought → action → observation")
    print("=" * 60)
