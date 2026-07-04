# Author: Mohan Raju Amuri
"""
02_react_pattern.py — ReAct: Reasoning + Acting, the standard agent framework

What to remember:
- ReAct = Reason + Act: the agent alternates "Thought" and "Action" steps
- Thought: LLM explains what it is going to do and why (chain-of-thought)
- Action: LLM calls a specific tool with specific parameters
- Observation: tool result is injected back into the prompt
- This loop repeats until the LLM outputs "Final Answer:"

Why ReAct works:
  - Thoughts force the LLM to plan before acting → fewer wrong tool calls
  - Observations ground subsequent thoughts in reality → less hallucination
  - The full trace is human-readable → easy to debug

What NOT to do:
- Don't skip the Thought step — agents without reasoning make random tool choices
- Don't let the model generate Action and Thought in the same line — parse them separately
- Don't truncate the observation — if tool output is long, summarize before feeding back

ReAct paper: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022)

Interview one-liner:
  "ReAct prompts the LLM to reason before acting — interleaving Thought, Action, Observation until done."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# Install: https://ollama.ai
# Pull:    ollama pull llama3.2:3b
# Serve:   ollama serve
#
# Falls back to a mock trace if Ollama is not running.
# ============================================================

import re
import json
from datetime import datetime
import math

# ── Tool registry ─────────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        # Safe eval: only math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(result, 6))
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Mock weather API."""
    weather_db = {
        "london": "Cloudy, 14°C, light rain expected",
        "new york": "Sunny, 22°C, clear skies",
        "tokyo": "Partly cloudy, 19°C, humidity 65%",
        "sydney": "Warm, 26°C, sunny",
    }
    return weather_db.get(city.lower(), f"Weather data unavailable for '{city}'")


def search_kb(query: str) -> str:
    """Mock knowledge base search."""
    kb = {
        "techcorp pricing": "Starter $99/mo, Professional $499/mo, Enterprise custom",
        "techcorp sla": "99.9% uptime for Pro/Enterprise, 99.5% for Starter",
        "techcorp free trial": "14-day free trial, full Professional features, no credit card",
        "python": "Python is a high-level, interpreted programming language",
        "react": "React is a JavaScript library for building user interfaces by Meta",
    }
    for key, val in kb.items():
        if any(word in query.lower() for word in key.split()):
            return val
    return "No relevant information found."


def get_current_time(_: str = "") -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "calculator":     {"fn": calculator,        "desc": "Evaluate math expressions. Input: expression string. Example: calculator('sqrt(144) + 5 * 3')"},
    "get_weather":    {"fn": get_weather,        "desc": "Get current weather for a city. Input: city name. Example: get_weather('London')"},
    "search_kb":      {"fn": search_kb,          "desc": "Search knowledge base. Input: search query. Example: search_kb('techcorp pricing')"},
    "get_current_time":{"fn": get_current_time, "desc": "Get current date and time. Input: empty string."},
}


# ── ReAct Prompt ──────────────────────────────────────────────────────────────
def build_react_prompt(task: str, history: list[dict]) -> str:
    tool_descriptions = "\n".join(
        f"- {name}: {info['desc']}" for name, info in TOOLS.items()
    )

    prompt = f"""You are an AI assistant that solves tasks step by step using tools.

Available tools:
{tool_descriptions}

To use a tool, output exactly:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool>

When you have the final answer, output:
Thought: I now have enough information to answer.
Final Answer: <your answer>

---
Task: {task}
"""
    # Append history (previous steps)
    for step in history:
        prompt += f"\nThought: {step['thought']}"
        if step.get("action"):
            prompt += f"\nAction: {step['action']}\nAction Input: {step['action_input']}\nObservation: {step['observation']}"
    prompt += "\nThought:"  # prime the LLM to continue
    return prompt


# ── Parse LLM output ──────────────────────────────────────────────────────────
def parse_react_output(text: str) -> dict:
    """Parse Thought / Action / Action Input / Final Answer from LLM output."""
    text = text.strip()

    if "Final Answer:" in text:
        thought = text.split("Final Answer:")[0].strip().lstrip("Thought:").strip()
        answer = text.split("Final Answer:")[-1].strip()
        return {"type": "final", "thought": thought, "answer": answer}

    thought_match = re.search(r"(?:Thought:)?\s*(.*?)(?=\nAction:|\Z)", text, re.DOTALL)
    action_match  = re.search(r"Action:\s*(\w+)", text)
    input_match   = re.search(r"Action Input:\s*(.*?)(?=\nObservation:|\Z)", text, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    action  = action_match.group(1).strip() if action_match else ""
    action_input = input_match.group(1).strip().strip("'\"") if input_match else ""

    return {"type": "action", "thought": thought, "action": action, "action_input": action_input}


# ── ReAct Agent ───────────────────────────────────────────────────────────────
class ReActAgent:
    def __init__(self, max_steps: int = 6, use_ollama: bool = True):
        self.max_steps = max_steps
        self.llm = self._init_llm() if use_ollama else None

    def _init_llm(self):
        try:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            client.models.list()
            return client
        except Exception:
            return None

    def _call_llm(self, prompt: str) -> str:
        if self.llm is None:
            return None  # fall back to mock
        try:
            resp = self.llm.chat.completions.create(
                model="llama3.2:3b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                stop=["Observation:"],  # stop before the model invents observations
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return None

    def run(self, task: str, mock_steps: list[dict] = None) -> str:
        print(f"\nTask: {task}")
        print("─" * 50)
        history = []

        for step_num in range(1, self.max_steps + 1):
            prompt = build_react_prompt(task, history)
            llm_output = self._call_llm(prompt)

            if llm_output is None:
                # Ollama not available — use mock steps if provided
                if mock_steps and step_num <= len(mock_steps):
                    parsed = mock_steps[step_num - 1]
                    parsed["_mocked"] = True
                else:
                    print("[Ollama not running — showing mock trace]")
                    break
            else:
                parsed = parse_react_output(llm_output)

            print(f"\nStep {step_num}:")
            print(f"  Thought: {parsed['thought']}")

            if parsed["type"] == "final":
                print(f"  Final Answer: {parsed['answer']}")
                return parsed["answer"]

            action = parsed.get("action", "")
            action_input = parsed.get("action_input", "")

            if action not in TOOLS:
                observation = f"Unknown tool '{action}'. Available: {list(TOOLS.keys())}"
            else:
                observation = TOOLS[action]["fn"](action_input)

            print(f"  Action: {action}('{action_input}')")
            print(f"  Observation: {observation}")

            history.append({
                "thought": parsed["thought"],
                "action": action,
                "action_input": action_input,
                "observation": observation,
            })

        return "Max steps reached without a final answer."


# ── Mock traces (used when Ollama is not running) ─────────────────────────────
mock_task1 = [
    {"type": "action", "thought": "I need to find the square root of 256. I'll use the calculator.",
     "action": "calculator", "action_input": "sqrt(256)"},
    {"type": "action", "thought": "Got 16. Now I need to multiply by 3.",
     "action": "calculator", "action_input": "16 * 3"},
    {"type": "final", "thought": "I have all the information.", "answer": "The square root of 256 is 16, and 16 × 3 = 48."},
]

mock_task2 = [
    {"type": "action", "thought": "I need weather for London and Paris. Starting with London.",
     "action": "get_weather", "action_input": "London"},
    {"type": "action", "thought": "Got London weather. Now let me check Paris — wait, not in my tools. Let me check New York instead.",
     "action": "get_weather", "action_input": "New York"},
    {"type": "final", "thought": "I have both cities' weather.", "answer": "London: Cloudy, 14°C, light rain. New York: Sunny, 22°C, clear."},
]

mock_task3 = [
    {"type": "action", "thought": "I need TechCorp pricing info. I'll search the knowledge base.",
     "action": "search_kb", "action_input": "techcorp pricing"},
    {"type": "action", "thought": "Found pricing. Now let me calculate the annual cost of the Professional plan.",
     "action": "calculator", "action_input": "499 * 12"},
    {"type": "final", "thought": "I have pricing and annual cost.", "answer": "Professional Plan is $499/month = $5,988/year. Starter is $99/month."},
]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — ReAct Pattern")
    print("=" * 60)

    agent = ReActAgent(max_steps=6)
    ollama_status = "connected" if agent.llm else "not running (using mock traces)"
    print(f"\nOllama: {ollama_status}")

    tasks = [
        ("Calculate sqrt(256) then multiply by 3", mock_task1),
        ("What is the weather in London and New York?", mock_task2),
        ("What is TechCorp's pricing and what does Professional cost annually?", mock_task3),
    ]

    for task, mock in tasks:
        print("\n" + "=" * 60)
        agent.run(task, mock_steps=mock)

    # Show the ReAct prompt structure
    print("\n" + "=" * 60)
    print("ReAct Prompt Structure")
    print("=" * 60)
    sample_prompt = build_react_prompt("What is 2 + 2?", [])
    print(sample_prompt[:500] + "...")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Thought before every Action forces explicit reasoning")
    print("  - stop=['Observation:'] prevents LLM from inventing tool outputs")
    print("  - parse_react_output extracts thought/action/answer from free text")
    print("  - Final Answer: signals the agent is done — check for it each step")
    print("  - ReAct is the foundation of LangChain agents, LlamaIndex agents, etc.")
    print("=" * 60)
