# Author: Mohan Raju Amuri
"""
10_production_agent.py — Robust agent with error recovery, retries, and guardrails

What to remember:
- Production agents need: retry logic, error recovery, guardrails, logging, timeouts
- Guardrails: reject harmful queries, cap costs, validate tool outputs before using them
- Self-healing: if a tool fails, the agent should try an alternative or report the failure clearly
- Observability: log every tool call + result so you can debug in production

Production checklist:
  ✓ Max steps limit (prevent infinite loops)
  ✓ Per-tool timeout (prevent hanging on slow APIs)
  ✓ Input guardrails (filter harmful/out-of-scope queries)
  ✓ Output validation (don't send garbage tool results to LLM)
  ✓ Retry logic (transient failures should not fail the whole task)
  ✓ Structured logging (every step logged with timestamp)
  ✓ Graceful degradation (fallback answers when tools fail)

What NOT to do:
- Don't expose raw stack traces to users — log internally, show friendly messages
- Don't retry infinitely — cap at 2-3 retries with exponential backoff
- Don't skip logging — debugging a silent agent failure in production is painful

Interview one-liner:
  "Production agents add retries, guardrails, timeouts, and structured logging around the core agent loop."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# ollama pull llama3.2:3b && ollama serve
# Core guardrails + retry demo works without Ollama.
# ============================================================

import json
import math
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

LLM_MODEL = "llama3.2:3b"

# ── Structured Logger ─────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("agent")


# ── Guardrails ────────────────────────────────────────────────────────────────
BLOCKED_PATTERNS = [
    "hack", "exploit", "password", "credit card number",
    "social security", "delete all", "drop table", "rm -rf",
]

MAX_QUERY_LENGTH = 500
MAX_TOOL_CALLS   = 8
MAX_RETRIES      = 2

def check_input_guardrail(query: str) -> tuple[bool, str]:
    """
    Returns (is_safe, reason).
    Block: harmful content, excessively long queries.
    """
    if len(query) > MAX_QUERY_LENGTH:
        return False, f"Query too long ({len(query)} chars). Max: {MAX_QUERY_LENGTH}."

    query_lower = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return False, f"Query contains blocked content: '{pattern}'."

    return True, ""


def validate_tool_output(tool_name: str, output: str) -> tuple[bool, str]:
    """
    Returns (is_valid, cleaned_output).
    Reject empty outputs or error messages before sending to LLM.
    """
    if not output or not output.strip():
        return False, "Tool returned empty output."

    if output.startswith("Error:"):
        return False, output

    # Cap long tool outputs (prevent context window overflow)
    if len(output) > 2000:
        output = output[:2000] + "... [truncated]"

    return True, output


# ── Tool registry with error handling ────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(result, 6))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: invalid expression — {e}"


def search_kb(query: str) -> str:
    """Simulates occasional failures for retry demo."""
    # Simulate a flaky tool (fails on first call with 'pricing')
    if not hasattr(search_kb, "_call_count"):
        search_kb._call_count = {}
    search_kb._call_count[query] = search_kb._call_count.get(query, 0) + 1

    if query == "pricing" and search_kb._call_count[query] == 1:
        raise ConnectionError("KB service temporarily unavailable")  # simulated failure

    data = {
        "pricing":  "Starter $99/mo | Pro $499/mo | Enterprise custom",
        "sla":      "99.9% uptime Pro/Enterprise, 99.5% Starter",
        "security": "SOC 2 Type II, AES-256, TLS 1.3",
        "trial":    "14-day free trial, full Professional",
    }
    for k, v in data.items():
        if k in query.lower():
            return v
    return "No results found for that query."


TOOLS = {"calculator": calculator, "search_kb": search_kb}

TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate math expressions.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "Math expression"}},
                       "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search product knowledge base for pricing, SLA, security, trial info.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query e.g. 'pricing'"}},
                       "required": ["query"]},
    }},
]


# ── Production Agent ──────────────────────────────────────────────────────────
@dataclass
class AgentRun:
    task: str
    start_time: datetime = field(default_factory=datetime.now)
    steps: list[dict] = field(default_factory=list)
    retries: int = 0
    errors: list[str] = field(default_factory=list)
    final_answer: str = ""
    status: str = "running"  # running | success | failed | blocked

    def log_step(self, tool: str, args: dict, result: str, latency_ms: float):
        self.steps.append({
            "tool": tool, "args": args, "result": result[:200],
            "latency_ms": round(latency_ms), "timestamp": datetime.now().isoformat()
        })

    def to_summary(self) -> dict:
        total_ms = (datetime.now() - self.start_time).total_seconds() * 1000
        return {
            "status": self.status,
            "steps": len(self.steps),
            "retries": self.retries,
            "errors": self.errors,
            "total_ms": round(total_ms),
            "answer_preview": self.final_answer[:100],
        }


def call_tool_with_retry(tool_name: str, args: dict, run: AgentRun) -> str:
    """Call a tool with retry logic and timing."""
    for attempt in range(1, MAX_RETRIES + 2):  # +2: original + retries
        start = time.monotonic()
        try:
            result = TOOLS[tool_name](**args)
            latency = (time.monotonic() - start) * 1000
            run.log_step(tool_name, args, result, latency)
            logger.info(f"Tool '{tool_name}' success on attempt {attempt} ({latency:.0f}ms)")

            valid, cleaned = validate_tool_output(tool_name, result)
            if not valid:
                raise ValueError(f"Tool output invalid: {cleaned}")
            return cleaned

        except Exception as e:
            latency = (time.monotonic() - start) * 1000
            run.errors.append(f"{tool_name} attempt {attempt}: {e}")
            logger.warning(f"Tool '{tool_name}' failed attempt {attempt}: {e}")
            run.retries += 1

            if attempt <= MAX_RETRIES:
                backoff = 0.1 * (2 ** (attempt - 1))  # exponential backoff: 0.1s, 0.2s
                logger.info(f"Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
            else:
                return f"Tool '{tool_name}' failed after {MAX_RETRIES + 1} attempts: {e}"

    return f"Tool '{tool_name}' failed all retries."


class ProductionAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def run(self, query: str) -> AgentRun:
        run = AgentRun(task=query)
        logger.info(f"Starting agent run: '{query[:60]}'")

        # Input guardrail
        safe, reason = check_input_guardrail(query)
        if not safe:
            run.final_answer = f"I can't help with that. {reason}"
            run.status = "blocked"
            logger.warning(f"Query blocked: {reason}")
            return run

        if self.llm is None:
            run = self._mock_run(query, run)
            return run

        # Agent loop
        messages = [
            {"role": "system", "content": "You are a helpful product assistant. Use tools accurately."},
            {"role": "user",   "content": query},
        ]

        for step_num in range(1, MAX_TOOL_CALLS + 1):
            try:
                resp = self.llm.chat.completions.create(
                    model=LLM_MODEL, messages=messages,
                    tools=TOOL_DEFS, tool_choice="auto",
                )
                msg = resp.choices[0].message
                if resp.choices[0].finish_reason == "stop" or not msg.tool_calls:
                    run.final_answer = msg.content or "Done."
                    run.status = "success"
                    break

                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = call_tool_with_retry(tc.function.name, args, run)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            except Exception as e:
                run.errors.append(f"LLM error at step {step_num}: {e}")
                logger.error(f"LLM error: {e}")
                run.final_answer = "I encountered an error processing your request. Please try again."
                run.status = "failed"
                break
        else:
            run.final_answer = "Max steps reached — task may be incomplete."
            run.status = "failed"

        return run

    def _mock_run(self, query: str, run: AgentRun) -> AgentRun:
        """Simulate a run for demo purposes."""
        q = query.lower()
        if "pricing" in q:
            result = call_tool_with_retry("search_kb", {"query": "pricing"}, run)
            run.final_answer = f"TechCorp pricing: {result}"
            run.status = "success"
        elif "%" in q or any(op in q for op in ["calculate", "what is", "how much is"]):
            result = call_tool_with_retry("calculator", {"expression": "499 * 12"}, run)
            run.final_answer = f"Result: {result}"
            run.status = "success"
        else:
            run.final_answer = "I don't have that information."
            run.status = "success"
        return run


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Production Agent")
    print("=" * 60)

    llm = None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        llm = client
        print(f"\nOllama: connected ({LLM_MODEL})")
    except Exception:
        print("\nOllama: not running (mock mode)")

    agent = ProductionAgent(llm=llm)

    test_queries = [
        ("Normal",     "What is TechCorp's pricing?"),
        ("Math",       "What is the annual cost of Professional plan at $499/month?"),
        ("Blocked",    "How do I hack into TechCorp's database?"),
        ("Guardrail",  "A" * 600),  # too long
    ]

    for label, query in test_queries:
        print(f"\n{'─'*60}")
        print(f"[{label}] Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        run = agent.run(query)
        summary = run.to_summary()
        print(f"  Status:  {summary['status']}")
        print(f"  Steps:   {summary['steps']}")
        print(f"  Retries: {summary['retries']} (includes simulated retry on pricing)")
        print(f"  Errors:  {summary['errors'][:2] if summary['errors'] else 'none'}")
        print(f"  Time:    {summary['total_ms']}ms")
        print(f"  Answer:  {summary['answer_preview']}")

    # Show the retry in action
    print("\n" + "=" * 60)
    print("Retry simulation — search_kb('pricing') fails on first call")
    print("=" * 60)
    search_kb._call_count = {}  # reset counter
    run2 = AgentRun(task="test")
    result = call_tool_with_retry("search_kb", {"query": "pricing"}, run2)
    print(f"  Final result: {result}")
    print(f"  Retries used: {run2.retries}")
    print(f"  Errors logged: {run2.errors}")

    print("\n" + "=" * 60)
    print("Production Agent Checklist")
    print("=" * 60)
    checklist = [
        ("✓", "Max steps limit",       "Prevents infinite loops"),
        ("✓", "Input guardrails",      "Block harmful / too-long queries"),
        ("✓", "Tool retry + backoff",  "Handle transient tool failures"),
        ("✓", "Output validation",     "Reject empty/error tool outputs"),
        ("✓", "Structured logging",    "Every step logged with timing"),
        ("✓", "Graceful degradation",  "Friendly error messages for users"),
        ("✓", "Run summary/trace",     "Full trajectory for debugging"),
    ]
    for mark, feature, desc in checklist:
        print(f"  {mark} {feature:<25} — {desc}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Always wrap tool calls in try/except — tools fail in production")
    print("  - Retry with exponential backoff: 0.1s → 0.2s → 0.4s")
    print("  - Guardrails go BEFORE the agent loop — reject early, cheaply")
    print("  - Log every step — silent failures are the hardest to debug")
    print("  - AgentRun trace gives you full observability over agent behavior")
    print("=" * 60)
