# Author: Mohan Raju Amuri
"""
09_agent_state_machine.py — Agent lifecycle with explicit states

Why state machines for agents?
  - Prevents agents from doing work when in the wrong state
  - Makes lifecycle auditable (log every state transition)
  - Enables pause / resume for long-running agents
  - Simplifies error handling (ERROR state catches all bad transitions)

States:
  IDLE → RUNNING → WAITING_FOR_TOOL → DONE
                ↘                  ↗
                 ──── ERROR ────

Transitions:
  IDLE       → RUNNING            (task received)
  RUNNING    → WAITING_FOR_TOOL   (tool call issued)
  RUNNING    → DONE               (answer generated without tools)
  RUNNING    → ERROR              (LLM or logic failure)
  WAITING... → RUNNING            (tool result received)
  WAITING... → ERROR              (tool call failed)
  DONE / ERROR → IDLE             (reset for next task)

No Ollama required — demonstrates state mechanics with mock agent.
"""

import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable


# ── State definitions ─────────────────────────────────────────────────────────

class AgentState(Enum):
    IDLE              = auto()
    RUNNING           = auto()
    WAITING_FOR_TOOL  = auto()
    DONE              = auto()
    ERROR             = auto()


# Valid transitions: current_state → set of allowed next states
VALID_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE:             {AgentState.RUNNING},
    AgentState.RUNNING:          {AgentState.WAITING_FOR_TOOL, AgentState.DONE, AgentState.ERROR},
    AgentState.WAITING_FOR_TOOL: {AgentState.RUNNING, AgentState.ERROR},
    AgentState.DONE:             {AgentState.IDLE},
    AgentState.ERROR:            {AgentState.IDLE},
}


# ── State machine ─────────────────────────────────────────────────────────────

@dataclass
class StateTransitionEvent:
    from_state: AgentState
    to_state:   AgentState
    reason:     str
    timestamp:  float = field(default_factory=time.time)


class AgentStateMachine:
    """
    Thread-safe state machine with transition validation and event log.
    """

    def __init__(self, name: str):
        self.name        = name
        self.state       = AgentState.IDLE
        self.history: list[StateTransitionEvent] = []
        self._lock       = threading.Lock()
        self._on_enter: dict[AgentState, list[Callable]] = {}

    def on_enter(self, state: AgentState, fn: Callable) -> None:
        """Register a callback that fires when entering a state."""
        self._on_enter.setdefault(state, []).append(fn)

    def transition(self, new_state: AgentState, reason: str = "") -> None:
        with self._lock:
            allowed = VALID_TRANSITIONS.get(self.state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"[{self.name}] invalid transition: {self.state.name} → {new_state.name} "
                    f"(allowed: {[s.name for s in allowed]})"
                )
            event = StateTransitionEvent(self.state, new_state, reason)
            self.history.append(event)
            old = self.state
            self.state = new_state
            print(f"  [{self.name}] {old.name} → {new_state.name}" +
                  (f"  ({reason})" if reason else ""))

        # Fire callbacks (outside lock to avoid deadlock)
        for cb in self._on_enter.get(new_state, []):
            cb()

    def reset(self) -> None:
        final = self.state
        if final not in (AgentState.DONE, AgentState.ERROR):
            raise ValueError(f"[{self.name}] can only reset from DONE or ERROR, currently {self.state.name}")
        self.transition(AgentState.IDLE, "reset for next task")

    def print_history(self) -> None:
        print(f"\n  [{self.name}] state history ({len(self.history)} transitions):")
        for ev in self.history:
            print(f"    {ev.from_state.name:20s} → {ev.to_state.name:20s}  {ev.reason}")


# ── Stateful agent ────────────────────────────────────────────────────────────

class StatefulAgent:
    """
    Agent that uses AgentStateMachine to manage its lifecycle.
    Simulates: receive task → maybe call tool → generate answer.
    """

    TOOLS = {
        "search_kb":   lambda q: f"KB result for '{q}': Pro=$499/mo",
        "calculator":  lambda expr: f"calc({expr})=5988",
        "get_weather": lambda city: f"Weather in {city}: 22°C, sunny",
    }

    def __init__(self, name: str):
        self.name = name
        self.sm   = AgentStateMachine(name)
        self._result: Optional[str] = None

        # Register callbacks
        self.sm.on_enter(AgentState.DONE,  lambda: print(f"  [{name}] callback: task complete"))
        self.sm.on_enter(AgentState.ERROR, lambda: print(f"  [{name}] callback: entering error state"))

    def _pick_tool(self, task: str) -> Optional[tuple[str, str]]:
        """Simple intent-based tool selection."""
        t = task.lower()
        if any(w in t for w in ["price", "plan", "cost"]): return ("search_kb",   task)
        if any(w in t for w in ["calcul", "annual", "*"]): return ("calculator",  "499*12")
        if any(w in t for w in ["weather", "temperature"]): return ("get_weather", "London")
        return None

    def run(self, task: str) -> str:
        # IDLE → RUNNING
        self.sm.transition(AgentState.RUNNING, f"task received: '{task[:40]}'")

        try:
            tool_pick = self._pick_tool(task)

            if tool_pick:
                tool_name, tool_arg = tool_pick
                # RUNNING → WAITING_FOR_TOOL
                self.sm.transition(AgentState.WAITING_FOR_TOOL, f"calling {tool_name}()")
                time.sleep(0.1)   # simulated tool latency

                tool_fn = self.TOOLS.get(tool_name)
                if not tool_fn:
                    self.sm.transition(AgentState.ERROR, f"tool '{tool_name}' not found")
                    self.sm.reset()
                    return f"Error: unknown tool '{tool_name}'"

                tool_result = tool_fn(tool_arg)
                # WAITING_FOR_TOOL → RUNNING
                self.sm.transition(AgentState.RUNNING, f"tool result received: {tool_result[:30]}")
                answer = f"Answer based on {tool_name}: {tool_result}"
            else:
                answer = f"Direct answer: {task[:60]}"

            # RUNNING → DONE
            self.sm.transition(AgentState.DONE, "answer generated")
            self._result = answer
            self.sm.reset()
            return answer

        except Exception as e:
            if self.sm.state not in (AgentState.DONE, AgentState.ERROR):
                self.sm.transition(AgentState.ERROR, str(e))
            self.sm.reset()
            return f"Error: {e}"


# ── Invalid transition demo ───────────────────────────────────────────────────

def demo_invalid_transition():
    print("\n── Invalid transition guard ──")
    sm = AgentStateMachine("GuardDemo")
    try:
        # Try jumping directly from IDLE to DONE (not allowed)
        sm.transition(AgentState.DONE, "skipping everything")
    except ValueError as e:
        print(f"  ✓ Blocked: {e}")


# ── Concurrent agents demo ────────────────────────────────────────────────────

def demo_concurrent():
    print("\n── Concurrent stateful agents ──")
    agents  = [StatefulAgent(f"Agent{i}") for i in range(3)]
    tasks   = [
        "What is the price of the Pro plan?",
        "Calculate annual cost of Pro",
        "What is the weather in London?",
    ]
    results = [None] * len(agents)

    def _run(i):
        results[i] = agents[i].run(tasks[i])

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(len(agents))]
    for t in threads: t.start()
    for t in threads: t.join()

    print("\n  Results:")
    for i, r in enumerate(results):
        print(f"    Agent{i}: {r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Agent State Machine — Lifecycle Management")
    print("=" * 65)

    # ── Single agent run ──────────────────────────────────────────────────────
    print("\n── Single agent — state transitions ──")
    agent  = StatefulAgent("Assistant")
    tasks  = [
        "What are TechCorp's pricing plans?",
        "What is the weather in London?",
        "Just say hello",
    ]
    for task in tasks:
        print(f"\n  Task: '{task}'")
        result = agent.run(task)
        print(f"  Result: {result}")

    agent.sm.print_history()

    demo_invalid_transition()
    demo_concurrent()

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • State machine: validates transitions, prevents illegal operations")
    print("  • WAITING_FOR_TOOL: explicit state prevents race conditions")
    print("  • reset() after DONE/ERROR makes agent reusable")
    print("  • on_enter() callbacks = lightweight hooks (logging, metrics)")
    print("  • Thread-safe with a lock — safe for concurrent agents")
    print("  • LangGraph, AutoGen use state machines internally")


if __name__ == "__main__":
    main()
