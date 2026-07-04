# Author: Mohan Raju Amuri
"""
03_agent_communication.py — How agents talk to each other

Three communication patterns:
  1. Direct call  — agent_a calls agent_b directly (tight coupling)
  2. Message bus  — agents publish/subscribe to a shared queue (loose coupling)
  3. Shared state — agents read/write a shared memory dict (simplest)

Which to use:
  - Direct call:    small pipelines (2-3 agents), strict sequence
  - Message bus:    event-driven systems, decoupled parallel agents
  - Shared state:   short-lived coordination, same process

No Ollama required — pure Python messaging mechanics.
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable


# ══════════════════════════════════════════════════════════════
# Pattern 1: Direct call (tight coupling)
# ══════════════════════════════════════════════════════════════

class DirectAgentA:
    """Agent A: collects user input, passes directly to Agent B."""

    def __init__(self, agent_b):
        self._b = agent_b      # hard reference — tight coupling

    def handle(self, text: str) -> str:
        print(f"  [AgentA] received: '{text}'")
        enriched = f"[A→] {text} | timestamp={int(time.time())}"
        return self._b.handle(enriched)    # direct call


class DirectAgentB:
    """Agent B: processes enriched message, passes to Agent C."""

    def __init__(self, agent_c):
        self._c = agent_c

    def handle(self, text: str) -> str:
        print(f"  [AgentB] received: '{text[:60]}'")
        processed = text.upper()
        return self._c.handle(processed)


class DirectAgentC:
    """Agent C: final handler."""

    def handle(self, text: str) -> str:
        print(f"  [AgentC] received: '{text[:60]}'")
        return f"Result: {text[:60]}"


def demo_direct():
    print("\n── Pattern 1: Direct Call ──")
    c = DirectAgentC()
    b = DirectAgentB(c)
    a = DirectAgentA(b)
    result = a.handle("hello world")
    print(f"  Final: {result}")
    print("  ⚠ Tight coupling: swapping B requires editing A")


# ══════════════════════════════════════════════════════════════
# Pattern 2: Message Bus (pub/sub)
# ══════════════════════════════════════════════════════════════

@dataclass
class BusMessage:
    topic:   str
    sender:  str
    payload: Any
    msg_id:  int = field(default_factory=lambda: int(time.time() * 1000))


class MessageBus:
    """
    Simple in-process pub/sub message bus.

    Agents subscribe to topics.
    Any agent can publish to any topic.
    Bus delivers to all subscribers of that topic.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: list[BusMessage] = []

    def subscribe(self, topic: str, handler: Callable) -> None:
        self._subscribers.setdefault(topic, []).append(handler)
        print(f"  [Bus] subscribed '{handler.__self__.name}' to '{topic}'")

    def publish(self, msg: BusMessage) -> None:
        self._history.append(msg)
        handlers = self._subscribers.get(msg.topic, [])
        print(f"  [Bus] '{msg.sender}' → topic='{msg.topic}' ({len(handlers)} subscriber(s))")
        for h in handlers:
            h(msg)

    def stats(self) -> str:
        return f"Bus: {len(self._history)} messages published, {sum(len(v) for v in self._subscribers.values())} subscriptions"


class BusAgent:
    """Agent that communicates via a message bus."""

    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus  = bus
        self.inbox: list[BusMessage] = []

    def on_message(self, msg: BusMessage) -> None:
        """Called by the bus when a subscribed message arrives."""
        self.inbox.append(msg)
        self._handle(msg)

    def _handle(self, msg: BusMessage) -> None:
        raise NotImplementedError

    def publish(self, topic: str, payload: Any) -> None:
        self.bus.publish(BusMessage(topic=topic, sender=self.name, payload=payload))


class ExtractorAgent(BusAgent):
    """Extracts keywords from raw text, publishes to 'keywords' topic."""

    def _handle(self, msg: BusMessage) -> None:
        text     = str(msg.payload)
        keywords = [w for w in text.lower().split() if len(w) > 4][:5]
        print(f"  [{self.name}] extracted keywords: {keywords}")
        self.publish("keywords", keywords)


class ClassifierAgent(BusAgent):
    """Classifies a keyword list into a category."""

    CATEGORIES = {
        frozenset(["price", "plan", "cost", "monthly", "annual"]): "pricing",
        frozenset(["secure", "security", "encrypt", "compliant"]):  "security",
        frozenset(["trial", "free", "start", "begin"]):              "onboarding",
    }

    def _handle(self, msg: BusMessage) -> None:
        kws = set(str(k).lower() for k in msg.payload)
        category = "general"
        for tag_set, cat in self.CATEGORIES.items():
            if kws & tag_set:
                category = cat
                break
        print(f"  [{self.name}] classified as: '{category}'")
        self.publish("category", category)


class ResponderAgent(BusAgent):
    """Generates a response based on the category."""

    RESPONSES = {
        "pricing":    "Our plans start at $99/month. Visit /pricing for details.",
        "security":   "We're SOC 2 Type II certified with AES-256 encryption.",
        "onboarding": "Start with our 14-day free trial — no credit card needed.",
        "general":    "How can I help you today? Feel free to ask anything.",
    }

    def _handle(self, msg: BusMessage) -> None:
        response = self.RESPONSES.get(str(msg.payload), self.RESPONSES["general"])
        print(f"  [{self.name}] response: '{response}'")
        self.publish("response", response)


def demo_message_bus():
    print("\n── Pattern 2: Message Bus (pub/sub) ──")
    bus      = MessageBus()
    extractor  = ExtractorAgent("Extractor",   bus)
    classifier = ClassifierAgent("Classifier", bus)
    responder  = ResponderAgent("Responder",   bus)

    # Wire up subscriptions
    bus.subscribe("raw_text",  extractor.on_message)
    bus.subscribe("keywords",  classifier.on_message)
    bus.subscribe("category",  responder.on_message)

    # Trigger the chain by publishing to the first topic
    bus.publish(BusMessage(topic="raw_text", sender="user",
                           payload="What is the monthly cost of your Pro plan?"))

    print(f"\n  {bus.stats()}")
    print("  ✓ Loose coupling: swap Classifier without editing Extractor")


# ══════════════════════════════════════════════════════════════
# Pattern 3: Shared state
# ══════════════════════════════════════════════════════════════

class SharedState:
    """
    Thread-safe key-value store for agent coordination.
    Agents read/write keys; state acts as a blackboard.
    """

    def __init__(self):
        self._state: dict[str, Any] = {}
        self._lock  = threading.Lock()

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._state)


def agent_parse(state: SharedState, raw: str) -> None:
    tokens = raw.lower().split()
    state.set("tokens", tokens)
    state.set("token_count", len(tokens))
    print(f"  [ParseAgent] tokens={len(tokens)}")


def agent_classify(state: SharedState) -> None:
    tokens = state.get("tokens", [])
    if any(t in tokens for t in ["price", "cost", "plan"]):
        state.set("intent", "pricing")
    elif any(t in tokens for t in ["secure", "safe", "encrypt"]):
        state.set("intent", "security")
    else:
        state.set("intent", "general")
    print(f"  [ClassifyAgent] intent='{state.get('intent')}'")


def agent_respond(state: SharedState) -> None:
    intent = state.get("intent", "general")
    answers = {
        "pricing":  "Starter $99 | Pro $499 | Enterprise custom",
        "security": "SOC 2, AES-256, TLS 1.3",
        "general":  "Ask me anything about TechCorp.",
    }
    state.set("response", answers[intent])
    print(f"  [RespondAgent] response='{answers[intent]}'")


def demo_shared_state():
    print("\n── Pattern 3: Shared State (blackboard) ──")
    state = SharedState()
    raw   = "What is the monthly price of the Pro plan?"

    agent_parse(state, raw)
    agent_classify(state)
    agent_respond(state)

    print(f"\n  Final state: {state.snapshot()}")
    print("  ✓ Simple, no coupling — agents only share the state object")
    print("  ⚠ State can get messy with many agents writing different keys")


# ── Comparison ────────────────────────────────────────────────────────────────

def print_comparison():
    print("""
╔═════════════════╦═══════════════════╦═══════════════════════╦═════════════════╗
║ Pattern         ║ Coupling          ║ Best for              ║ Drawback        ║
╠═════════════════╬═══════════════════╬═══════════════════════╬═════════════════╣
║ Direct call     ║ Tight             ║ 2–3 fixed agents      ║ Hard to swap    ║
║ Message bus     ║ Loose (pub/sub)   ║ Event-driven, parallel║ More boilerplate║
║ Shared state    ║ None (blackboard) ║ Same-process agents   ║ Key collisions  ║
╚═════════════════╩═══════════════════╩═══════════════════════╩═════════════════╝
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Agent Communication Patterns")
    print("=" * 65)

    print_comparison()
    demo_direct()
    demo_message_bus()
    demo_shared_state()

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Direct call: fast to build, painful to maintain at scale")
    print("  • Message bus: production default — decoupled, observable")
    print("  • Shared state: great for short coordination within one process")
    print("  • Most frameworks (LangGraph, AutoGen) use shared state + bus")
    print("  • Always use a lock when multiple agents write shared state")


if __name__ == "__main__":
    main()
