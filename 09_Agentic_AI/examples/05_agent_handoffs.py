# Author: Mohan Raju Amuri
"""
05_agent_handoffs.py — Delegating tasks between agents

Handoff: an agent detects it can't (or shouldn't) handle a task
         and transfers control to a more suitable specialist agent.

Three handoff types:
  1. Hard handoff   — transfer full context, original agent stops
  2. Soft handoff   — spawn sub-agent, get result, continue
  3. Escalation     — pass up to a higher-authority agent

Real-world:
  - Customer support: general agent → billing specialist
  - Code assistant: general agent → security reviewer
  - Triage: intake agent → domain-specific agent

No Ollama required — uses intent detection to route.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Handoff protocol ──────────────────────────────────────────────────────────

@dataclass
class HandoffRequest:
    from_agent:  str
    to_agent:    str
    reason:      str
    context:     dict = field(default_factory=dict)   # preserved across handoff
    handoff_type: str = "hard"                         # hard | soft | escalation


@dataclass
class AgentResponse:
    agent:        str
    content:      str
    handled:      bool = True
    handoff:      Optional[HandoffRequest] = None


# ── Specialist agents ─────────────────────────────────────────────────────────

class BillingAgent:
    name = "BillingAgent"

    ANSWERS = {
        "pricing":  "Our plans: Starter $99/mo, Pro $499/mo, Enterprise custom.",
        "invoice":  "You can download invoices from Settings → Billing → Invoices.",
        "refund":   "Refunds are processed within 5 business days to your original payment.",
        "upgrade":  "Upgrade anytime from Settings → Plan. Prorated immediately.",
    }

    def handle(self, context: dict) -> AgentResponse:
        query   = context.get("query", "").lower()
        key     = next((k for k in self.ANSWERS if k in query), None)
        content = self.ANSWERS[key] if key else "I'll connect you with our billing team."
        print(f"  [{self.name}] handled billing query")
        return AgentResponse(agent=self.name, content=content)


class TechnicalAgent:
    name = "TechnicalAgent"

    def handle(self, context: dict) -> AgentResponse:
        query   = context.get("query", "")
        print(f"  [{self.name}] handling technical query")
        if "api" in query.lower():
            return AgentResponse(agent=self.name,
                content="API docs at docs.techcorp.com/api. Auth: Bearer token in header.")
        if "error" in query.lower() or "bug" in query.lower():
            return AgentResponse(agent=self.name,
                content="Please share your error code. Common fix: clear cache and retry.")
        return AgentResponse(agent=self.name,
            content=f"Technical answer: {query[:60]} — checked our runbooks, no exact match.")


class SecurityAgent:
    name = "SecurityAgent"

    def handle(self, context: dict) -> AgentResponse:
        print(f"  [{self.name}] handling security query")
        return AgentResponse(agent=self.name,
            content="SOC 2 Type II certified. AES-256 at rest, TLS 1.3 in transit. "
                    "GDPR and CCPA compliant. Pen-tested annually.")


class EscalationAgent:
    name = "EscalationAgent"

    def handle(self, context: dict) -> AgentResponse:
        print(f"  [{self.name}] escalating to human — reason: {context.get('reason','unknown')}")
        return AgentResponse(agent=self.name,
            content="I'm connecting you with a human specialist. "
                    f"Reference ID: ESC-{abs(hash(context.get('query',''))[:4] if isinstance(context.get('query',''), str) else 1234) % 9999:04d}. "
                    "Expected wait: < 2 min.")


SPECIALIST_REGISTRY = {
    "BillingAgent":    BillingAgent(),
    "TechnicalAgent":  TechnicalAgent(),
    "SecurityAgent":   SecurityAgent(),
    "EscalationAgent": EscalationAgent(),
}


# ── General agent with handoff logic ──────────────────────────────────────────

class GeneralAgent:
    """
    First-contact agent. Handles simple queries directly.
    Routes specialized or complex queries to the right specialist.
    """
    name = "GeneralAgent"

    # Intent → specialist routing rules
    ROUTING = [
        (["price", "plan", "cost", "billing", "invoice", "refund", "upgrade"],  "BillingAgent"),
        (["api", "sdk", "error", "bug", "crash", "code", "integrate"],          "TechnicalAgent"),
        (["secure", "security", "encrypt", "gdpr", "compliance", "soc"],        "SecurityAgent"),
        (["angry", "urgent", "escalate", "manager", "lawsuit", "unacceptable"], "EscalationAgent"),
    ]

    def _detect_specialist(self, query: str) -> Optional[str]:
        q = query.lower()
        for keywords, specialist in self.ROUTING:
            if any(k in q for k in keywords):
                return specialist
        return None

    def handle(self, query: str, context: dict = None) -> AgentResponse:
        ctx  = context or {"query": query}
        ctx["query"] = query
        print(f"\n  [{self.name}] received: '{query[:60]}'")

        # Try to handle simple queries directly
        if any(w in query.lower() for w in ["hello", "hi", "help", "what can you do"]):
            return AgentResponse(
                agent=self.name,
                content="Hi! I can help with billing, technical issues, or security questions. What do you need?"
            )

        # Detect if specialist is needed
        specialist = self._detect_specialist(query)
        if specialist:
            print(f"  [{self.name}] → handoff to {specialist}")
            return AgentResponse(
                agent=self.name,
                content="",
                handled=False,
                handoff=HandoffRequest(
                    from_agent=self.name,
                    to_agent=specialist,
                    reason=f"Query matched specialist: {specialist}",
                    context=ctx,
                )
            )

        return AgentResponse(
            agent=self.name,
            content=f"I can help with that: '{query[:80]}'. Could you be more specific?"
        )


# ── Router: executes handoffs ─────────────────────────────────────────────────

class AgentRouter:
    """
    Receives responses from GeneralAgent.
    If a handoff is requested, dispatches to the specialist.
    Logs the full handoff trace.
    """

    def __init__(self):
        self.general   = GeneralAgent()
        self.trace: list[dict] = []

    def route(self, query: str) -> str:
        ctx      = {"query": query}
        response = self.general.handle(query, ctx)
        self.trace.append({"agent": response.agent, "type": "direct" if response.handled else "handoff"})

        # Follow handoff chain (max depth 3)
        depth = 0
        while response.handoff and depth < 3:
            depth   += 1
            hw       = response.handoff
            print(f"  [Router] {hw.handoff_type} handoff: {hw.from_agent} → {hw.to_agent} (reason: {hw.reason})")

            specialist = SPECIALIST_REGISTRY.get(hw.to_agent)
            if not specialist:
                return f"Error: no specialist '{hw.to_agent}' registered."

            response = specialist.handle(hw.context)
            self.trace.append({"agent": response.agent, "type": hw.handoff_type})

        return response.content

    def print_trace(self):
        path = " → ".join(f"{t['agent']}({t['type']})" for t in self.trace)
        print(f"  Trace: {path}")
        self.trace.clear()


# ── Soft handoff: spawn sub-agent, get result, continue ──────────────────────

class ResearchWithFactCheck:
    """
    Soft handoff example: ResearchAgent spawns FactChecker as a sub-agent,
    gets its result, incorporates it, then continues.
    """

    class _FactChecker:
        def check(self, claim: str) -> bool:
            # Simplified: flag any claim with a number as potentially checkable
            return bool(re.search(r"\d", claim))

    def run(self, topic: str) -> str:
        # Step 1: research (primary agent work)
        raw_answer = f"TechCorp Pro plan costs $499/month with 99.9% SLA."
        print(f"  [ResearchAgent] draft: '{raw_answer[:50]}'")

        # Step 2: soft handoff — spawn fact-checker, get result, continue
        fc     = self._FactChecker()
        check  = fc.check(raw_answer)
        status = "fact-checked ✓" if check else "unverified"
        print(f"  [FactChecker] sub-agent result: {status}")

        # Step 3: incorporate sub-agent result, return final answer
        return f"{raw_answer} [{status}]"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Agent Handoffs — Routing, Delegation, Escalation")
    print("=" * 65)

    router = AgentRouter()
    queries = [
        "Hello, what can you help me with?",
        "What is the price of the Pro plan?",
        "I'm getting a 401 error from your API",
        "Are you GDPR compliant and SOC 2 certified?",
        "This is unacceptable! I want to speak to a manager NOW!",
        "How do I export data?",
    ]

    print("\n── Hard Handoffs (GeneralAgent → Specialist) ──")
    for q in queries:
        answer = router.route(q)
        router.print_trace()
        print(f"  Answer: {answer}\n")

    print("── Soft Handoff (sub-agent spawned inline) ──")
    agent = ResearchWithFactCheck()
    result = agent.run("TechCorp pricing")
    print(f"  Final: {result}")

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Hard handoff: context transferred, original agent done")
    print("  • Soft handoff: sub-agent result folded back in, primary continues")
    print("  • Escalation: human-in-the-loop trigger")
    print("  • Route on intent, not on content length")
    print("  • Always log the handoff trace for debugging")
    print("  • Cap handoff depth to prevent loops")


if __name__ == "__main__":
    main()
