# Author: Mohan Raju Amuri
"""
06_supervisor_pattern.py — Supervisor agent monitors and corrects workers

SupervisorAgent:
  1. Assigns task to WorkerAgent
  2. Evaluates the output quality
  3. If quality < threshold → sends feedback, asks worker to revise
  4. If quality >= threshold → approves and returns
  5. If worker fails MAX_ROUNDS times → escalates

Why supervisors?
  - Single-agent output is often suboptimal on the first pass
  - LLMs improve significantly when given specific feedback
  - Prevents bad output from reaching users automatically

Used in: AutoGen (AssistantAgent + UserProxyAgent), LangGraph self-critique

Runs with Ollama (llama3.2:3b) or falls back to mock revision.
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Optional
# ============================================================
# Falls back to mock if Ollama is not running.
# ============================================================

import re
from dataclasses import dataclass, field


MAX_ROUNDS   = 3        # max revision rounds before escalation
PASS_SCORE   = 7        # score >= this → approved
SCORE_RANGE  = (0, 10)


# ── Quality evaluator ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    score:    int
    passed:   bool
    issues:   list[str] = field(default_factory=list)
    feedback: str       = ""


def evaluate_output(task: str, output: str) -> EvalResult:
    """
    Rule-based quality evaluator.
    In production: replace with an LLM judge.
    """
    score  = 0
    issues = []

    if len(output) > 50:
        score += 2
    else:
        issues.append("Response too short (< 50 chars)")

    if len(output) > 150:
        score += 1

    # check that at least one number appears (for factual answers)
    if re.search(r"\$\d+|\d+%|\d+\.\d+", output):
        score += 3
    else:
        issues.append("Missing specific figures (numbers, prices, percentages)")

    # check it addresses the question topic
    topic_words = set(task.lower().split()) - {"what", "how", "is", "the", "a", "an", "are"}
    overlap     = sum(1 for w in topic_words if w in output.lower())
    if overlap >= 2:
        score += 2
    else:
        issues.append("Response doesn't address the question topic")

    # check for hedging without substance
    if "i don't know" in output.lower() or "i cannot" in output.lower():
        score -= 2
        issues.append("Response hedges without providing information")

    score  = max(SCORE_RANGE[0], min(SCORE_RANGE[1], score))
    passed = score >= PASS_SCORE

    feedback = ""
    if not passed and issues:
        feedback = "Please revise. Issues:\n" + "\n".join(f"  - {i}" for i in issues)

    return EvalResult(score=score, passed=passed, issues=issues, feedback=feedback)


# ── Worker agent ──────────────────────────────────────────────────────────────

class WorkerAgent:
    """Generates answers. Can use Ollama or fall back to mock."""

    # Mock responses that improve over rounds
    MOCK_RESPONSES = {
        0: "TechCorp has plans.",
        1: "TechCorp offers Starter and Pro plans with various pricing.",
        2: "TechCorp plans: Starter $99/month (5 users), Pro $499/month (25 users, GPU training), Enterprise custom pricing with dedicated infrastructure.",
        3: "TechCorp plans: Starter $99/month (5 users, 100 GB, CPU), Pro $499/month (25 users, 1 TB, GPU), Enterprise (custom, unlimited, dedicated infra). Pro includes 99.9% SLA.",
    }

    def __init__(self):
        self.llm    = self._init_llm()
        self._round = 0

    def _init_llm(self):
        try:
            from openai import OpenAI
            c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            c.models.list()
            print("  WorkerAgent: connected to Ollama")
            return c
        except Exception:
            print("  WorkerAgent: Ollama not running → using mock responses")
            return None

    def generate(self, task: str, feedback: str = "") -> str:
        self._round += 1
        print(f"    [Worker] generating (round {self._round})")

        if self.llm:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for TechCorp. Answer concisely with specific details."},
                {"role": "user",   "content": task},
            ]
            if feedback:
                messages.append({"role": "assistant", "content": self._last_output or "..."})
                messages.append({"role": "user",      "content": f"Please revise. Feedback:\n{feedback}"})
            try:
                resp = self.llm.chat.completions.create(
                    model="llama3.2:3b", messages=messages,
                    temperature=0.3, max_tokens=200,
                )
                output = resp.choices[0].message.content.strip()
                self._last_output = output
                return output
            except Exception as e:
                print(f"    [Worker] LLM error: {e}")

        # mock: each round returns a better response
        idx    = min(self._round, len(self.MOCK_RESPONSES) - 1)
        output = self.MOCK_RESPONSES[idx]
        self._last_output = output
        return output


# ── Supervisor agent ──────────────────────────────────────────────────────────

@dataclass
class SupervisionLog:
    task:    str
    rounds:  list[dict] = field(default_factory=list)
    final:   str        = ""
    status:  str        = "pending"   # approved | escalated


class SupervisorAgent:
    """
    Supervises a WorkerAgent through iterative revision.

    Loop:
      assign task → worker generates → evaluate → (approve OR feedback+revise)
    """

    def __init__(self, worker: WorkerAgent, max_rounds: int = MAX_ROUNDS):
        self.worker     = worker
        self.max_rounds = max_rounds

    def supervise(self, task: str) -> SupervisionLog:
        log      = SupervisionLog(task=task)
        feedback = ""
        print(f"\n  [Supervisor] task: '{task[:60]}'")

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n  [Supervisor] round {round_num}/{self.max_rounds}")

            # Worker generates
            output = self.worker.generate(task, feedback=feedback)
            print(f"    Output: '{output[:80]}'")

            # Supervisor evaluates
            eval_result = evaluate_output(task, output)
            print(f"    Score: {eval_result.score}/10 — {'✓ Approved' if eval_result.passed else '✗ Needs revision'}")

            log.rounds.append({
                "round":    round_num,
                "output":   output,
                "score":    eval_result.score,
                "passed":   eval_result.passed,
                "issues":   eval_result.issues,
            })

            if eval_result.passed:
                log.final  = output
                log.status = "approved"
                print(f"  [Supervisor] ✓ approved after {round_num} round(s)")
                return log

            # Prepare feedback for next round
            feedback = eval_result.feedback
            if round_num < self.max_rounds:
                print(f"    Feedback: {feedback[:100]}")

        # Max rounds exceeded → escalate
        last_output = log.rounds[-1]["output"] if log.rounds else ""
        log.final   = last_output
        log.status  = "escalated"
        print(f"  [Supervisor] ✗ escalating after {self.max_rounds} failed round(s)")
        return log

    def print_log(self, log: SupervisionLog) -> None:
        print(f"\n  Supervision summary for: '{log.task[:50]}'")
        print(f"  Status: {log.status} | Rounds: {len(log.rounds)}")
        for r in log.rounds:
            tick = "✓" if r["passed"] else "✗"
            print(f"    Round {r['round']}: score={r['score']}/10 {tick}")
        print(f"  Final output: '{log.final[:100]}'")


# ── Multi-agent supervision ───────────────────────────────────────────────────

class ReviewBoard:
    """
    Multiple supervisors evaluate one worker output independently.
    Final decision by majority vote.
    """

    def __init__(self, n_supervisors: int = 3):
        self.supervisors = [SupervisorAgent(WorkerAgent(), max_rounds=1)
                            for _ in range(n_supervisors)]

    def evaluate(self, task: str) -> dict:
        # Each supervisor generates their own worker output and evaluates
        worker = WorkerAgent()
        output = worker.generate(task)
        print(f"\n  [ReviewBoard] evaluating output: '{output[:60]}'")

        scores  = []
        verdicts = []
        for i, sup in enumerate(self.supervisors):
            ev = evaluate_output(task, output)
            scores.append(ev.score)
            verdicts.append("approved" if ev.passed else "rejected")
            print(f"    Supervisor {i+1}: score={ev.score} → {verdicts[-1]}")

        approved = verdicts.count("approved")
        decision = "approved" if approved > len(verdicts) / 2 else "rejected"
        print(f"  [ReviewBoard] majority vote: {approved}/{len(verdicts)} approved → {decision}")
        return {"output": output, "scores": scores, "decision": decision}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Supervisor Pattern — Iterative Quality Control")
    print("=" * 65)

    # ── Single supervisor ────────────────────────────────────────────────────
    print("\n── Single Supervisor (iterative revision) ──")
    supervisor = SupervisorAgent(WorkerAgent(), max_rounds=MAX_ROUNDS)

    tasks = [
        "What are TechCorp's pricing plans?",
        "What security certifications does TechCorp have?",
    ]
    for task in tasks:
        log = supervisor.supervise(task)
        supervisor.print_log(log)
        supervisor.worker._round = 0   # reset for next demo

    # ── Multi-supervisor review board ────────────────────────────────────────
    print("\n── Review Board (majority vote) ──")
    board  = ReviewBoard(n_supervisors=3)
    result = board.evaluate("What are TechCorp's SLA guarantees?")
    print(f"\n  Final decision: {result['decision']}")
    print(f"  Scores: {result['scores']}")

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Supervisor: assign → evaluate → feedback → revise loop")
    print("  • Max rounds prevents infinite revision spirals")
    print("  • Specific feedback > generic 'try again'")
    print("  • Review board: multiple evaluators + majority vote")
    print("  • In production: replace rule-based eval with LLM judge")
    print("  • Track revision history for audit + model improvement")


if __name__ == "__main__":
    main()
