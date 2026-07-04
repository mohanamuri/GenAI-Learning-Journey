# Author: Mohan Raju Amuri
"""
07_agent_workflows.py — Structured execution graphs

Beyond pipelines: real workflows have branches, loops, and parallelism.

Workflow types:
  1. Sequential  — A → B → C (each step needs the previous)
  2. Parallel    — A || B || C, then merge at join node
  3. Conditional — if condition → branch X, else → branch Y
  4. Loop        — repeat step until quality threshold met

This example implements a lightweight DAG workflow engine
(no external library — pure Python dataclasses + threading).

No Ollama required — demonstrates workflow mechanics with mock agents.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor


# ── Workflow data model ───────────────────────────────────────────────────────

@dataclass
class WorkflowNode:
    name:      str
    fn:        Callable[[dict], Any]           # receives shared context, returns result
    depends:   list[str]   = field(default_factory=list)   # upstream node names
    condition: Optional[Callable[[dict], bool]] = None     # if None, always runs
    max_retries: int = 1


@dataclass
class WorkflowResult:
    node:    str
    output:  Any
    skipped: bool  = False
    retries: int   = 0
    elapsed: float = 0.0
    error:   str   = ""


# ── Workflow engine ───────────────────────────────────────────────────────────

class WorkflowEngine:
    """
    Executes a DAG of WorkflowNodes.
    - Respects dependencies (topological order)
    - Runs independent nodes in parallel (threads)
    - Evaluates conditions before running nodes
    - Retries on failure up to node.max_retries
    """

    def __init__(self, nodes: list[WorkflowNode]):
        self.nodes    = {n.name: n for n in nodes}
        self.context  = {}            # shared state across all nodes
        self.results: dict[str, WorkflowResult] = {}
        self._lock    = threading.Lock()
        self._done    = {n.name: threading.Event() for n in nodes}

    def _run_node(self, node: WorkflowNode) -> None:
        # Wait for all dependencies
        for dep in node.depends:
            self._done[dep].wait()

        # Check condition (skip if returns False)
        if node.condition and not node.condition(self.context):
            r = WorkflowResult(node=node.name, output=None, skipped=True)
            with self._lock:
                self.results[node.name] = r
            self._done[node.name].set()
            print(f"  ○ [{node.name}] skipped (condition False)")
            return

        # Execute with retries
        start   = time.perf_counter()
        retries = 0
        output  = None
        error   = ""

        for attempt in range(node.max_retries):
            try:
                output  = node.fn(dict(self.context))   # pass snapshot of context
                with self._lock:
                    self.context[node.name] = output    # write result to shared context
                error = ""
                break
            except Exception as e:
                error   = str(e)
                retries = attempt + 1
                if attempt < node.max_retries - 1:
                    time.sleep(0.05 * (attempt + 1))

        elapsed = time.perf_counter() - start
        r = WorkflowResult(node=node.name, output=output, retries=retries,
                           elapsed=elapsed, error=error)
        with self._lock:
            self.results[node.name] = r
        self._done[node.name].set()

        status = "✓" if not error else "✗"
        skip   = f" (retry {retries})" if retries else ""
        print(f"  {status} [{node.name}] elapsed={elapsed:.2f}s{skip}")

    def run(self, initial_context: dict = None) -> dict[str, WorkflowResult]:
        if initial_context:
            self.context.update(initial_context)

        with ThreadPoolExecutor(max_workers=len(self.nodes)) as pool:
            for node in self.nodes.values():
                pool.submit(self._run_node, node)

        return self.results

    def summary(self) -> None:
        print("\n  Workflow summary:")
        for name, r in self.results.items():
            if r.skipped:
                print(f"    ○ {name}: skipped")
            elif r.error:
                print(f"    ✗ {name}: ERROR — {r.error}")
            else:
                short = str(r.output)[:50] if r.output else "—"
                print(f"    ✓ {name} ({r.elapsed:.2f}s): {short}")


# ── Demo workflows ────────────────────────────────────────────────────────────

# Simulated work functions
def step_fetch_user(ctx)     -> str:  time.sleep(0.1); return "user=Alice, plan=Pro"
def step_fetch_pricing(ctx)  -> str:  time.sleep(0.2); return "Pro=$499/mo, Starter=$99/mo"
def step_fetch_sla(ctx)      -> str:  time.sleep(0.15); return "Pro SLA=99.9%"
def step_compute(ctx)        -> str:
    pricing = ctx.get("fetch_pricing", "Pro=$499/mo")
    return "Annual Pro = $5,988"
def step_format_user(ctx)    -> str:  return f"Formatted: {ctx.get('fetch_user','')}"
def step_draft(ctx)          -> str:
    return (f"Draft: {ctx.get('fetch_user','')} | "
            f"{ctx.get('fetch_pricing','')} | {ctx.get('compute','')}")
def step_review(ctx)         -> str:
    draft = ctx.get("draft", "")
    return "Approved" if len(draft) > 30 else "Rejected"
def step_send(ctx)           -> str:
    return f"Sent to Alice: {ctx.get('draft','')[:60]}"
def step_archive(ctx)        -> str:  return f"Archived: {ctx.get('review','')}"


def demo_sequential():
    print("\n── 1. Sequential workflow (A → B → C) ──")
    nodes = [
        WorkflowNode("fetch_user",    fn=step_fetch_user),
        WorkflowNode("format_user",   fn=step_format_user,   depends=["fetch_user"]),
        WorkflowNode("draft",         fn=step_draft,          depends=["format_user"]),
        WorkflowNode("review",        fn=step_review,         depends=["draft"]),
    ]
    engine = WorkflowEngine(nodes)
    engine.run()
    engine.summary()


def demo_parallel():
    print("\n── 2. Parallel workflow (A || B || C → merge) ──")
    nodes = [
        WorkflowNode("fetch_user",    fn=step_fetch_user),     # parallel
        WorkflowNode("fetch_pricing", fn=step_fetch_pricing),  # parallel
        WorkflowNode("fetch_sla",     fn=step_fetch_sla),      # parallel
        WorkflowNode("compute",       fn=step_compute,
                     depends=["fetch_pricing"]),
        WorkflowNode("draft",         fn=step_draft,
                     depends=["fetch_user", "fetch_pricing", "fetch_sla", "compute"]),
    ]
    start  = time.perf_counter()
    engine = WorkflowEngine(nodes)
    engine.run()
    elapsed = time.perf_counter() - start
    engine.summary()
    print(f"\n  Total wall time: {elapsed:.2f}s (fetch nodes ran in parallel)")


def demo_conditional():
    print("\n── 3. Conditional workflow (branch on plan type) ──")

    def step_check_plan(ctx)   -> str:  return "Pro"
    def step_pro_features(ctx) -> str:  return "GPU training, 1 TB storage enabled"
    def step_basic_features(ctx)-> str: return "CPU training, 100 GB storage"
    def step_notify(ctx)       -> str:
        features = ctx.get("pro_features") or ctx.get("basic_features", "")
        return f"Notified user: {features}"

    is_pro = lambda ctx: ctx.get("check_plan") == "Pro"

    nodes = [
        WorkflowNode("check_plan",     fn=step_check_plan),
        WorkflowNode("pro_features",   fn=step_pro_features,   depends=["check_plan"],
                     condition=is_pro),
        WorkflowNode("basic_features", fn=step_basic_features, depends=["check_plan"],
                     condition=lambda ctx: not is_pro(ctx)),
        WorkflowNode("notify",         fn=step_notify,
                     depends=["check_plan", "pro_features", "basic_features"]),
    ]
    engine = WorkflowEngine(nodes)
    engine.run()
    engine.summary()


def demo_loop():
    print("\n── 4. Loop workflow (retry until quality met) ──")
    attempt = [0]   # mutable ref for closure

    def step_generate(ctx) -> str:
        attempt[0] += 1
        if attempt[0] < 3:
            return "short"   # quality check will fail
        return "TechCorp Pro plan: $499/month, 25 users, GPU training, 99.9% SLA."

    def step_quality(ctx) -> str:
        output = ctx.get("generate", "")
        score  = len(output) // 10
        return f"score={min(score, 10)}"

    # Simulate a loop by chaining generate → quality → generate → quality
    # Real loops use a while loop around the engine or a loop node type
    MAX_LOOP = 4
    for i in range(MAX_LOOP):
        engine = WorkflowEngine([
            WorkflowNode("generate", fn=step_generate),
            WorkflowNode("quality",  fn=step_quality, depends=["generate"]),
        ])
        engine.run()
        score_str = engine.context.get("quality", "score=0")
        score     = int(score_str.split("=")[1]) if "=" in score_str else 0
        output    = engine.context.get("generate", "")
        print(f"    Loop {i+1}: score={score} | output='{output[:50]}'")
        if score >= 5:
            print(f"    ✓ Quality threshold met after {i+1} iteration(s)")
            break
    else:
        print("    ✗ Max iterations reached without meeting quality threshold")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Agent Workflows — Sequential, Parallel, Conditional, Loop")
    print("=" * 65)

    demo_sequential()
    demo_parallel()
    demo_conditional()
    demo_loop()

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Sequential: depends=[] chain → use when each step needs prior output")
    print("  • Parallel: no shared depends → run simultaneously, merge at join")
    print("  • Conditional: condition= fn → skip branch if condition False")
    print("  • Loop: while + retry → repeat until quality threshold met")
    print("  • DAG engine = topological sort + threading for parallel branches")
    print("  • LangGraph, Prefect, Airflow all implement these same patterns")


if __name__ == "__main__":
    main()
