# Author: Mohan Raju Amuri
"""
04_parallel_agents.py — Running agents concurrently

Sequential:  Agent A → Agent B → Agent C  →  total = A+B+C seconds
Parallel:    Agent A  }
             Agent B  } all at once        →  total = max(A, B, C) seconds
             Agent C  }

When to parallelize:
  - Agents are independent (no output of A needed by B)
  - Tasks are I/O-bound (LLM calls, DB queries, API requests)
  - Speed matters more than simplicity

Python tools: concurrent.futures.ThreadPoolExecutor (I/O bound)
              multiprocessing.Pool              (CPU bound, rare for agents)

No Ollama required — uses simulated latency to show speedup.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


# ── Simulated LLM agent ───────────────────────────────────────────────────────

@dataclass
class AgentTask:
    name:    str
    query:   str
    latency: float   # simulated seconds of "LLM thinking"
    result:  str     = ""
    elapsed: float   = 0.0
    error:   str     = ""


def run_agent(task: AgentTask) -> AgentTask:
    """Simulates an agent doing work with a fixed latency."""
    start = time.perf_counter()
    try:
        time.sleep(task.latency)   # simulated LLM / API call
        task.result  = f"[{task.name}] answer to '{task.query[:30]}'"
        task.elapsed = time.perf_counter() - start
        print(f"  ✓ {task.name} done in {task.elapsed:.2f}s")
    except Exception as e:
        task.error   = str(e)
        task.elapsed = time.perf_counter() - start
        print(f"  ✗ {task.name} failed: {e}")
    return task


# ── Sequential execution ───────────────────────────────────────────────────────

def run_sequential(tasks: list[AgentTask]) -> tuple[list[AgentTask], float]:
    """Run all agents one after another."""
    start   = time.perf_counter()
    results = [run_agent(t) for t in tasks]
    total   = time.perf_counter() - start
    return results, total


# ── Parallel execution ─────────────────────────────────────────────────────────

def run_parallel(tasks: list[AgentTask], max_workers: int = 4) -> tuple[list[AgentTask], float]:
    """Run all agents concurrently with a thread pool."""
    start   = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_agent, t): t for t in tasks}
        for fut in as_completed(futures):
            results.append(fut.result())
    total = time.perf_counter() - start
    return results, total


# ── Fan-out / Fan-in pattern ───────────────────────────────────────────────────

class FanOutFanIn:
    """
    Fan-out: distribute one query to N specialist agents in parallel.
    Fan-in:  collect all results and merge into one answer.

    Classic pattern for:
      - Searching multiple knowledge bases simultaneously
      - Getting multiple LLM perspectives and voting
      - Parallel tool calls (like OpenAI parallel tool calling)
    """

    def __init__(self, agents: list[Callable[[str], str]]):
        self.agents = agents

    def run(self, query: str) -> dict[str, str]:
        """Fan-out to all agents, fan-in results."""
        results: dict[str, str] = {}
        lock = threading.Lock()

        def _call(fn: Callable) -> None:
            res = fn(query)
            with lock:
                results[fn.__name__] = res

        threads = [threading.Thread(target=_call, args=(fn,)) for fn in self.agents]
        start   = time.perf_counter()
        for t in threads: t.start()
        for t in threads: t.join()
        elapsed = time.perf_counter() - start

        print(f"  Fan-out: {len(self.agents)} agents, wall time={elapsed:.2f}s")
        return results


# ── Parallel with dependencies ─────────────────────────────────────────────────

@dataclass
class DAGTask:
    name:     str
    fn:       Callable
    depends:  list[str] = field(default_factory=list)   # names of tasks this waits for
    result:   Any       = None
    done:     bool      = False


def run_dag(tasks: list[DAGTask]) -> dict[str, Any]:
    """
    Execute tasks respecting dependencies.
    Independent tasks run in parallel; dependent tasks wait.

    Example:
      A ──┐
      B ──┴─── C ──── D
    A and B run in parallel; C waits for both; D waits for C.
    """
    by_name: dict[str, DAGTask] = {t.name: t for t in tasks}
    results: dict[str, Any]     = {}
    lock   = threading.Lock()
    done_event: dict[str, threading.Event] = {t.name: threading.Event() for t in tasks}

    def _run(task: DAGTask) -> None:
        # wait for all dependencies
        for dep in task.depends:
            done_event[dep].wait()

        dep_results = {d: by_name[d].result for d in task.depends}
        task.result = task.fn(dep_results)
        with lock:
            results[task.name] = task.result
        done_event[task.name].set()
        print(f"  [DAG] '{task.name}' completed (deps={task.depends or 'none'})")

    threads = [threading.Thread(target=_run, args=(t,)) for t in tasks]
    start   = time.perf_counter()
    for th in threads: th.start()
    for th in threads: th.join()
    elapsed = time.perf_counter() - start
    print(f"  DAG total wall time: {elapsed:.2f}s")
    return results


# ── Demo helpers ──────────────────────────────────────────────────────────────

def make_tasks(latencies: list[tuple[str, float]]) -> list[AgentTask]:
    return [AgentTask(name=n, query=f"answer for {n}", latency=lat)
            for n, lat in latencies]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Parallel Agents — Concurrent Execution Patterns")
    print("=" * 65)

    # ── 1. Sequential vs Parallel speedup ────────────────────────────────────
    print("\n── 1. Sequential vs Parallel (simulated latency) ──")
    latencies = [("SearchAgent", 0.5), ("CalcAgent", 0.4), ("WeatherAgent", 0.6), ("NewsAgent", 0.3)]
    expected_seq = sum(lat for _, lat in latencies)

    print(f"\nSequential (expected ~{expected_seq:.1f}s):")
    tasks_seq, seq_time = run_sequential(make_tasks(latencies))
    print(f"  Wall time: {seq_time:.2f}s")

    print(f"\nParallel (expected ~{max(lat for _, lat in latencies):.1f}s):")
    tasks_par, par_time = run_parallel(make_tasks(latencies))
    print(f"  Wall time: {par_time:.2f}s")
    print(f"  Speedup: {seq_time / par_time:.1f}×")

    # ── 2. Fan-out / Fan-in ───────────────────────────────────────────────────
    print("\n── 2. Fan-out / Fan-in ──")

    def search_pricing(query: str)  -> str: time.sleep(0.3); return "Starter $99, Pro $499"
    def search_security(query: str) -> str: time.sleep(0.4); return "SOC 2, AES-256"
    def search_sla(query: str)      -> str: time.sleep(0.2); return "99.9% uptime Pro"

    fanout  = FanOutFanIn([search_pricing, search_security, search_sla])
    answers = fanout.run("Tell me about TechCorp")
    print(f"  Merged answers:")
    for name, ans in answers.items():
        print(f"    {name}: {ans}")

    # ── 3. DAG with dependencies ──────────────────────────────────────────────
    print("\n── 3. DAG (tasks with dependencies) ──")

    dag_tasks = [
        DAGTask("fetch_pricing",  fn=lambda _: (time.sleep(0.3), "Pro $499")[1],    depends=[]),
        DAGTask("fetch_trial",    fn=lambda _: (time.sleep(0.3), "14-day free")[1], depends=[]),
        DAGTask("compute_annual", fn=lambda d: f"Annual: {d['fetch_pricing']} × 12 = $5,988", depends=["fetch_pricing"]),
        DAGTask("write_summary",  fn=lambda d: f"Summary: {d['fetch_pricing']} | {d['fetch_trial']} | {d['compute_annual']}", depends=["fetch_pricing","fetch_trial","compute_annual"]),
    ]

    results = run_dag(dag_tasks)
    print(f"\n  Final summary: {results['write_summary']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Parallel agents: speedup = sequential_time / max_agent_time")
    print("  • Only parallelize INDEPENDENT agents — no data dependency")
    print("  • ThreadPoolExecutor for I/O (LLM calls, DB, API)")
    print("  • DAG lets you mix parallel + dependent steps")
    print("  • Fan-out/fan-in = search many sources, merge answers")
    print("  • Always set max_workers to avoid overwhelming downstream APIs")


if __name__ == "__main__":
    main()
