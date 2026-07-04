# Author: Mohan Raju Amuri
"""
10_production_multi_agent.py — Production-grade multi-agent system

Production concerns beyond basic agents:
  1. Circuit breaker   — stop calling a failing agent, fail fast
  2. Shared memory     — agents share state with TTL (time-to-live)
  3. Health checks     — per-agent readiness before dispatch
  4. Metrics           — success rate, avg latency, error count per agent
  5. Graceful degradation — fallback if one agent is unavailable

Used patterns:
  - Circuit breaker: CLOSED → OPEN (after N failures) → HALF_OPEN (probe) → CLOSED
  - Shared memory TTL: cache tool results, expire stale data
  - Health check: each agent exposes a health() → bool

Runs with simulated agents — no Ollama required.
"""

import time
import threading
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
# 1. Circuit Breaker
# ══════════════════════════════════════════════════════════════

class CircuitState(Enum):
    CLOSED    = auto()   # normal: calls go through
    OPEN      = auto()   # tripped: calls blocked, return fallback
    HALF_OPEN = auto()   # probing: one call allowed to test recovery


@dataclass
class CircuitBreaker:
    """
    Wraps a callable. Trips OPEN after failure_threshold consecutive failures.
    Auto-resets to HALF_OPEN after recovery_timeout seconds.
    """
    name:               str
    failure_threshold:  int   = 3
    recovery_timeout:   float = 5.0
    state:              CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures:          int   = field(default=0, init=False)
    _opened_at:         float = field(default=0.0, init=False)
    _lock:              threading.Lock = field(default_factory=threading.Lock, init=False)

    def call(self, fn: Callable, *args, fallback: Any = None, **kwargs) -> Any:
        with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self._opened_at >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    print(f"  [CB:{self.name}] OPEN → HALF_OPEN (probing)")
                else:
                    print(f"  [CB:{self.name}] OPEN — blocking call, returning fallback")
                    return fallback

        try:
            result = fn(*args, **kwargs)
            with self._lock:
                if self.state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
                    if self._failures > 0:
                        self._failures = 0
                    if self.state == CircuitState.HALF_OPEN:
                        self.state = CircuitState.CLOSED
                        print(f"  [CB:{self.name}] HALF_OPEN → CLOSED (recovered)")
            return result

        except Exception as e:
            with self._lock:
                self._failures += 1
                print(f"  [CB:{self.name}] failure {self._failures}/{self.failure_threshold}: {e}")
                if self._failures >= self.failure_threshold and self.state != CircuitState.OPEN:
                    self.state    = CircuitState.OPEN
                    self._opened_at = time.time()
                    print(f"  [CB:{self.name}] CLOSED → OPEN (threshold reached)")
            return fallback


# ══════════════════════════════════════════════════════════════
# 2. Shared Memory with TTL
# ══════════════════════════════════════════════════════════════

@dataclass
class _CacheEntry:
    value:      Any
    expires_at: float


class SharedMemory:
    """
    Thread-safe key-value store with per-entry TTL.
    Expired entries return None and are evicted on access.
    """

    def __init__(self):
        self._store: dict[str, _CacheEntry] = {}
        self._lock  = threading.Lock()

    def set(self, key: str, value: Any, ttl: float = 60.0) -> None:
        with self._lock:
            self._store[key] = _CacheEntry(value=value, expires_at=time.time() + ttl)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                return None
            return entry.value

    def evict_expired(self) -> int:
        now     = time.time()
        with self._lock:
            before = len(self._store)
            self._store = {k: v for k, v in self._store.items() if v.expires_at > now}
            return before - len(self._store)

    def stats(self) -> dict:
        with self._lock:
            now  = time.time()
            live = sum(1 for v in self._store.values() if v.expires_at > now)
            return {"total_keys": len(self._store), "live": live, "expired": len(self._store) - live}


# ══════════════════════════════════════════════════════════════
# 3. Agent Metrics
# ══════════════════════════════════════════════════════════════

@dataclass
class AgentMetrics:
    name:           str
    calls:          int   = 0
    successes:      int   = 0
    failures:       int   = 0
    total_latency:  float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def record(self, success: bool, latency: float) -> None:
        with self._lock:
            self.calls         += 1
            self.total_latency += latency
            if success: self.successes += 1
            else:       self.failures  += 1

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.calls if self.calls else 0.0

    def summary(self) -> str:
        return (f"{self.name}: calls={self.calls} success={self.success_rate:.0%} "
                f"avg={self.avg_latency*1000:.0f}ms err={self.failures}")


# ══════════════════════════════════════════════════════════════
# 4. Production Agent
# ══════════════════════════════════════════════════════════════

class ProductionAgent:
    """
    Agent wrapper with:
    - Circuit breaker protecting the inner function
    - Shared memory cache (skip re-calling for same input)
    - Metrics tracking
    - Health check endpoint
    - Graceful degradation (fallback response)
    """

    def __init__(self, name: str, fn: Callable[[str], str],
                 memory: SharedMemory, fallback: str = "Service temporarily unavailable.",
                 failure_rate: float = 0.0):
        self.name         = name
        self._fn          = fn
        self._memory      = memory
        self._fallback    = fallback
        self._failure_rate = failure_rate   # simulate flaky agents
        self._cb          = CircuitBreaker(name, failure_threshold=3, recovery_timeout=3.0)
        self._metrics     = AgentMetrics(name)
        self._healthy     = True

    def health(self) -> bool:
        """Health check — returns False if circuit is OPEN."""
        return self._cb.state != CircuitState.OPEN

    def _flaky(self, query: str) -> str:
        """Wraps inner fn with simulated failure."""
        if self._failure_rate > 0 and random.random() < self._failure_rate:
            raise RuntimeError(f"[{self.name}] simulated transient failure")
        return self._fn(query)

    def call(self, query: str, cache_ttl: float = 30.0) -> str:
        # 1. Cache hit?
        cache_key = f"{self.name}:{query}"
        cached    = self._memory.get(cache_key)
        if cached:
            print(f"  [{self.name}] cache hit for '{query[:30]}'")
            return cached

        # 2. Health check
        if not self.health():
            print(f"  [{self.name}] unhealthy — returning fallback")
            return self._fallback

        # 3. Call with circuit breaker
        start   = time.perf_counter()
        result  = self._cb.call(self._flaky, query, fallback=self._fallback)
        elapsed = time.perf_counter() - start
        success = result != self._fallback

        # 4. Record metrics
        self._metrics.record(success=success, latency=elapsed)

        # 5. Cache result if successful
        if success:
            self._memory.set(cache_key, result, ttl=cache_ttl)

        return result


# ══════════════════════════════════════════════════════════════
# 5. Multi-Agent System
# ══════════════════════════════════════════════════════════════

class ProductionMultiAgentSystem:
    """
    Orchestrates multiple ProductionAgents with:
    - Per-agent health checks before dispatch
    - Parallel execution where possible
    - Aggregated metrics dashboard
    """

    def __init__(self, agents: list[ProductionAgent]):
        self.agents  = {a.name: a for a in agents}
        self.memory  = SharedMemory()

    def run_query(self, agent_name: str, query: str) -> str:
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Unknown agent: '{agent_name}'"
        return agent.call(query)

    def run_all(self, query: str) -> dict[str, str]:
        """Fan-out: same query to all healthy agents."""
        results: dict[str, str] = {}
        lock     = threading.Lock()

        def _call(name: str, agent: ProductionAgent) -> None:
            r = agent.call(query)
            with lock:
                results[name] = r

        threads = [threading.Thread(target=_call, args=(n, a))
                   for n, a in self.agents.items() if a.health()]
        for t in threads: t.start()
        for t in threads: t.join()
        return results

    def dashboard(self) -> None:
        print("\n  ── Metrics Dashboard ──────────────────────────────────")
        for agent in self.agents.values():
            health = "✓ healthy" if agent.health() else "✗ circuit open"
            print(f"    {agent._metrics.summary()} | {health}")
        print(f"  Memory: {self.memory.stats()}")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Production Multi-Agent System")
    print("=" * 65)

    memory = SharedMemory()

    # Inner agent functions (simulated)
    def search_kb(q: str)   -> str: time.sleep(0.1); return f"KB: Pro=$499/mo, SLA=99.9%"
    def calculator(q: str)  -> str: time.sleep(0.05); return f"Calc: $5,988/year"
    def weather(q: str)     -> str: time.sleep(0.08); return f"Weather: 22°C sunny"
    def flaky_news(q: str)  -> str: time.sleep(0.1); return f"News: TechCorp raises $50M"

    agents = [
        ProductionAgent("SearchAgent",  search_kb,    memory, failure_rate=0.0),
        ProductionAgent("CalcAgent",    calculator,   memory, failure_rate=0.0),
        ProductionAgent("WeatherAgent", weather,      memory, failure_rate=0.0),
        ProductionAgent("NewsAgent",    flaky_news,   memory, failure_rate=0.7),  # 70% failure rate
    ]
    system = ProductionMultiAgentSystem(agents)

    # ── 1. Normal queries ─────────────────────────────────────────────────────
    print("\n── 1. Normal agent calls ──")
    print(f"  SearchAgent:  {system.run_query('SearchAgent', 'pricing')}")
    print(f"  CalcAgent:    {system.run_query('CalcAgent', 'annual cost')}")
    print(f"  WeatherAgent: {system.run_query('WeatherAgent', 'London')}")

    # ── 2. Cache hit ──────────────────────────────────────────────────────────
    print("\n── 2. Cache hit (same query) ──")
    print(f"  SearchAgent (cached): {system.run_query('SearchAgent', 'pricing')}")

    # ── 3. Flaky agent → circuit breaker ─────────────────────────────────────
    print("\n── 3. Flaky NewsAgent → circuit breaker ──")
    for i in range(6):
        result = system.run_query("NewsAgent", f"query {i}")
        print(f"  Call {i+1}: '{result[:50]}'")

    # ── 4. Fan-out to all healthy agents ─────────────────────────────────────
    print("\n── 4. Fan-out: run healthy agents in parallel ──")
    results = system.run_all("What can you tell me?")
    for name, r in results.items():
        print(f"  {name}: {r}")
    skipped = set(system.agents) - set(results)
    if skipped:
        print(f"  Skipped (unhealthy): {skipped}")

    # ── 5. Memory stats ───────────────────────────────────────────────────────
    print("\n── 5. Evict expired cache entries ──")
    evicted = memory.evict_expired()
    print(f"  Evicted: {evicted} expired entries")

    # ── Dashboard ─────────────────────────────────────────────────────────────
    system.dashboard()

    print("\n── Key takeaways ──────────────────────────────────────────")
    print("  • Circuit breaker: fail fast, don't hammer a broken service")
    print("  • CLOSED → OPEN (N failures) → HALF_OPEN (probe) → CLOSED")
    print("  • Shared memory + TTL: skip duplicate LLM calls for same query")
    print("  • Health check: skip unhealthy agents before dispatching")
    print("  • Metrics: track success rate + latency per agent for alerting")
    print("  • Graceful degradation: fallback > crash")


if __name__ == "__main__":
    main()
