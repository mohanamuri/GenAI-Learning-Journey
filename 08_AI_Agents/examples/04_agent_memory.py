# Author: Mohan Raju Amuri
"""
04_agent_memory.py — Short-term and long-term memory for agents

What to remember:
- Agents need memory to handle multi-turn tasks and remember past actions
- Short-term memory = conversation history in the current session (chat messages list)
- Long-term memory = persisted storage (vector DB, key-value store) across sessions
- Memory management matters: LLMs have context window limits — don't blindly grow history

Memory types:
  Sensory/buffer  — last N messages (most recent context)
  Summary memory  — LLM summarizes old turns to compress, keeps recent ones verbatim
  Entity memory   — extract and track key entities (people, places, facts)
  Vector memory   — embed past interactions, retrieve by semantic similarity

What NOT to do:
- Don't append unlimited history — context window overflows → truncation → lost context
- Don't store everything in long-term memory — add only important facts
- Don't use raw conversation history for long-term storage — summarize it first

Interview one-liner:
  "Agent memory = short-term (chat history) + long-term (vector/KV store) — manage both or context breaks."
"""

import json
from datetime import datetime
from collections import deque

# ── 1. Buffer Memory (last N messages) ───────────────────────────────────────
class BufferMemory:
    """
    Keep only the last `k` conversation turns.
    Oldest turns are dropped when k is exceeded.
    Cheapest memory — use as the baseline.
    """
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        # Each turn = (user_message, assistant_response)
        self._buffer: deque = deque(maxlen=max_turns)
        self.system_prompt = "You are a helpful assistant."

    def add(self, user: str, assistant: str) -> None:
        self._buffer.append({"user": user, "assistant": assistant})

    def get_messages(self, current_user_msg: str) -> list[dict]:
        """Build the messages list for the LLM API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self._buffer:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": current_user_msg})
        return messages

    def __len__(self):
        return len(self._buffer)

    def summary(self) -> str:
        return f"BufferMemory: {len(self._buffer)}/{self.max_turns} turns stored"


# ── 2. Summary Memory (compress old turns, keep recent verbatim) ───────────
class SummaryMemory:
    """
    LLM summarizes conversation history when it gets too long.
    Keeps a rolling summary + last k verbatim turns.
    Good for long conversations where early context matters but exact wording doesn't.
    """
    def __init__(self, max_verbatim_turns: int = 3, llm=None):
        self.max_verbatim = max_verbatim_turns
        self.summary = ""          # compressed summary of older turns
        self.recent: list = []     # last N turns verbatim
        self.llm = llm

    def add(self, user: str, assistant: str) -> None:
        self.recent.append({"user": user, "assistant": assistant})
        if len(self.recent) > self.max_verbatim:
            # Compress the oldest turn into the summary
            oldest = self.recent.pop(0)
            new_info = f"User asked: '{oldest['user']}'. Assistant said: '{oldest['assistant']}'."
            self.summary = (self.summary + " " + new_info).strip() if self.summary else new_info

    def get_messages(self, current_user_msg: str) -> list[dict]:
        system = "You are a helpful assistant."
        if self.summary:
            system += f"\n\nConversation summary so far:\n{self.summary}"
        messages = [{"role": "system", "content": system}]
        for turn in self.recent:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": current_user_msg})
        return messages

    def summary_stats(self) -> dict:
        return {
            "summary_chars": len(self.summary),
            "verbatim_turns": len(self.recent),
            "max_verbatim": self.max_verbatim,
        }


# ── 3. Entity Memory (track key facts about people/things) ─────────────────
class EntityMemory:
    """
    Extract and store facts about named entities mentioned in conversation.
    Example: "My name is Alice" → entities["Alice"]["name"] = "Alice"
    In production: use an LLM to extract entities. Here: simple keyword patterns.
    """
    def __init__(self):
        self.entities: dict[str, dict] = {}

    def extract_and_store(self, user_msg: str) -> None:
        """Extract simple facts from user message (rule-based for demo)."""
        msg = user_msg.lower()

        # Name detection
        import re
        name_match = re.search(r"my name is (\w+)", msg)
        if name_match:
            name = name_match.group(1).title()
            self.entities.setdefault("user", {})["name"] = name

        # Preference detection
        like_match = re.search(r"i (?:like|love|prefer) (\w+(?:\s\w+)?)", msg)
        if like_match:
            pref = like_match.group(1)
            self.entities.setdefault("user", {}).setdefault("preferences", []).append(pref)

        # Location
        loc_match = re.search(r"i(?:'m| am) (?:in|from) (\w+(?:\s\w+)?)", msg)
        if loc_match:
            self.entities.setdefault("user", {})["location"] = loc_match.group(1).title()

    def get_context(self) -> str:
        """Return entity facts as a context string for the LLM."""
        if not self.entities:
            return ""
        parts = []
        for entity, facts in self.entities.items():
            for key, val in facts.items():
                if isinstance(val, list):
                    parts.append(f"{entity} {key}: {', '.join(val)}")
                else:
                    parts.append(f"{entity} {key}: {val}")
        return "Known facts:\n" + "\n".join(f"  - {p}" for p in parts)

    def get_messages(self, current_user_msg: str, history: list) -> list[dict]:
        entity_ctx = self.get_context()
        system = "You are a helpful assistant."
        if entity_ctx:
            system += f"\n\n{entity_ctx}"
        messages = [{"role": "system", "content": system}]
        for turn in history[-4:]:  # last 4 turns
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": current_user_msg})
        return messages


# ── 4. Long-term Vector Memory ────────────────────────────────────────────────
class VectorMemory:
    """
    Store important facts as embeddings — retrieve by semantic similarity.
    Use for: remembering past tasks, previous answers, user preferences across sessions.
    """
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.np = np
            self.memories: list[dict] = []     # {text, vec, timestamp, importance}
            self.available = True
        except ImportError:
            self.available = False

    def remember(self, text: str, importance: int = 1) -> None:
        """Store a memory with its embedding."""
        if not self.available:
            return
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        self.memories.append({
            "text": text,
            "vec": vec,
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
        })

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve top-k most relevant memories for a query."""
        if not self.available or not self.memories:
            return []
        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        scored = [
            (float(self.np.dot(q_vec, m["vec"])), m["text"])
            for m in self.memories
        ]
        scored.sort(reverse=True)
        return [text for _, text in scored[:top_k]]

    def stats(self) -> str:
        return f"VectorMemory: {len(self.memories)} memories stored"


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Memory Systems")
    print("=" * 60)

    # 1. Buffer Memory
    print("\n[1] Buffer Memory — keeps last N turns")
    buf = BufferMemory(max_turns=3)
    conversation = [
        ("What is Python?",        "Python is a high-level programming language."),
        ("Who created it?",        "Guido van Rossum created Python in 1991."),
        ("What is it used for?",   "Python is used for web dev, data science, AI, and automation."),
        ("Is it fast?",            "Python is slower than C but fast enough for most tasks."),
        ("What about type hints?", "Python supports optional type hints via the typing module."),
    ]
    for user, assistant in conversation:
        buf.add(user, assistant)
        print(f"  After adding turn: {buf.summary()}")

    print(f"\n  Messages sent to LLM for 'Tell me more':")
    msgs = buf.get_messages("Tell me more about type hints")
    for m in msgs:
        print(f"    [{m['role']}] {m['content'][:60]}")
    print("  → Only last 3 turns kept — oldest 2 are gone (buffer overflow)")

    # 2. Summary Memory
    print("\n[2] Summary Memory — compress old, keep recent verbatim")
    summ = SummaryMemory(max_verbatim_turns=2)
    for user, assistant in conversation:
        summ.add(user, assistant)

    print(f"  Stats: {summ.summary_stats()}")
    print(f"  Compressed summary: {summ.summary[:120]}...")
    print(f"  Recent verbatim turns: {len(summ.recent)}")
    msgs = summ.get_messages("What was the first thing we discussed?")
    print(f"\n  System message (includes summary):")
    print(f"    {msgs[0]['content'][:200]}")

    # 3. Entity Memory
    print("\n[3] Entity Memory — track facts about entities")
    entity_mem = EntityMemory()
    history = []
    user_inputs = [
        "Hi, my name is Alice",
        "I'm from London",
        "I like machine learning and Python",
        "What topics should I focus on?",
    ]
    mock_responses = [
        "Nice to meet you, Alice!",
        "London is a great city!",
        "Great choices — both are in high demand.",
        "Focus on PyTorch, transformers, and RAG pipelines.",
    ]
    for user, resp in zip(user_inputs, mock_responses):
        entity_mem.extract_and_store(user)
        history.append({"user": user, "assistant": resp})

    print(f"  Extracted entities: {json.dumps(entity_mem.entities, indent=4)}")
    msgs = entity_mem.get_messages("Recommend a learning path for me", history)
    print(f"\n  System message with entity context:")
    print(f"    {msgs[0]['content']}")

    # 4. Vector Memory
    print("\n[4] Vector Memory — semantic retrieval of past facts")
    vmem = VectorMemory()
    if vmem.available:
        facts = [
            "User prefers Python over JavaScript",
            "User has 3 years of ML experience",
            "User's company uses AWS for deployment",
            "User wants to learn about transformer architectures",
            "User's team is 5 engineers, working on NLP products",
            "User struggles with CUDA out-of-memory errors",
        ]
        for fact in facts:
            vmem.remember(fact, importance=1)
        print(f"  {vmem.stats()}")
        queries = [
            "What cloud platform does the user use?",
            "What is the user's experience level?",
            "What technical problems has the user faced?",
        ]
        for q in queries:
            recalled = vmem.recall(q, top_k=2)
            print(f"\n  Query: '{q}'")
            for r in recalled:
                print(f"    → {r}")
    else:
        print("  (sentence-transformers not installed — skipping vector memory demo)")

    # Memory comparison
    print("\n" + "=" * 60)
    print("Memory Types — When to Use Each")
    print("=" * 60)
    table = [
        ("Buffer",  "Last N turns verbatim",              "Short chats, chatbots"),
        ("Summary", "Compressed history + recent turns",  "Long conversations, assistants"),
        ("Entity",  "Extracted facts about people/things","Personal assistants, CRM bots"),
        ("Vector",  "Semantic retrieval of past memories","Cross-session memory, long-term agents"),
    ]
    print(f"  {'Type':<10}  {'What it stores':<40}  {'Best for'}")
    for t, s, u in table:
        print(f"  {t:<10}  {s:<40}  {u}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Always cap buffer size — unbounded history fills the context window")
    print("  - Summary memory is the best default for most agents")
    print("  - Entity memory personalizes responses without storing full history")
    print("  - Vector memory enables cross-session recall — pairs with ChromaDB")
    print("=" * 60)
