# Author: Mohan Raju Amuri
"""
Chat History (Multi-Turn Conversations)
-----------------------------------------
One-liner: LLMs are stateless — you resend the full conversation history every call.

Remember:
- There is NO server-side memory — you maintain and send the full messages list
- Each turn: append user message → call API → append assistant response → repeat
- History grows with every turn → tokens increase → cost increases
- Strategies to manage: sliding window, summarization, trim old messages
- System prompt counts tokens every call — keep it short

Token cost per call = system_prompt + ALL previous messages + new user message + response
  (not just the new message!)

Don't:
- Don't assume the LLM remembers previous conversations between sessions
- Don't let history grow unbounded — implement a trimming strategy
- Don't include irrelevant history — every token costs money (or slows local model)
- Don't store raw messages in DB without metadata (timestamp, session_id)

Running on Ollama (local, free — no API key needed):
  ollama pull llama3.2:3b
"""

from openai import OpenAI
import tiktoken

# ── Using Ollama (local LLM) — free, no API key required ──────────────────
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "llama3.2:3b"

# tiktoken approximates token counts — works for Ollama models too (similar tokenizers)
enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict]) -> int:
    return sum(len(enc.encode(m["content"])) for m in messages)


# ── 1. Basic multi-turn chat ──────────────────────────────────────────────
# Each call sends the full history — the model has no state between calls.
# Watch the token count grow with each turn.
print("=== 1. Basic Multi-Turn Chat ===")

class SimpleChatSession:
    def __init__(self, system: str = ""):
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=MODEL,
            messages=self.messages,  # full history sent every time
            max_tokens=200,
        )
        reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    @property
    def token_count(self) -> int:
        return count_tokens(self.messages)


session = SimpleChatSession(system="You are a helpful Python tutor. Be concise.")
turns = [
    "What is a list comprehension?",
    "Can you show me an example that filters even numbers?",
    "Now modify it to return squares of those even numbers",
]

for user_msg in turns:
    reply = session.chat(user_msg)
    print(f"User    : {user_msg}")
    print(f"Bot     : {reply[:100]}...")
    # Token count grows each turn — this is why windowing matters
    print(f"Tokens  : {session.token_count}\n")


# ── 2. Sliding window (trim old messages) ────────────────────────────────
# Once history exceeds max_tokens, drop oldest message pairs.
# Trade-off: saves tokens at the cost of losing early context.
print("=== 2. Sliding Window ===")

class WindowedChatSession:
    def __init__(self, system: str = "", max_tokens: int = 1500):
        self.system = system
        self.max_tokens = max_tokens
        self.history: list[dict] = []

    def _build_messages(self) -> list[dict]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.history)
        return messages

    def _trim(self):
        # Remove oldest user+assistant pairs until within budget
        while count_tokens(self._build_messages()) > self.max_tokens:
            if len(self.history) <= 2:
                break
            self.history.pop(0)  # remove oldest user message
            if self.history:
                self.history.pop(0)  # remove its paired assistant response

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim()
        messages = self._build_messages()
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=150,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply


windowed = WindowedChatSession(system="Be concise.", max_tokens=500)
topics = [
    "Explain variables in Python",
    "Now explain loops",
    "And functions?",
    "What about classes?",
    "How do I use decorators?",
]
print(f"Max context: 500 tokens")
for topic in topics:
    reply = windowed.chat(topic)
    total = count_tokens(windowed._build_messages())
    kept = len(windowed.history) // 2
    print(f"  [{kept} turns kept, {total} tokens] {topic[:40]}")


# ── 3. History storage format for persistence ─────────────────────────────
# When building a real app, serialize the session to DB or file.
# Include metadata so you can reconstruct context across sessions.
print("\n=== 3. Persistence Format ===")
import json

session_data = {
    "session_id": "user123-session-456",
    "created_at": "2025-07-04T10:00:00",
    "model": MODEL,
    "messages": session.messages,
}
print("Serializable session (save to DB/file):")
print(json.dumps(session_data, indent=2)[:300] + "...")
