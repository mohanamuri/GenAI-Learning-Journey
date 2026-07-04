"""
Core chatbot logic — system prompt, sliding window, streaming, cost tracking.
"""

from openai import OpenAI
import tiktoken

SYSTEM_PROMPT = """You are a helpful AI engineering tutor.
- Answer questions about AI, ML, Python, and software engineering
- Be concise — 3-5 sentences max unless asked for more
- Use code examples when helpful
- If unsure, say so rather than guessing
- Don't use filler phrases like "Great question!" or "Certainly!"
"""

PRICES = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "llama3.2:3b":  {"input": 0.0,   "output": 0.0},   # local = free
}

MAX_HISTORY_TOKENS = 3000  # sliding window limit


class SmartChatbot:
    def __init__(self, use_local: bool = True, model_override: str | None = None):
        # Default: Ollama local LLM — free, no API key required.
        # Pass use_local=False to switch to OpenAI cloud (needs OPENAI_API_KEY).
        self.use_local = use_local

        if use_local:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",  # Ollama local server
                api_key="ollama",
            )
            self.model = model_override or "llama3.2:3b"
        else:
            self.client = OpenAI()
            self.model = model_override or "gpt-4o-mini"

        self.enc = tiktoken.get_encoding("cl100k_base")
        self.history: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _count_tokens(self, messages: list[dict]) -> int:
        return sum(len(self.enc.encode(m["content"])) for m in messages)

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

    def _trim_history(self):
        """Remove oldest message pairs until within token budget."""
        while self._count_tokens(self._build_messages()) > MAX_HISTORY_TOKENS:
            if len(self.history) <= 2:
                break
            self.history.pop(0)  # remove oldest user message
            if self.history:
                self.history.pop(0)  # remove its assistant reply

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        full_response = []

        with self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(),
            max_tokens=400,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
                    full_response.append(delta)

        reply = "".join(full_response)
        self.history.append({"role": "assistant", "content": reply})

        # Track tokens
        input_tokens  = self._count_tokens(self._build_messages()[:-1])
        output_tokens = len(self.enc.encode(reply))
        self.total_input_tokens  += input_tokens
        self.total_output_tokens += output_tokens

        return reply

    def welcome_message(self) -> str:
        mode = f"Local ({self.model})" if self.use_local else f"Cloud ({self.model})"
        return f"\n{'='*50}\n  AI Engineering Tutor  [{mode}]\n{'='*50}"

    def print_history(self):
        print(f"\n--- Context window ({self._count_tokens(self._build_messages())} tokens, {len(self.history)//2} turns) ---")
        for msg in self.history:
            role = "You" if msg["role"] == "user" else "Bot"
            print(f"  {role}: {msg['content'][:80]}...")
        print()

    def print_cost_summary(self):
        price = PRICES.get(self.model, {"input": 0, "output": 0})
        cost = (self.total_input_tokens / 1_000_000 * price["input"]) + \
               (self.total_output_tokens / 1_000_000 * price["output"])

        print(f"\n--- Session Summary ---")
        print(f"  Model          : {self.model}")
        print(f"  Input tokens   : {self.total_input_tokens:,}")
        print(f"  Output tokens  : {self.total_output_tokens:,}")
        print(f"  Total tokens   : {self.total_input_tokens + self.total_output_tokens:,}")
        if self.use_local:
            print(f"  Cost           : $0.00 (local)")
        else:
            print(f"  Approx cost    : ${cost:.5f}")
