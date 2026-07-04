# Author: Mohan Raju Amuri
"""
personal_assistant/agent.py — Personal AI assistant with tools, memory, and guardrails

A production-style agent that acts as your personal assistant:
- Answers questions using a local knowledge base (RAG)
- Does math calculations
- Provides weather info
- Remembers your name and preferences across the session
- Handles errors gracefully

Usage:
    python agent.py                    # interactive mode
    python agent.py --task "..."       # one-shot task

Requirements:
    pip install chromadb sentence-transformers openai
    ollama pull llama3.2:3b && ollama serve
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# Install: https://ollama.ai
# Pull:    ollama pull llama3.2:3b
# Serve:   ollama serve
# ============================================================

import json
import math
import re
import logging
import argparse
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime
from collections import deque
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL   = "llama3.2:3b"
EMBED_MODEL    = "all-MiniLM-L6-v2"
MAX_STEPS      = 6
MAX_HISTORY    = 8   # conversation turns to keep
MIN_KB_SCORE   = 0.30

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", level=logging.WARNING)
logger = logging.getLogger("assistant")


# ── Knowledge Base ─────────────────────────────────────────────────────────────
KB_DOCS = [
    ("Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",   {"section": "pricing"}),
    ("Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",   {"section": "pricing"}),
    ("Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",    {"section": "pricing"}),
    ("SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",      {"section": "security"}),
    ("99.9% uptime SLA for Professional and Enterprise. Starter SLA is 99.5%.",       {"section": "sla"}),
    ("14-day free trial with full Professional features, no credit card required.",   {"section": "faq"}),
    ("Enterprise support: 24/7 phone + Slack, dedicated support engineer.",           {"section": "support"}),
    ("Starter support: community forum + email, 48h response time.",                  {"section": "support"}),
    ("NLP models: BERT, GPT-2, T5, custom transformer architectures.",                {"section": "models"}),
    ("CV models: ResNet, EfficientNet, YOLO for image classification and detection.", {"section": "models"}),
    ("Python SDK: client = techcorp.Client(api_key='...'); model.predict(data)",      {"section": "sdk"}),
]

def build_kb() -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.create_collection("assistant_kb", embedding_function=ef,
                                    metadata={"hnsw:space": "cosine"})
    col.add(documents=[d for d, _ in KB_DOCS],
            metadatas=[m for _, m in KB_DOCS],
            ids=[f"kb_{i}" for i in range(len(KB_DOCS))])
    return col


# ── Tools ─────────────────────────────────────────────────────────────────────
def make_kb_tool(collection):
    def search_kb(query: str) -> str:
        results = collection.query(query_texts=[query], n_results=3)
        hits = [(doc, 1 - dist/2) for doc, dist in
                zip(results["documents"][0], results["distances"][0])
                if 1 - dist/2 >= MIN_KB_SCORE]
        if not hits:
            return "No relevant information found in the knowledge base."
        return "\n".join(f"• {doc}" for doc, _ in hits)
    return search_kb

def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(round(result, 4))
    except Exception as e:
        return f"Calculation error: {e}"

def get_weather(city: str) -> str:
    data = {
        "london": "14°C, overcast, humidity 82%", "paris": "17°C, partly cloudy",
        "tokyo": "21°C, sunny",  "new york": "22°C, clear", "sydney": "26°C, warm",
        "dubai": "38°C, hot and sunny",
    }
    return data.get(city.lower(), f"Weather data not available for '{city}'")

def get_datetime(_: str = "") -> str:
    now = datetime.now()
    return f"{now.strftime('%A, %B %d %Y')} — {now.strftime('%H:%M')}"


# ── Tool Definitions (OpenAI format) ─────────────────────────────────────────
TOOL_DEFS = [
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search the TechCorp product knowledge base for pricing, features, SLA, security, support, and SDK info.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "Search query, e.g. 'pricing' or 'security'"}},
                       "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate mathematical expressions: arithmetic, percentages, powers, roots.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string", "description": "e.g. '499 * 12' or 'sqrt(144)'"}},
                       "required": ["expression"]},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]},
    }},
    {"type": "function", "function": {
        "name": "get_datetime",
        "description": "Get the current date and time.",
        "parameters": {"type": "object",
                       "properties": {"_": {"type": "string", "description": "Pass empty string"}},
                       "required": []},
    }},
]


# ── Entity Memory ──────────────────────────────────────────────────────────────
class EntityMemory:
    def __init__(self):
        self.facts: dict = {}

    def extract(self, text: str) -> None:
        t = text.lower()
        m = re.search(r"my name is (\w+)", t)
        if m: self.facts["name"] = m.group(1).title()
        m = re.search(r"i(?:'m| am) from (\w[\w\s]+)", t)
        if m: self.facts["location"] = m.group(1).strip().title()
        m = re.search(r"i (?:like|love|prefer) ([\w\s]+?)(?:\.|$)", t)
        if m: self.facts.setdefault("preferences", []).append(m.group(1).strip())

    def as_context(self) -> str:
        if not self.facts:
            return ""
        lines = []
        for k, v in self.facts.items():
            if isinstance(v, list):
                lines.append(f"- {k}: {', '.join(v)}")
            else:
                lines.append(f"- {k}: {v}")
        return "What I know about you:\n" + "\n".join(lines)


# ── Personal Assistant ────────────────────────────────────────────────────────
class PersonalAssistant:
    def __init__(self):
        print("Initializing personal assistant...")
        self.kb = build_kb()
        self.tools = {
            "search_kb":    make_kb_tool(self.kb),
            "calculator":   calculator,
            "get_weather":  get_weather,
            "get_datetime": get_datetime,
        }
        self.memory  = EntityMemory()
        self.history: deque = deque(maxlen=MAX_HISTORY)
        self.llm = self._init_llm()
        print(f"  Knowledge base: {len(KB_DOCS)} documents")
        print(f"  LLM: {OLLAMA_MODEL if self.llm else 'NOT CONNECTED (run ollama serve)'}")
        print(f"  Tools: {list(self.tools.keys())}")

    def _init_llm(self):
        try:
            from openai import OpenAI
            c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            c.models.list()
            return c
        except Exception:
            return None

    def _system_prompt(self) -> str:
        entity_ctx = self.memory.as_context()
        prompt = (
            "You are a helpful personal AI assistant. "
            "Use tools to give accurate, grounded answers. "
            "For TechCorp product questions, always use search_kb. "
            "For math, use calculator. "
            "Be concise and friendly."
        )
        if entity_ctx:
            prompt += f"\n\n{entity_ctx}"
        return prompt

    def chat(self, user_input: str) -> str:
        self.memory.extract(user_input)

        if self.llm is None:
            return "[Ollama not running. Install from https://ollama.ai and run `ollama pull llama3.2:3b`]"

        messages = [{"role": "system", "content": self._system_prompt()}]
        for turn in self.history:
            messages.append({"role": "user",      "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_input})

        for step in range(MAX_STEPS):
            try:
                resp = self.llm.chat.completions.create(
                    model=OLLAMA_MODEL, messages=messages,
                    tools=TOOL_DEFS, tool_choice="auto",
                    temperature=0.2, max_tokens=500,
                )
                msg = resp.choices[0].message
                if resp.choices[0].finish_reason == "stop" or not msg.tool_calls:
                    answer = msg.content or "I'm not sure how to help with that."
                    self.history.append({"user": user_input, "assistant": answer})
                    return answer

                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    try:
                        result = self.tools[tc.function.name](**args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.info(f"Tool: {tc.function.name}({args}) → {result[:60]}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            except Exception as e:
                logger.error(f"LLM error: {e}")
                return "I encountered an error. Please try again."

        return "I reached my step limit. Please try a simpler question."

    def run_interactive(self):
        print("\n" + "=" * 60)
        print("Personal AI Assistant")
        print("Type 'quit' to exit | 'clear' to reset memory | 'debug' to see memory")
        print("=" * 60)
        print("\nHello! I'm your personal AI assistant. How can I help you today?")

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                self.history.clear()
                self.memory = EntityMemory()
                print("Memory cleared.")
                continue
            if user_input.lower() == "debug":
                print(f"History turns: {len(self.history)}")
                print(f"Entity memory: {self.memory.facts}")
                continue

            response = self.chat(user_input)
            print(f"\nAssistant: {response}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Personal AI Assistant")
    parser.add_argument("--task", type=str, help="One-shot task")
    args = parser.parse_args()

    assistant = PersonalAssistant()

    if args.task:
        print(f"\nTask: {args.task}")
        response = assistant.chat(args.task)
        print(f"Response: {response}")
    else:
        assistant.run_interactive()


if __name__ == "__main__":
    main()
