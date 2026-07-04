# Author: Mohan Raju Amuri
"""
08_agent_with_rag.py — Combine agent reasoning with RAG retrieval

What to remember:
- RAG is just a tool for an agent — the agent decides WHEN to retrieve
- The agent can retrieve AND use other tools in the same reasoning loop
- This solves: the agent calls RAG for factual KB queries, calculator for math, web for current events
- Hybrid: agent selects tools dynamically — RAG is one option among many

What NOT to do:
- Don't force every query through RAG — only use retrieval when the question needs KB knowledge
- Don't build a separate RAG pipeline AND an agent pipeline — unify them: RAG-as-tool
- Don't skip the "no results" handling — agent must fall back gracefully when retrieval fails

Pattern: Agentic RAG
  User asks → Agent thinks → "does this need KB?" → if yes: retrieve → generate grounded answer
                           → "does this need math?" → if yes: calculate
                           → "does this need current info?" → if yes: web search

Interview one-liner:
  "Agentic RAG treats retrieval as one tool among many — the agent decides when to search the KB."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# ollama pull llama3.2:3b && ollama serve
# Also needs: pip install chromadb sentence-transformers
# ============================================================

import json
import math
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path

LLM_MODEL  = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

# ── Build Knowledge Base ───────────────────────────────────────────────────────
KB_CHUNKS = [
    ("Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",   {"section": "pricing"}),
    ("Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",   {"section": "pricing"}),
    ("Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",    {"section": "pricing"}),
    ("SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",      {"section": "security"}),
    ("99.9% uptime SLA for Professional and Enterprise plans. Starter SLA is 99.5%.", {"section": "sla"}),
    ("14-day free trial with full Professional features, no credit card required.",   {"section": "faq"}),
    ("Enterprise support: 24/7 phone + Slack, dedicated support engineer.",           {"section": "support"}),
    ("Starter support: community forum + email, 48h response time.",                  {"section": "support"}),
    ("NLP models: BERT, GPT-2, T5, custom transformer architectures.",                {"section": "models"}),
    ("CV models: ResNet, EfficientNet, YOLO for image classification and detection.", {"section": "models"}),
    ("Python SDK: client = techcorp.Client(api_key='...'); model.predict(data)",      {"section": "sdk"}),
    ("Deployment: REST API, batch pipeline, edge ONNX, Kafka streaming.",             {"section": "deployment"}),
]

def build_kb() -> chromadb.Collection:
    client = chromadb.EphemeralClient()
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = client.create_collection("agent_kb", embedding_function=ef,
                                    metadata={"hnsw:space": "cosine"})
    col.add(
        documents=[t for t, _ in KB_CHUNKS],
        metadatas=[m for _, m in KB_CHUNKS],
        ids=[f"doc_{i}" for i in range(len(KB_CHUNKS))],
    )
    return col


# ── Tools ─────────────────────────────────────────────────────────────────────
def make_rag_tool(collection: chromadb.Collection):
    """Factory: creates a retrieval tool bound to a specific collection."""
    def retrieve_from_kb(query: str) -> str:
        results = collection.query(query_texts=[query], n_results=3)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        if not docs:
            return "No relevant information found in the knowledge base."
        parts = []
        for doc, meta, dist in zip(docs, metas, dists):
            sim = 1.0 - dist / 2.0
            if sim >= 0.3:
                parts.append(f"[{meta['section']}] {doc}")
        return "\n".join(parts) if parts else "No sufficiently relevant results found."
    return retrieve_from_kb


def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        return str(round(eval(expression, {"__builtins__": {}}, allowed), 4))
    except Exception as e:
        return f"Error: {e}"


def general_knowledge(question: str) -> str:
    """Mock general knowledge tool for non-KB questions."""
    answers = {
        "python": "Python is a high-level interpreted programming language created in 1991.",
        "llm": "LLMs (Large Language Models) are transformer-based models trained on large text corpora.",
        "transformer": "Transformer is a neural network architecture using self-attention, from the 'Attention Is All You Need' paper (2017).",
    }
    for key, ans in answers.items():
        if key in question.lower():
            return ans
    return f"I don't have specific knowledge about: {question}"


# ── OpenAI Tool Definitions ───────────────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_from_kb",
            "description": "Search the TechCorp product knowledge base for product info, pricing, features, SLA, support, and SDK details.",
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string", "description": "Search query"}},
                           "required": ["query"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Use for pricing calculations, percentages, and unit math.",
            "parameters": {"type": "object",
                           "properties": {"expression": {"type": "string", "description": "Math expression"}},
                           "required": ["expression"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_knowledge",
            "description": "Answer general knowledge questions not in the product KB (e.g. programming concepts, technology definitions).",
            "parameters": {"type": "object",
                           "properties": {"question": {"type": "string", "description": "Question to answer"}},
                           "required": ["question"]},
        },
    },
]


# ── Agentic RAG ───────────────────────────────────────────────────────────────
class AgenticRAG:
    def __init__(self, llm=None):
        print("  Building knowledge base...")
        self.collection = build_kb()
        self.retrieve_fn = make_rag_tool(self.collection)
        self.tools = {
            "retrieve_from_kb": self.retrieve_fn,
            "calculator": calculator,
            "general_knowledge": general_knowledge,
        }
        self.llm = llm

    def ask(self, query: str, max_steps: int = 4) -> str:
        if not self.llm:
            return None  # fallback to simulation

        messages = [
            {"role": "system",
             "content": ("You are a TechCorp product assistant. "
                         "Use retrieve_from_kb for product questions. "
                         "Use calculator for math. "
                         "Use general_knowledge for tech concepts. "
                         "Always cite where you got the information.")},
            {"role": "user", "content": query},
        ]

        for _ in range(max_steps):
            resp = self.llm.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if resp.choices[0].finish_reason == "stop" or not msg.tool_calls:
                return msg.content

            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self.tools[tc.function.name](**args)
                print(f"    [{tc.function.name}] {args} → {result[:80]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        return "Max steps reached."

    def simulate(self, query: str) -> None:
        """Demonstrate the routing logic without Ollama."""
        q = query.lower()
        print(f"\n  Query: '{query}'")

        if any(w in q for w in ["price", "cost", "plan", "trial", "support", "security", "sla", "deploy"]):
            result = self.retrieve_fn(query)
            print(f"  → Routed to: retrieve_from_kb")
            print(f"  KB result:\n{result}")
            if "499" in result and "annual" in q:
                calc = calculator("499 * 12")
                print(f"  → Also called: calculator('499 * 12') → {calc}")

        elif any(c in query for c in ["+", "-", "*", "/", "%"]) or "calculate" in q:
            expr = re.search(r"[\d\s\+\-\*/\.\(\)]+", query)
            if expr:
                result = calculator(expr.group(0).strip())
                print(f"  → Routed to: calculator → {result}")

        else:
            result = general_knowledge(query)
            print(f"  → Routed to: general_knowledge → {result}")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import re
    print("=" * 60)
    print("AI Agents — Agent with RAG")
    print("=" * 60)

    llm = None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        llm = client
        print(f"\nOllama: connected ({OLLAMA_MODEL})")
    except Exception:
        print("\nOllama: not running (showing routing simulation)")

    print("\nBuilding agentic RAG system...")
    agent = AgenticRAG(llm=llm)

    queries = [
        "What does the Professional plan cost annually?",      # KB + calculator
        "Is TechCorp's platform secure and SOC 2 certified?",  # KB
        "What is a transformer model?",                        # general knowledge
        "What NLP models are supported and what is BERT?",     # KB + general knowledge
        "If I have 10 users, which plan should I choose?",     # KB + reasoning
    ]

    for q in queries:
        print(f"\n{'─'*60}")
        if llm:
            answer = agent.ask(q)
            print(f"Q: {q}\nA: {answer}")
        else:
            agent.simulate(q)

    print("\n" + "=" * 60)
    print("Agentic RAG vs Standard RAG")
    print("=" * 60)
    print("""
  Standard RAG:
    Every query → retrieve → generate
    Problem: retrieves even when the LLM already knows the answer
             retrieves even for math/code questions (useless)

  Agentic RAG:
    Every query → agent decides → retrieve? calculate? general knowledge?
    Benefit: retrieves ONLY when KB knowledge is needed
             chains tools for complex queries (retrieve + calculate)
             can escalate to web search if KB has no answer
    """)

    print("=" * 60)
    print("Key Takeaways:")
    print("  - RAG is just a tool — let the agent decide when to use it")
    print("  - Multi-tool agents combine RAG + calculator + search in one loop")
    print("  - Tool descriptions determine routing accuracy")
    print("  - Agentic RAG reduces unnecessary retrieval calls by 40-60%")
    print("=" * 60)
