# Author: Mohan Raju Amuri
"""
07_multi_tool_agent.py — Agent with multiple tools and dynamic tool selection

What to remember:
- Multi-tool agent = agent that picks the right tool from a set based on the query
- The LLM reads tool descriptions and selects the most appropriate one
- Tool descriptions quality > number of tools — clear descriptions beat vague ones
- Tool chaining: result of tool A feeds into tool B in the next step

What NOT to do:
- Don't give the agent 20+ tools — accuracy degrades fast above 10 tools
- Don't use overlapping tool descriptions — the LLM gets confused and picks wrong tools
- Don't skip tool output validation — always check if the tool returned useful data

Multi-tool patterns:
  Sequential: Q → tool1 → result → tool2 → result → answer
  Parallel:   Q → tool1 + tool2 simultaneously → merge results → answer
  Conditional: Q → tool1 → if result.condition → tool2 else → tool3

Interview one-liner:
  "Multi-tool agents route to the right tool via description matching — quality of descriptions determines accuracy."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# ollama pull llama3.2:3b && ollama serve
# ============================================================

import json
import math
import re
from datetime import datetime, timedelta

LLM_MODEL = "llama3.2:3b"


# ── A rich set of tools ───────────────────────────────────────────────────────
class ToolRegistry:
    """Central registry for all available tools."""
    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, fn, description: str, params: dict):
        self._tools[name] = {"fn": fn, "description": description, "params": params}

    def call(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            return self._tools[name]["fn"](**kwargs)
        except Exception as e:
            return f"Error calling {name}: {e}"

    def as_openai_tools(self) -> list[dict]:
        result = []
        for name, info in self._tools.items():
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": info["params"],
                        "required": list(info["params"].keys()),
                    },
                },
            })
        return result

    @property
    def names(self) -> list:
        return list(self._tools.keys())


# ── Register tools ────────────────────────────────────────────────────────────
registry = ToolRegistry()

registry.register(
    "calculator",
    lambda expression: str(round(eval(expression, {"__builtins__": {}},
        {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}), 4)),
    "Evaluate math: arithmetic, percentages, powers, square roots.",
    {"expression": {"type": "string", "description": "Math expression, e.g. 'sqrt(144) + 20%'"}},
)

registry.register(
    "get_weather",
    lambda city: {
        "london": "13°C, overcast, humidity 82%",
        "paris":  "16°C, partly cloudy",
        "tokyo":  "21°C, clear, UV-index 7",
        "dubai":  "38°C, sunny, humidity 55%",
    }.get(city.lower(), f"No data for {city}"),
    "Get current weather for a city. Use for any weather-related questions.",
    {"city": {"type": "string", "description": "City name"}},
)

registry.register(
    "search_knowledge_base",
    lambda query: {
        "pricing":   "Starter $99/mo | Professional $499/mo | Enterprise custom",
        "sla":       "99.9% uptime for Pro/Enterprise; 99.5% for Starter",
        "security":  "SOC 2 Type II, AES-256 at rest, TLS 1.3 in transit",
        "trial":     "14-day free trial, full Professional, no credit card",
        "models":    "BERT, GPT-2, T5, XGBoost, ResNet, YOLO, ARIMA, Prophet",
        "sdk":       "import techcorp; client = techcorp.Client(api_key=...)",
    }.get(next((k for k in ["pricing","sla","security","trial","models","sdk"]
                if k in query.lower()), ""), f"No KB result for '{query}'"),
    "Search the product knowledge base for product info, pricing, and features.",
    {"query": {"type": "string", "description": "Search query"}},
)

registry.register(
    "unit_converter",
    lambda value, from_unit, to_unit: str(round({
        ("c","f"): value * 9/5 + 32, ("f","c"): (value-32)*5/9,
        ("km","miles"): value*0.621, ("miles","km"): value*1.609,
        ("kg","lbs"): value*2.205, ("lbs","kg"): value*0.453,
        ("usd","eur"): value*0.93, ("eur","usd"): value*1.08,
    }.get((from_unit.lower()[:2], to_unit.lower()[:2]), None) or f"Cannot convert {from_unit} to {to_unit}",4)),
    "Convert between units: temperature (C/F), distance (km/miles), weight (kg/lbs), currency (USD/EUR).",
    {
        "value":     {"type": "number", "description": "Numeric value to convert"},
        "from_unit": {"type": "string", "description": "Source unit"},
        "to_unit":   {"type": "string", "description": "Target unit"},
    },
)

registry.register(
    "get_date_info",
    lambda date_str="today": {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "day":  datetime.now().strftime("%A"),
        "week": datetime.now().isocalendar()[1],
        "days_until_year_end": (datetime(datetime.now().year, 12, 31) - datetime.now()).days,
    } if date_str == "today" else {"error": "only 'today' supported in demo"},
    "Get date and time information: today's date, day of week, week number.",
    {"date_str": {"type": "string", "description": "Use 'today' for current date"}},
)

registry.register(
    "text_analyzer",
    lambda text: json.dumps({
        "word_count":  len(text.split()),
        "char_count":  len(text),
        "sentence_count": len(re.findall(r'[.!?]+', text)),
        "avg_word_length": round(sum(len(w) for w in text.split()) / max(len(text.split()), 1), 1),
    }),
    "Analyze text: count words, characters, sentences, average word length.",
    {"text": {"type": "string", "description": "Text to analyze"}},
)


# ── Multi-tool Agent ──────────────────────────────────────────────────────────
def init_llm():
    try:
        from openai import OpenAI
        c = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        c.models.list()
        return c
    except Exception:
        return None


def run_multi_tool_agent(query: str, llm, max_steps: int = 4) -> str:
    """Agent loop: may call multiple tools to answer a complex query."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools to give accurate answers. After getting tool results, synthesize a clear final answer."},
        {"role": "user",   "content": query},
    ]
    tools = registry.as_openai_tools()
    steps_taken = 0

    while steps_taken < max_steps:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        finish = resp.choices[0].finish_reason

        if finish == "stop" or not msg.tool_calls:
            return msg.content or "Done."

        # Execute tool calls
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = registry.call(tc.function.name, **args)
            print(f"    Tool: {tc.function.name}({args}) → {result[:80]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        steps_taken += 1

    return "Max steps reached."


# ── Manual simulation (when Ollama not running) ───────────────────────────────
def simulate_multi_tool(query: str) -> None:
    """Show how multiple tools would be chained."""
    q = query.lower()
    print(f"\n  Simulating: '{query}'")

    if "weather" in q and ("celsius" in q or "fahrenheit" in q or "convert" in q):
        city = re.search(r"in (\w+)", q)
        city = city.group(1) if city else "london"
        weather = registry.call("get_weather", city=city)
        print(f"    Tool 1: get_weather(city='{city}') → {weather}")
        temp = re.search(r"(\d+)°C", weather)
        if temp:
            converted = registry.call("unit_converter",
                value=float(temp.group(1)), from_unit="C", to_unit="F")
            print(f"    Tool 2: unit_converter({temp.group(1)}, C→F) → {converted}°F")
        print(f"    Answer: Weather in {city}: {weather}. In Fahrenheit: {converted}°F")

    elif "pricing" in q and "annual" in q:
        pricing = registry.call("search_knowledge_base", query="pricing")
        print(f"    Tool 1: search_kb('pricing') → {pricing}")
        annual = registry.call("calculator", expression="499 * 12")
        print(f"    Tool 2: calculator('499 * 12') → ${annual}")
        print(f"    Answer: {pricing}. Professional annual = ${annual}.")

    elif "analyze" in q or "word count" in q:
        sample = "TechCorp AI Platform enables data scientists to build models at scale."
        result = registry.call("text_analyzer", text=sample)
        print(f"    Tool: text_analyzer(text=...) → {result}")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Multi-Tool Agent")
    print("=" * 60)
    print(f"\nRegistered tools: {registry.names}")

    llm = init_llm()
    print(f"Ollama: {'connected' if llm else 'not running (showing simulation)'}")

    queries = [
        "What's the weather in London and what is it in Fahrenheit?",
        "What is TechCorp's Professional plan pricing and what does it cost annually?",
        "What is 15% of 4800 and convert $720 to EUR?",
        "Analyze this sentence: 'The quick brown fox jumps over the lazy dog'",
    ]

    for query in queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        if llm:
            answer = run_multi_tool_agent(query, llm)
            print(f"  Answer: {answer}")
        else:
            simulate_multi_tool(query)

    # Tool selection accuracy
    print("\n" + "=" * 60)
    print("Tool Description Quality — What Matters")
    print("=" * 60)
    print("""
  GOOD description (precise, no overlap):
    "Evaluate mathematical expressions: arithmetic, percentages, powers."
    "Get current weather conditions for a city by name."
    "Search the product knowledge base for pricing and features."

  BAD description (vague, overlapping):
    "Do calculations or look up information"   ← too vague
    "Answer questions about numbers"           ← overlaps with KB search
    "Get information"                          ← everything matches this

  Rule: each tool should have a unique, specific domain.
  If two tools' descriptions overlap → the model will randomly pick one.
    """)

    print("=" * 60)
    print("Key Takeaways:")
    print("  - Agent calls multiple tools in sequence to answer complex queries")
    print("  - Tool description quality is the #1 factor in correct tool selection")
    print("  - Cap at 8-10 tools — accuracy degrades with too many options")
    print("  - Parallel tool calls (multiple tool_calls in one response) save latency")
    print("  - Always validate tool output before passing to the next step")
    print("=" * 60)
