# Author: Mohan Raju Amuri
"""
03_tool_calling.py — Define tools, let the LLM choose and execute them

What to remember:
- Tool calling = LLM returns structured JSON instead of prose, specifying tool name + args
- Much more reliable than ReAct text parsing — no regex needed, model outputs valid JSON
- OpenAI-compatible format: tools = [{type, function: {name, description, parameters}}]
- Ollama supports function calling with llama3.1:8b and newer models

What NOT to do:
- Don't use tool calling models for tools that require long free-text input (use ReAct instead)
- Don't define more than 5-10 tools per call — model accuracy drops with too many choices
- Don't skip the description field — it's what the model reads to decide which tool to call

Tool vs ReAct:
  ReAct:        free-text prompt, regex parsing, works with any LLM
  Tool calling: structured JSON output, no parsing, needs a capable model
  Prefer tool calling when available — much more robust

Interview one-liner:
  "Tool calling returns structured JSON from the LLM — no parsing needed, just call the function."
"""

# ============================================================
# 💻  RUNS LOCALLY — Ollama Required
# ============================================================
# Pull a model that supports function calling:
#   ollama pull llama3.1:8b      (best quality for tool use)
#   ollama pull llama3.2:3b      (also supports tools, lighter)
# Start: ollama serve
#
# Falls back to simulated structured output if Ollama not running.
# ============================================================

import json
import math
import re
from datetime import datetime, timedelta

LLM_MODEL = "llama3.2:3b"  # supports tool calling

# ── Tool Definitions (OpenAI tool format) ─────────────────────────────────────
# This is the schema the LLM reads to understand each tool.
# description + parameter descriptions are critical — they guide tool selection.

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Use for any arithmetic, percentages, square roots, or unit conversions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A valid Python math expression, e.g. '15 * 7 + 23' or 'sqrt(144)'"
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city. Always use this when asked about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city, e.g. 'London', 'Tokyo', 'New York'"
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_events",
            "description": "Get upcoming calendar events within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient. Use when the user explicitly asks to send or draft an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": "Convert between units: temperature (celsius/fahrenheit), length (km/miles), weight (kg/lbs).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value":     {"type": "number", "description": "The numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit (celsius, fahrenheit, km, miles, kg, lbs)"},
                    "to_unit":   {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


# ── Tool Implementations ──────────────────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {round(result, 6)}"
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    data = {
        "london":   "Cloudy 14°C, humidity 80%, light rain",
        "new york": "Sunny 22°C, humidity 45%, clear",
        "tokyo":    "Partly cloudy 19°C, humidity 65%",
        "sydney":   "Warm 26°C, sunny, UV index 8",
        "paris":    "Overcast 16°C, chance of drizzle",
    }
    return data.get(city.lower(), f"No weather data for '{city}'")


def get_calendar_events(start_date: str, end_date: str) -> str:
    events = [
        {"date": "2026-07-05", "title": "Team standup",       "time": "09:00"},
        {"date": "2026-07-06", "title": "Product review",     "time": "14:00"},
        {"date": "2026-07-08", "title": "Sprint planning",    "time": "10:00"},
        {"date": "2026-07-10", "title": "Customer demo",      "time": "16:00"},
    ]
    filtered = [e for e in events if start_date <= e["date"] <= end_date]
    if not filtered:
        return f"No events between {start_date} and {end_date}"
    return json.dumps(filtered, indent=2)


def send_email(to: str, subject: str, body: str) -> str:
    # Mock — in production would call an email API
    return f"[MOCK] Email sent to {to} | Subject: '{subject}' | Body: {len(body)} chars"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        ("km", "miles"):           lambda v: v * 0.621371,
        ("miles", "km"):           lambda v: v * 1.60934,
        ("kg", "lbs"):             lambda v: v * 2.20462,
        ("lbs", "kg"):             lambda v: v * 0.453592,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Unsupported conversion: {from_unit} → {to_unit}"
    result = conversions[key](value)
    return f"{value} {from_unit} = {round(result, 4)} {to_unit}"


TOOL_REGISTRY = {
    "calculator":         calculator,
    "get_weather":        get_weather,
    "get_calendar_events": get_calendar_events,
    "send_email":         send_email,
    "unit_converter":     unit_converter,
}


# ── Execute a tool call ────────────────────────────────────────────────────────
def execute_tool_call(tool_call) -> str:
    """Execute a tool call returned by the LLM."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Error: invalid JSON in tool arguments"

    if name not in TOOL_REGISTRY:
        return f"Unknown tool: {name}"

    return TOOL_REGISTRY[name](**args)


# ── Agent with tool calling ───────────────────────────────────────────────────
def run_tool_agent(query: str, llm=None) -> str:
    """One agent turn with tool calling."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools to answer accurately."},
        {"role": "user",   "content": query},
    ]

    if llm is None:
        return None  # signal to fall back to mock

    try:
        # First call: LLM decides which tool to use
        response = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # No tool call — LLM answered directly
        if not msg.tool_calls:
            return msg.content

        # Execute all tool calls
        tool_results = []
        for tc in msg.tool_calls:
            result = execute_tool_call(tc)
            tool_results.append({"call": tc, "result": result})
            print(f"  Tool called: {tc.function.name}({tc.function.arguments})")
            print(f"  Result:      {result}")

        # Second call: LLM synthesizes tool results into a final answer
        messages.append(msg)
        for tr in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tr["call"].id,
                "content": tr["result"],
            })

        final = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
        )
        return final.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AI Agents — Tool Calling")
    print("=" * 60)

    # Init Ollama
    llm = None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        llm = client
        print(f"\nOllama connected — using {LLM_MODEL}")
    except Exception:
        print(f"\nOllama not running — showing manual tool call simulation")

    # Test cases
    test_cases = [
        ("calculator",      "What is 15% of 840?"),
        ("get_weather",     "What's the weather like in Tokyo?"),
        ("unit_converter",  "Convert 100 km to miles"),
        ("calendar",        "What meetings do I have from July 5 to July 10 2026?"),
        ("multi-step",      "What is the weather in London, and convert 14°C to Fahrenheit?"),
    ]

    for label, query in test_cases:
        print(f"\n{'─'*60}")
        print(f"[{label}] Query: '{query}'")

        answer = run_tool_agent(query, llm)

        if answer is None:
            # Manual simulation — call tools directly to show the concept
            print("  Simulating tool call pipeline:")
            if "weather" in query.lower():
                city = re.search(r'in (\w+)', query)
                city = city.group(1) if city else "London"
                result = get_weather(city)
                print(f"  Tool: get_weather(city='{city}') → {result}")
            elif "%" in query or "calculate" in query.lower() or "what is" in query.lower()[:20]:
                expr = "0.15 * 840"
                result = calculator(expr)
                print(f"  Tool: calculator(expression='{expr}') → {result}")
            elif "km" in query or "celsius" in query.lower() or "fahrenheit" in query.lower():
                result = unit_converter(100, "km", "miles")
                print(f"  Tool: unit_converter(value=100, from_unit='km', to_unit='miles') → {result}")
            elif "meeting" in query.lower() or "event" in query.lower() or "calendar" in query.lower():
                result = get_calendar_events("2026-07-05", "2026-07-10")
                print(f"  Tool: get_calendar_events → {result}")
        else:
            print(f"  Answer: {answer}")

    # Show the tool definition schema
    print("\n" + "=" * 60)
    print("Tool Definition Schema (what the LLM reads)")
    print("=" * 60)
    print(json.dumps(TOOL_DEFINITIONS[0], indent=2))

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Tool definitions = JSON schema the LLM reads to pick tools")
    print("  - LLM returns tool_calls JSON → you execute → feed result back")
    print("  - Two-call pattern: (1) pick tool, (2) synthesize answer")
    print("  - Description quality directly determines tool selection accuracy")
    print("  - tool_choice='auto' lets LLM decide; 'none' forces prose; specific name forces that tool")
    print("=" * 60)
