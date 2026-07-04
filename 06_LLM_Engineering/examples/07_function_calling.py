# Author: Mohan Raju Amuri
"""
Function Calling (Tool Use)
-----------------------------
One-liner: LLM decides WHEN to call a function and WHAT arguments to pass — you execute it.

How it works:
  1. You define tools (function name + JSON schema of parameters)
  2. LLM reads user message + tool definitions
  3. LLM returns: either a normal response OR a tool_call with arguments
  4. You execute the function with those arguments
  5. You send the result back to the LLM for a final response

Remember:
- LLM does NOT execute the function — it only returns the name + args as JSON
- You always execute the function in your code
- Multiple tools can be defined; LLM picks the right one (or none)
- Tool descriptions matter — write them clearly, LLM uses them to decide

Don't:
- Don't expose dangerous functions as tools (file deletion, DB writes without auth)
- Don't skip validating tool arguments — LLM can hallucinate parameter values
- Don't forget to send tool results back to get the final response
- Don't use function calling for tasks a simple prompt handles — adds latency

Running on Ollama (local, free — requires a model that supports tool use):
  ollama pull llama3.1:8b   ← supports function calling
  (llama3.2:3b may have limited tool support — use 3.1:8b for this example)
"""

import json
from openai import OpenAI

# ── Using Ollama (local LLM) — free, no API key required ──────────────────
# Important: function calling requires a model that supports tool use.
# llama3.1:8b and mistral:7b support it. llama3.2:3b has limited support.
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL = "llama3.1:8b"  # use a model with tool support


# ── Real functions (your code executes these, NOT the LLM) ────────────────
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulate a weather API call. In production, this calls a real weather API."""
    fake_data = {
        "London":   {"temp": 15, "condition": "cloudy"},
        "Tokyo":    {"temp": 28, "condition": "sunny"},
        "New York": {"temp": 22, "condition": "partly cloudy"},
    }
    data = fake_data.get(city, {"temp": 20, "condition": "unknown"})
    return {"city": city, "temperature": data["temp"], "unit": unit, "condition": data["condition"]}


def calculate(expression: str) -> dict:
    """Safe arithmetic evaluator. Never use eval() on untrusted input in production."""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return {"error": "Invalid characters in expression"}
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


# ── Tool definitions (JSON schema) ────────────────────────────────────────
# The LLM reads these descriptions to decide which tool to call and with what args.
# Clear descriptions = better tool selection. Vague descriptions = wrong tool calls.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Call this when user asks about weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string",  "description": "City name"},
                    "unit":  {"type": "string",  "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression. Call this for arithmetic questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '15 * 24'"},
                },
                "required": ["expression"],
            },
        },
    },
]

FUNCTION_MAP = {"get_weather": get_weather, "calculate": calculate}


# ── Core function-calling loop ────────────────────────────────────────────
def chat_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    # Step 1: LLM decides whether to call a tool
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",   # LLM decides: call a tool or answer directly
    )

    msg = response.choices[0].message

    # Step 2: No tool call — return direct answer
    if not msg.tool_calls:
        return msg.content

    # Step 3: Execute each requested tool
    # The LLM only returns function name + JSON args — we run the actual function
    messages.append(msg)

    for tool_call in msg.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        print(f"  [Tool called] {name}({args})")

        # Validate args before executing — LLM can hallucinate wrong values
        result = FUNCTION_MAP[name](**args)

        # Step 4: Send tool result back so LLM can formulate a natural response
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

    final = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return final.choices[0].message.content


# ── Test queries ──────────────────────────────────────────────────────────
queries = [
    "What's the weather like in Tokyo?",
    "What is 847 multiplied by 23?",
    "What's the capital of France?",   # no tool needed — direct answer
]

for query in queries:
    print(f"\nUser : {query}")
    answer = chat_with_tools(query)
    print(f"LLM  : {answer}")
