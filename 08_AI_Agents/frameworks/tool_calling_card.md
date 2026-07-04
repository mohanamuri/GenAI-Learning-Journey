# Framework: OpenAI Tool Calling (Function Calling)

## What is it?
A standardized way for LLMs to request function calls by returning structured JSON instead of prose.
The model outputs `tool_calls` objects with `name` and `arguments` — you execute them and return results.
Supported by: OpenAI GPT-4, Anthropic Claude, Ollama (llama3.1+), and most modern LLMs.

## What existed before?
- **ReAct text parsing**: LLM writes "Action: tool_name" as free text → fragile regex parsing
- **Manual JSON prompting**: "Respond with JSON like {action: ..., input: ...}" → model often disobeys
- **MRKL output parsing**: proprietary parsing schemes per-model

## Why tool calling is better than ReAct
- Structured output: model returns valid JSON — no parsing failures
- Parallel tool calls: model can call multiple tools in one response
- Strong type enforcement: parameter schema validated before execution
- Lower token cost: no Thought/Observation formatting overhead

## Core code pattern

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Step 1: Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city. Use for any weather question.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "e.g. 'London'"}
            },
            "required": ["city"],
        },
    },
}]

# Step 2: First call — LLM picks tool
response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",  # "none" | "auto" | {"type": "function", "function": {"name": ...}}
)
msg = response.choices[0].message

# Step 3: Execute tool calls
for tc in msg.tool_calls:
    name = tc.function.name
    args = json.loads(tc.function.arguments)
    result = your_tools[name](**args)

    # Step 4: Add result to messages
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

# Step 5: Second call — LLM synthesizes answer
final = client.chat.completions.create(model="llama3.2:3b", messages=messages)
```

## tool_choice options

| Value | Effect |
|-------|--------|
| `"auto"` | LLM decides whether to call a tool |
| `"none"` | Force prose response, no tool calls |
| `{"type": "function", "function": {"name": "get_weather"}}` | Force specific tool |
| `"required"` | Force at least one tool call |

## Parallel tool calls

The model can return multiple tool_calls in one response — execute them all before the second LLM call:

```python
# LLM returns both tool calls at once
for tc in response.choices[0].message.tool_calls:  # may have multiple
    result = execute(tc)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
```

## When to use vs skip

**Use tool calling when:**
- The LLM supports it (llama3.1+, GPT-4, Claude)
- You need reliable, parseable output
- You want parallel tool execution

**Skip tool calling (use ReAct) when:**
- Using smaller models that don't support structured output
- You need very detailed reasoning traces
- Debugging agent behavior (ReAct traces are more human-readable)

## Install
```bash
pip install openai  # works with Ollama, OpenAI, Anthropic via compatible endpoints
```
