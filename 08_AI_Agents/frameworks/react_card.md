# Framework: ReAct (Reasoning + Acting)

## What is it?
A prompting pattern (not a library) for agent reasoning.
Interleaves Thought (chain-of-thought reasoning) and Action (tool call) steps, with Observation (tool result) fed back after each action.
Introduced in: "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022).

## What existed before?
- **Chain-of-thought (CoT)**: just reasoning, no tool calls
- **Tool-augmented LLMs**: tool calls without explicit reasoning step — prone to random tool selection
- **MRKL systems**: early tool-augmented agents, no standardized reasoning format

## Why ReAct works
Thoughts force the model to plan before acting → fewer wrong tool calls.
Observations ground the next thought in real tool output → less hallucination.
The full trace is human-readable → easy to debug when the agent goes wrong.

## The prompt format

```
You are a helpful assistant. Use tools to answer.

Thought: I need to find the price. I'll search the knowledge base.
Action: search_kb
Action Input: pricing
Observation: Starter $99/mo | Professional $499/mo

Thought: I have the pricing. The user asked about Professional, which is $499/mo.
Final Answer: The Professional plan costs $499 per month.
```

## Key implementation details

```python
# Stop generation before the model invents observations
response = llm.generate(prompt, stop=["Observation:"])

# Parse output (regex-based)
thought = re.search(r"Thought:\s*(.*?)(?=\nAction:)", text)
action  = re.search(r"Action:\s*(\w+)", text)
ainput  = re.search(r"Action Input:\s*(.*?)(?=\nObservation:|\Z)", text)
final   = re.search(r"Final Answer:\s*(.*)", text)
```

## When to use vs skip

**Use ReAct when:**
- The LLM doesn't support structured tool calling (e.g. smaller models)
- You need interpretable reasoning traces (debugging)
- The task is open-ended and may require dynamic replanning

**Skip ReAct (use tool calling) when:**
- The LLM supports structured function calling (llama3.1+, GPT-4, Claude)
- You need reliable JSON output — ReAct parsing can fail on malformed output
- Latency is critical — tool calling requires fewer tokens

## Install
No install — just a prompting pattern. Compatible with any LLM.
