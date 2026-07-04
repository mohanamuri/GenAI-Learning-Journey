# Module 06 — LLM Engineering

> One-liner: Stop training models, start calling them — LLM Engineering is about using powerful models via API reliably, cheaply, and safely.

---

## What We Built

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | [01_openai_basics.py](examples/01_openai_basics.py) | Chat completions, token usage, model selection |
| 02 | [02_anthropic_basics.py](examples/02_anthropic_basics.py) | Claude API, key differences from OpenAI |
| 03 | [03_system_prompts.py](examples/03_system_prompts.py) | Role, tone, constraints, injection defense |
| 04 | [04_prompt_templates.py](examples/04_prompt_templates.py) | Reusable, validated, versionable prompts |
| 05 | [05_streaming.py](examples/05_streaming.py) | Token-by-token output, when to stream |
| 06 | [06_json_mode.py](examples/06_json_mode.py) | Structured output, Pydantic validation |
| 07 | [07_function_calling.py](examples/07_function_calling.py) | Tool use — LLM picks + params, you execute |
| 08 | [08_chat_history.py](examples/08_chat_history.py) | Stateless API, sliding window, session storage |
| 09 | [09_cost_optimization.py](examples/09_cost_optimization.py) | Model selection, prompt compression, caching |
| 10 | [10_error_handling.py](examples/10_error_handling.py) | Retry strategy, fallback chain, error taxonomy |
| 11 | [11_ollama_local_llm.py](examples/11_ollama_local_llm.py) | Local LLMs, cloud↔local swap pattern |
| — | [project/smart_chatbot/](project/smart_chatbot/) | CLI chatbot with history, streaming, cost tracking |

---

## One-Liners to Remember

- **Stateless**: LLM has no memory — you send full history every call.
- **System prompt tokens**: Charged on every single call — keep it under 200 tokens.
- **Model default**: Use `gpt-4o-mini` or `claude-haiku-4-5` — 30× cheaper, handles 80% of tasks.
- **max_tokens**: Always set it — prevents runaway costs on long outputs.
- **JSON mode**: Guarantees valid JSON syntax, not correct values. Always validate.
- **Streaming**: Same total cost/latency — just better perceived speed for users.
- **Function calling**: LLM returns function name + args. YOU execute the function.
- **Rate limits**: Exponential backoff + jitter. Never retry auth errors.
- **Ollama**: Same OpenAI SDK, different `base_url`. Free, local, private.
- **Prompt templates**: Prompts are code. Version them, test them, validate inputs.

---

## What NOT to Do

| Mistake | Why |
|---------|-----|
| Hardcode API keys | Use env vars or `.env` — never commit keys |
| Use gpt-4o for everything | gpt-4o-mini is 30× cheaper and handles most tasks |
| Omit `max_tokens` | Runaway output = unexpected cost |
| Put system prompt instructions in user prompt | Wrong role confuses the model |
| Skip Pydantic validation on JSON mode output | Valid JSON ≠ correct types or values |
| Let chat history grow unbounded | Token cost grows every turn — implement windowing |
| Retry auth errors | API key is wrong — retrying wastes money and time |
| Trust function call arguments blindly | LLM can hallucinate parameter values — validate |
| Rebuild client per request | Expensive — instantiate once, reuse |
| Use streaming for batch/programmatic jobs | Adds complexity with no benefit |

---

## Frameworks Introduced

See [frameworks/README.md](frameworks/README.md)

| Framework | Purpose |
|-----------|---------|
| OpenAI SDK | GPT-4o, GPT-4o-mini via API |
| Anthropic SDK | Claude models via API |
| Ollama | Local open-source LLMs (free) |

---

## Project

[Smart Chatbot](project/smart_chatbot/) — CLI chatbot with system prompt, sliding window history, streaming, and per-session cost tracking.

---

## API Keys Needed

This is the first module requiring API keys.

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

See [PREREQUISITES.md](PREREQUISITES.md) for setup details.
