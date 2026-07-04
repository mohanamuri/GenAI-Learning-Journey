# LLM Engineering — Interview Questions

## API & SDK

**Q: OpenAI vs Anthropic SDK — key structural difference?**
Anthropic: `system=` is a top-level parameter, NOT inside `messages`. `max_tokens` is required. Response text is `content[0].text` not `choices[0].message.content`.

**Q: How does the OpenAI SDK handle rate limits?**
`OpenAI(max_retries=3)` enables built-in exponential backoff. For custom control, catch `RateLimitError` and implement your own backoff with jitter.

**Q: What is Ollama?**
Local LLM server. Exposes an OpenAI-compatible API at `localhost:11434`. Same SDK code works — just change `base_url`. Free, private, no API key.

---

## Prompting

**Q: What belongs in the system prompt vs user prompt?**
System prompt: persistent role, tone, output format, constraints, refusals.
User prompt: the specific task, data, context for this request.

**Q: What is prompt injection?**
User tries to override system instructions: "Ignore previous instructions and...". Mitigate with explicit constraints + output validation — can't fully prevent with prompting alone.

**Q: Why use prompt templates instead of f-strings?**
Templates are testable, versionable, and validated. f-strings are fragile — missing variable = silent failure or wrong output. Templates fail loudly.

**Q: What is few-shot prompting?**
Include 2-3 input/output examples in the prompt to demonstrate the expected format/behavior. More reliable than zero-shot for structured output.

**Q: Chain-of-Thought prompting — when to use?**
Add "Think step by step" for reasoning tasks, math, multi-hop questions. Dramatically improves accuracy. Not needed for simple classification or extraction.

---

## Architecture & Cost

**Q: Are LLMs stateful between API calls?**
No. Every call is stateless. You send the full conversation history (`messages` list) every time. Implementing "memory" is your responsibility.

**Q: How do you manage growing chat history costs?**
Sliding window (trim oldest messages), summarization (compress history into a summary message), or vector memory (store old turns as embeddings, retrieve relevant ones).

**Q: How would you cut LLM API costs by 80%?**
1. Switch from GPT-4o to GPT-4o-mini (30× cheaper, similar quality for most tasks)
2. Compress prompts — remove fluff from system prompt
3. Set `max_tokens` — cap output length
4. Cache repeated/similar queries
5. Use batch API for non-realtime jobs (50% cheaper on OpenAI)

**Q: What is the difference between streaming and non-streaming?**
Same total tokens, same cost, same latency. Streaming shows tokens as they generate → better perceived UX. Use for chat UI. Skip for batch/programmatic jobs.

---

## Reliability

**Q: Which errors should you retry? Which should you not?**
Retry: `RateLimitError`, `APITimeoutError`, `APIConnectionError`, `InternalServerError`.
Never retry: `AuthenticationError` (fix the key), `BadRequestError` (fix the request).

**Q: What is exponential backoff?**
Wait 1s after first failure, 2s after second, 4s after third... Add random jitter (±0-0.5s) to prevent thundering herd when many clients retry simultaneously.

**Q: How do you prevent token limit errors?**
Count tokens with tiktoken before sending. Truncate or chunk input if it exceeds `model_limit - reserved_output_tokens`. Use `max_tokens` to cap output.

---

## Function Calling

**Q: Explain function calling.**
1. Define tools as JSON schemas. 2. LLM reads user message + tool definitions. 3. LLM returns `tool_calls` with function name + arguments (as JSON). 4. YOU execute the function. 5. Send result back for final response. LLM never runs code directly.

**Q: How do you prevent LLM from hallucinating function arguments?**
Validate arguments with Pydantic before executing. Define strict schemas with enums where possible. Log all tool calls for debugging.

---

## Structured Output

**Q: JSON mode vs structured outputs?**
JSON mode: guarantees valid JSON syntax, schema defined only in prompt — field values can still be wrong.
Structured outputs (`.parse()` + Pydantic): schema enforced at token level — structure is guaranteed correct.

**Q: Should you trust JSON mode output without validation?**
No. Valid JSON ≠ correct values or types. Always validate with Pydantic. Use `try/except ValidationError`.
