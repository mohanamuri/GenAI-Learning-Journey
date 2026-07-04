# Smart Chatbot

CLI chatbot that demonstrates all core Module 06 LLM engineering patterns.

## What it demonstrates

| Pattern | Where |
|---------|-------|
| System prompt | `chatbot.py` — SYSTEM_PROMPT constant |
| Sliding window history | `_trim_history()` — keeps last N tokens |
| Streaming responses | `stream=True` in `chat()` |
| Cost tracking | `total_input_tokens` + `total_output_tokens` |
| Cloud ↔ local swap | `--local` flag switches to Ollama |

## Run

```bash
# Cloud (needs OPENAI_API_KEY)
python main.py

# Local (needs Ollama running + model pulled)
python main.py --local

# Specific model
python main.py --model gpt-4o
python main.py --local --model llama3.1:8b
```

## Commands

| Command | What it does |
|---------|-------------|
| `history` | Show current context window + token count |
| `cost` | Show session token usage + estimated cost |
| `quit` / `exit` | End session, print final summary |

## Expected Output

```
==================================================
  AI Engineering Tutor  [Cloud (gpt-4o-mini)]
==================================================
Type 'quit' to end. Type 'history'. Type 'cost'.

You: What is RAG?
Bot: RAG (Retrieval-Augmented Generation) combines a retrieval
system with an LLM...

You: cost

--- Session Summary ---
  Model          : gpt-4o-mini
  Input tokens   : 847
  Output tokens  : 312
  Total tokens   : 1,159
  Approx cost    : $0.00032
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, command handling |
| `chatbot.py` | Core logic — history, streaming, cost tracking |
