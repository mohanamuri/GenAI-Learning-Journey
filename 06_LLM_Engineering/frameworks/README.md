# Frameworks — Module 06 LLM Engineering

| Framework | File | Why Introduced |
|-----------|------|----------------|
| OpenAI SDK | [openai_sdk.md](openai_sdk.md) | Call GPT models via API — the industry standard |
| Anthropic SDK | [anthropic_sdk.md](anthropic_sdk.md) | Call Claude models — different API, same pattern |
| Ollama | [ollama.md](ollama.md) | Run LLMs locally — no API key, no cost |

---

## Framework Evolution

```
Module 01  Pure Python
Module 02  scikit-learn
Module 03  TensorFlow / Keras
Module 04  NLTK, spaCy, Gensim, TextBlob, sentence-transformers
Module 05  Hugging Face Transformers, tiktoken
Module 06  OpenAI SDK, Anthropic SDK, Ollama        ← HERE
           (cloud LLMs via API + local LLMs)
Module 07  FAISS, ChromaDB                           [coming]
Module 08  LangGraph, CrewAI                         [coming]
Module 09  MLflow, FastAPI, Docker                   [coming]
```

**The shift in Module 06:** We stop training or loading models locally for most tasks.
We call a hosted API instead — faster, more capable, cost-per-token.
Ollama is the exception: local for dev, privacy, or cost control.
