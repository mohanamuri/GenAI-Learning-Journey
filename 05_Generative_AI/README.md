# Module 05 — Generative AI

> One-liner: Transformers learn patterns in data and generate new content — text, images, code — by predicting what comes next.

---

## What We Built

| # | File | What it demonstrates |
|---|------|----------------------|
| 01 | [01_tokenizers.py](examples/01_tokenizers.py) | Text → token IDs, tiktoken, token budget |
| 02 | [02_huggingface_pipeline.py](examples/02_huggingface_pipeline.py) | Run any model in 2 lines |
| 03 | [03_bert_embeddings.py](examples/03_bert_embeddings.py) | Contextual word vectors |
| 04 | [04_text_generation_gpt2.py](examples/04_text_generation_gpt2.py) | Autoregressive generation, greedy vs sampling |
| 05 | [05_temperature_topp_topk.py](examples/05_temperature_topp_topk.py) | Sampling knobs — creativity vs precision |
| 06 | [06_prompt_engineering.py](examples/06_prompt_engineering.py) | Zero-shot, few-shot, CoT, format control |
| 07 | [07_context_window.py](examples/07_context_window.py) | Token limits, sliding window, chunking |
| 08 | [08_hallucinations.py](examples/08_hallucinations.py) | What they are, types, mitigation |
| 09 | [09_structured_outputs.py](examples/09_structured_outputs.py) | JSON mode, Pydantic validation, retry |
| 10 | [10_attention_mechanism.py](examples/10_attention_mechanism.py) | Q/K/V math, causal mask from scratch |
| 11 | [11_bert_vs_gpt.py](examples/11_bert_vs_gpt.py) | Encoder vs decoder, when to use which |
| 12 | [12_embeddings.py](examples/12_embeddings.py) | Semantic search, clustering, FAQ match |
| — | [project/text_intelligence/](project/text_intelligence/) | Summarization + QA + zero-shot classification pipeline |

---

## Mental Models

```
Transformer = Attention + Feed Forward + Positional Encoding, stacked N times

BERT  (encoder)  → reads full sentence bidirectionally → understand
GPT   (decoder)  → reads left-to-right, generates next token → create
T5    (enc+dec)  → translate one sequence to another → transform
```

---

## One-Liners to Remember

- **Tokenization**: 1 token ≈ 4 chars ≈ 0.75 words. Count tokens before every API call.
- **Attention**: Each token votes on how much to attend to every other token. O(n²) cost.
- **BERT**: Bidirectional — sees whole sentence. Good at understanding, bad at generating.
- **GPT**: Left-to-right only. Predicts next token. Base ≠ instruction-tuned.
- **Temperature 0**: Deterministic, factual. Temperature >0: creative, varied.
- **Top-P**: Adaptive sampling — adapts to context. Prefer over top-K.
- **Context window**: Input + output combined. Key info at start or end, not middle.
- **Hallucination**: Model generates plausible, not necessarily true. Always ground with context.
- **Embeddings**: Similar meaning → nearby vectors. Backbone of semantic search and RAG.
- **Structured output**: JSON mode guarantees valid syntax, not correct values — still validate.

---

## What NOT to Do

| Mistake | Why |
|---------|-----|
| Use word count for token budget | Token count can be 2-3× word count for non-English text |
| Use BERT for text generation | It's encoder-only — not designed for it |
| Use GPT for classification without fine-tuning | BERT is better at understanding tasks |
| Set temperature >1 in production | Output becomes incoherent |
| Set both top_k and top_p | Pick one — both together is redundant |
| Skip token count before API call | Silent truncation → wrong answers |
| Use same embedding model as different module | Embeddings are incompatible across models |
| Ship LLM output without validation | Hallucinations, wrong JSON, wrong types |
| Put key info in middle of long context | "Lost in the middle" — model ignores it |
| Trust citations from LLM output | LLMs fabricate plausible-looking references |

---

## Frameworks Introduced

See [frameworks/README.md](frameworks/README.md)

| Framework | Purpose |
|-----------|---------|
| Hugging Face Transformers | Run BERT, GPT-2, T5 locally |
| tiktoken | Exact token counting for GPT models |

---

## Project

[Text Intelligence Pipeline](project/text_intelligence/) — Summarization + QA + zero-shot classification in one pipeline.

---

## Setup

```bash
pip install transformers torch tiktoken sentence-transformers pydantic
```

> First run downloads model weights (~250MB for distilbert). Cached in `~/.cache/huggingface`.
