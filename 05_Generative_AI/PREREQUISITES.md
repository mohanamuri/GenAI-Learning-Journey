# Prerequisites — Module 05 Generative AI

## No API Keys — Runs Locally

All examples use Hugging Face Transformers, tiktoken, and sentence-transformers.
No cloud services, no API keys. Models download from Hugging Face on first run and cache locally.

---

## Install

```bash
pip install transformers torch tiktoken sentence-transformers pydantic scikit-learn
```

Verify:
```bash
python -c "import transformers, torch, tiktoken, sentence_transformers, pydantic; print('All good')"
```

> **Apple Silicon (M1/M2/M3)?** PyTorch with MPS support:
> ```bash
> pip install torch torchvision torchaudio
> ```
>
> **GPU (CUDA)?** Replace torch install:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> python -c "import torch; print('GPU:', torch.cuda.is_available())"
> ```

---

## What Each Example Does & How to Run

| Example | What it does | Model download | Run |
|---------|-------------|----------------|-----|
| `01_tokenizers.py` | Text → tokens → IDs, token budget, tiktoken | ~500 KB (tokenizer only) | `python 01_tokenizers.py` |
| `02_huggingface_pipeline.py` | Sentiment, summarization, QA, zero-shot in 2 lines | ~3.5 GB total | `python 02_huggingface_pipeline.py` |
| `03_bert_embeddings.py` | Contextual word vectors, "bank" disambiguation | ~260 MB | `python 03_bert_embeddings.py` |
| `04_text_generation_gpt2.py` | Autoregressive generation, greedy vs sampling | ~500 MB | `python 04_text_generation_gpt2.py` |
| `05_temperature_topp_topk.py` | Sampling knobs — creativity vs precision | Reuses GPT-2 cache | `python 05_temperature_topp_topk.py` |
| `06_prompt_engineering.py` | Zero-shot, few-shot, CoT, format control | Reuses GPT-2 cache | `python 06_prompt_engineering.py` |
| `07_context_window.py` | Token limits, sliding window, chunking | None (tiktoken only) | `python 07_context_window.py` |
| `08_hallucinations.py` | Types, examples, mitigation patterns | None (no model) | `python 08_hallucinations.py` |
| `09_structured_outputs.py` | JSON extraction, Pydantic validation, retry | None (no model) | `python 09_structured_outputs.py` |
| `10_attention_mechanism.py` | Q/K/V attention math from scratch (numpy) | None (numpy only) | `python 10_attention_mechanism.py` |
| `11_bert_vs_gpt.py` | Encoder vs decoder, fill-mask vs generate | ~240 MB (T5-small) | `python 11_bert_vs_gpt.py` |
| `12_embeddings.py` | Semantic search, clustering, FAQ matching | ~80 MB (MiniLM) | `python 12_embeddings.py` |

---

## Suggested Run Order

**Start here — instant, no downloads:**
```bash
python 07_context_window.py
python 08_hallucinations.py
python 09_structured_outputs.py
python 10_attention_mechanism.py
```

**Small downloads (~500 MB):**
```bash
python 01_tokenizers.py
python 04_text_generation_gpt2.py
python 05_temperature_topp_topk.py
python 06_prompt_engineering.py
```

**Medium downloads (~260–340 MB each):**
```bash
python 03_bert_embeddings.py
python 12_embeddings.py
python 11_bert_vs_gpt.py
```

**Heaviest — run last (~3.5 GB total, downloads all at once):**
```bash
python 02_huggingface_pipeline.py
```

---

## Model Downloads (Automatic on First Run)

All models cache at `~/.cache/huggingface`. Total first-run download: **~4–5 GB**.

| Example | Model | Size |
|---------|-------|------|
| `02_huggingface_pipeline.py` | `distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB |
| | `sshleifer/distilbart-cnn-12-6` | ~1.2 GB |
| | `distilbert-base-cased-distilled-squad` | ~260 MB |
| | `facebook/bart-large-mnli` | ~1.6 GB |
| `03_bert_embeddings.py` | `distilbert-base-uncased` | ~260 MB |
| `04_text_generation_gpt2.py` | `gpt2` | ~500 MB |
| `11_bert_vs_gpt.py` | `t5-small` | ~240 MB |
| `12_embeddings.py` | `all-MiniLM-L6-v2` | ~80 MB |

Models are shared — if `distilbert` is already cached from `02`, `03` reuses it.

---

## Check Cache Size

```bash
du -sh ~/.cache/huggingface
```

---

## No API Keys — No Cost

Everything runs locally. Module 06 is where API keys are first needed.
