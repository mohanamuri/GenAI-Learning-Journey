# Prerequisites — Module 05 Generative AI

## Install

```bash
pip install transformers torch tiktoken sentence-transformers pydantic scikit-learn
```

## No Extra Downloads Needed

Models download automatically from Hugging Face on first run.
Cached at `~/.cache/huggingface` — never re-downloaded.

## Model Download Map

| Example | Model | Size | Downloads When |
|---------|-------|------|----------------|
| `01_tokenizers.py` | `bert-base-uncased` (tokenizer only) | ~500 KB | First run |
| `02_huggingface_pipeline.py` | `distilbert-base-uncased-finetuned-sst-2-english` | ~260 MB | First run |
| | `sshleifer/distilbart-cnn-12-6` | ~1.2 GB | First run |
| | `distilbert-base-cased-distilled-squad` | ~260 MB | First run |
| | `facebook/bart-large-mnli` | ~1.6 GB | First run |
| `03_bert_embeddings.py` | `distilbert-base-uncased` | ~260 MB | First run |
| `04_text_generation_gpt2.py` | `gpt2` (117M params) | ~500 MB | First run |
| `05_temperature_topp_topk.py` | `gpt2` | — | Already cached |
| `06_prompt_engineering.py` | `gpt2` | — | Already cached |
| `07_context_window.py` | None (tiktoken only) | — | Instant |
| `08_hallucinations.py` | None (no model) | — | Instant |
| `09_structured_outputs.py` | None (no model) | — | Instant |
| `10_attention_mechanism.py` | None (numpy only) | — | Instant |
| `11_bert_vs_gpt.py` | `distilbert-base-uncased`, `gpt2`, `t5-small` | ~240 MB | First run |
| `12_embeddings.py` | `all-MiniLM-L6-v2` | ~80 MB | First run |

**Total first-run download: ~4–5 GB**

## No API Keys Needed

Everything in this module runs locally. Module 06 (LLM Engineering) is where API keys
(OpenAI, Anthropic) are first required.

## Suggested Run Order

**Start here — instant, no downloads:**
```
07_context_window.py
08_hallucinations.py
09_structured_outputs.py
10_attention_mechanism.py
```

**Small downloads (~500 MB total):**
```
01_tokenizers.py
04_text_generation_gpt2.py
05_temperature_topp_topk.py
06_prompt_engineering.py
12_embeddings.py
```

**Larger downloads (~1.5 GB total):**
```
03_bert_embeddings.py
11_bert_vs_gpt.py
```

**Heaviest — run last (~3 GB):**
```
02_huggingface_pipeline.py
```

## Sanity Check

```bash
python -c "import transformers, torch, tiktoken, sentence_transformers, pydantic; print('All good')"
```

## GPU (Optional)

Examples run on CPU by default. For GPU (CUDA), replace torch install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check GPU availability:
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```
