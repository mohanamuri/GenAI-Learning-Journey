# Hugging Face Transformers

**What it is:** The standard library for downloading, running, and fine-tuning pre-trained transformer models (BERT, GPT-2, T5, LLaMA, etc.). Also the model hub at huggingface.co.

**Before Hugging Face (pre-2018):**
Each model had its own repo, its own format, its own loading code. Using BERT meant cloning Google's repo and manually downloading checkpoints. Switching models meant rewriting your pipeline. No unified API.

**Why we picked it here:**
Every concept in Module 05 (BERT, GPT, T5, attention) is best understood by running the actual models. Hugging Face makes that accessible in 2 lines — no infrastructure, no training. The `pipeline()` API hides all complexity for learners; the lower-level `AutoModel` API shows what's actually happening.

**When to use:**
- Running any pre-trained model locally (research, offline, privacy)
- Fine-tuning a model on your own dataset
- Learning what transformers actually do under the hood
- Any task where you need model weights locally

**When NOT to use:**
- When you just need an LLM API call → use OpenAI/Anthropic SDK (Module 06)
- When model is too large for your hardware → use API or quantized models
- When latency matters more than customization → API has faster infra

**Key APIs:**
```python
# High-level (easiest)
from transformers import pipeline
clf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
clf("I love this!")  # → [{'label': 'POSITIVE', 'score': 0.9998}]

# Low-level (full control)
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

**Tasks available via pipeline():**
`text-classification`, `text-generation`, `summarization`, `translation`, `question-answering`, `fill-mask`, `ner`, `zero-shot-classification`, `image-classification`, `automatic-speech-recognition`

**Model naming convention:**
- `distilbert-*` → smaller, faster BERT (use for learning + production)
- `bert-base-*` → standard BERT
- `gpt2` → smallest GPT-2 (117M params)
- `t5-small` → smallest T5
- `facebook/bart-*` → Facebook's BART

**Install:**
```bash
pip install transformers torch
# or for GPU:
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Coming up:** In Module 06 (LLM Engineering), we move from local models to cloud APIs (OpenAI, Anthropic). Hugging Face remains for open-source/local LLM use cases.

**Introduced in:** Module 05 — Generative AI (all examples)
