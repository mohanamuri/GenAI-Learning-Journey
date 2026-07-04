# Text Intelligence Pipeline

Combines 3 NLP tasks in one pipeline using Hugging Face pre-trained models — no training required.

## Tasks

| Task | Model | What it does |
|------|-------|-------------|
| Summarization | `distilbart-cnn-12-6` | Condenses long text to key points |
| Question Answering | `distilbert-base-cased-distilled-squad` | Extracts answers from a given context |
| Zero-Shot Classification | `bart-large-mnli` | Labels text without any task-specific training |

## Run

```bash
pip install transformers torch
python main.py
```

## Expected Output

```
─────────────────────────────────
Article: AI in Healthcare
─────────────────────────────────
Summary:
  AI is revolutionizing healthcare by enabling faster diagnoses...

Q&A:
  Q: What can AI detect in medical images?
  A: cancers, diabetic retinopathy (confidence: 87%)

Topic: healthcare
Scores: healthcare=0.91, technology=0.06, finance=0.01
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, orchestrates pipeline |
| `pipeline.py` | `TextIntelligencePipeline` class — models loaded once |
| `data.py` | Sample articles with questions and candidate labels |

## Key design decisions

- **Models loaded once** in `__init__` — never reinstantiate inside a loop
- **Zero-shot classification** — no labelled training data needed, just provide candidate labels
- **Truncation=True** on summarizer — prevents error on long inputs
- Clean separation: `pipeline.py` handles models, `main.py` handles output
