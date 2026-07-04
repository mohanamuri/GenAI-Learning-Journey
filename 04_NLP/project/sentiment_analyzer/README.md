# Sentiment Analyzer

Batch sentiment analysis on product reviews using VADER + TextBlob ensemble.

## What it demonstrates

- Text cleaning that preserves negation words
- VADER (rule-based) + TextBlob (lexicon) ensemble
- Per-product aggregation and summary
- Sentiment vs star-rating agreement check

## Run

```bash
pip install nltk textblob
python main.py
```

## Expected Output

```
============================
  PRODUCT REVIEW SENTIMENT ANALYZER
============================

ID   Product      Label      Score   Review
-------------------------------------------------------------------
1    Laptop       POSITIVE   0.712   Absolutely love this laptop!...
6    Laptop       NEGATIVE  -0.681   Terrible experience. Overheats...
...

SUMMARY BY PRODUCT
Laptop (5 reviews)
  Avg sentiment score : +0.123
  Avg star rating     : 3.0/5
  Positive: 2  Negative: 2  Neutral: 1
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `analyzer.py` | Core sentiment logic (clean → score → classify → report) |
| `data.py` | Sample reviews dataset |

## Key design decisions

- **Don't remove stopwords** — "not good", "never satisfied" need negation words
- **Ensemble score** = (VADER compound + TextBlob polarity) / 2 — reduces false positives
- **Clean minimally** — preserve `!` and `?` which VADER uses for emphasis scoring
