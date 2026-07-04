# TextBlob

**What it is:** Simplified NLP library built on top of NLTK. One-liner sentiment analysis, translation, spell correction, noun phrase extraction.

**Before TextBlob:**
Sentiment required custom lexicon lookup or a trained classifier. TextBlob made it accessible in 2 lines — no model training needed.

**Why we picked it here:**
It's the fastest way to add basic sentiment to any project. Great for quick experiments. In the project (sentiment_analyzer), we use it alongside VADER as an ensemble — two quick scorers, no model downloads.

**When to use TextBlob:**
- Quick prototyping / experiments
- Small scripts where you need sentiment in 2 lines
- Ensemble with VADER for rule-based sentiment

**When NOT to use TextBlob:**
- Production sentiment (use a fine-tuned transformer)
- Complex NLP — it's just a convenience wrapper
- Performance-critical code

**vs VADER:**
VADER = tuned for social media (slang, caps, emojis). TextBlob = general purpose, also gives subjectivity score.

**Key API:**
```python
blob = TextBlob("I love this!")
blob.sentiment.polarity     # -1.0 to 1.0
blob.sentiment.subjectivity # 0.0 (objective) to 1.0 (subjective)
blob.noun_phrases           # ["machine learning", ...]
blob.correct()              # spell correction
```

**Install:** `pip install textblob`

**Introduced in:** Module 04 — NLP (example 12, project)
