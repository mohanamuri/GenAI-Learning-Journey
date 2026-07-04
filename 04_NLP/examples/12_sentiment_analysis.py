# Author: Mohan Raju Amuri
"""
Sentiment Analysis
------------------
One-liner: Determine if text is positive, negative, or neutral.

Tools:
- VADER (NLTK): rule-based, fast, great for social media / short text
- TextBlob: simple, good for quick experiments
- Transformers: most accurate, use for production

VADER scores:
- compound: -1 (most negative) to +1 (most positive)
- compound >= 0.05  → positive
- compound <= -0.05 → negative
- else              → neutral

Remember:
- VADER handles slang, emojis, CAPS, punctuation!!!
- TextBlob polarity: -1 to 1 | subjectivity: 0 (objective) to 1 (subjective)
- For production → use a transformer model (cardiffnlp/twitter-roberta-base-sentiment)

Don't:
- Don't remove stopwords before sentiment analysis ("not bad" → "bad" ← wrong)
- Don't use VADER for long formal documents — it's tuned for social media
- Don't trust polarity alone — check subjectivity too
  → "The sky is blue" is neutral, not positive — low subjectivity
"""

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("vader_lexicon", quiet=True)

sia = SentimentIntensityAnalyzer()

# --- VADER ---
texts = [
    "I LOVE this product!!! Best purchase ever 😍",
    "This is absolutely terrible. Complete waste of money.",
    "The product is okay. Nothing special.",
    "Not bad at all, actually quite decent",        # double negative
    "The movie was not good",                       # negation
    "The movie was not bad",                        # negation
]

print("=== VADER Sentiment ===")
print(f"{'Text':<50} {'Compound':>9} {'Label'}")
print("-" * 72)
for text in texts:
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    print(f"{text[:48]:<50} {compound:>9.3f}  {label}")

# --- TextBlob ---
print("\n=== TextBlob Sentiment ===")
print(f"{'Text':<45} {'Polarity':>9} {'Subjectivity':>13} {'Label'}")
print("-" * 75)
for text in texts[:4]:
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    label = "POS" if pol > 0 else ("NEG" if pol < 0 else "NEU")
    print(f"{text[:43]:<45} {pol:>9.3f} {subj:>13.3f}  {label}")

# --- Batch sentiment for a dataset ---
print("\n=== Batch processing ===")
reviews = [
    "Excellent service, very happy!",
    "Waited for 2 hours. Unacceptable.",
    "Average experience, nothing memorable.",
    "Would definitely recommend to friends!",
]
results = [(r, sia.polarity_scores(r)["compound"]) for r in reviews]
pos = sum(1 for _, s in results if s >= 0.05)
neg = sum(1 for _, s in results if s <= -0.05)
neu = len(results) - pos - neg
print(f"Positive: {pos} | Negative: {neg} | Neutral: {neu}")
