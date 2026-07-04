"""
Core sentiment analysis logic.
Combines VADER + TextBlob scores and classifies each review.
"""

import re
import string
from collections import defaultdict, Counter

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

nltk.download("vader_lexicon", quiet=True)


def clean_text(text: str) -> str:
    """Minimal cleaning — preserve negation words for sentiment accuracy."""
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)           # remove URLs
    text = re.sub(r"[^\w\s'!?]", " ", text)       # keep apostrophes + ! ?
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify(compound: float) -> str:
    if compound >= 0.05:
        return "POSITIVE"
    elif compound <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, review: dict) -> dict:
        text = review["review"]
        cleaned = clean_text(text)

        # VADER
        vader_scores = self.vader.polarity_scores(cleaned)
        compound = vader_scores["compound"]

        # TextBlob
        blob = TextBlob(cleaned)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Ensemble: average compound and polarity (normalize polarity to -1..1)
        ensemble_score = (compound + polarity) / 2
        label = classify(ensemble_score)

        return {
            "id": review["id"],
            "product": review["product"],
            "review": text[:60] + "..." if len(text) > 60 else text,
            "vader_compound": round(compound, 3),
            "textblob_polarity": round(polarity, 3),
            "ensemble_score": round(ensemble_score, 3),
            "subjectivity": round(subjectivity, 3),
            "label": label,
            "actual_rating": review["rating"],
        }

    def analyze_batch(self, reviews: list[dict]) -> list[dict]:
        return [self.analyze(r) for r in reviews]

    def print_report(self, results: list[dict]) -> None:
        # Per-review results
        print(f"\n{'ID':<4} {'Product':<12} {'Label':<10} {'Score':>7}  {'Review'}")
        print("-" * 75)
        for r in results:
            print(
                f"{r['id']:<4} {r['product']:<12} {r['label']:<10} "
                f"{r['ensemble_score']:>7.3f}  {r['review'][:45]}"
            )

        # Summary by product
        print("\n" + "=" * 60)
        print("  SUMMARY BY PRODUCT")
        print("=" * 60)

        by_product = defaultdict(list)
        for r in results:
            by_product[r["product"]].append(r)

        for product, items in by_product.items():
            labels = Counter(r["label"] for r in items)
            avg_score = sum(r["ensemble_score"] for r in items) / len(items)
            avg_rating = sum(r["actual_rating"] for r in items) / len(items)
            print(f"\n{product} ({len(items)} reviews)")
            print(f"  Avg sentiment score : {avg_score:+.3f}")
            print(f"  Avg star rating     : {avg_rating:.1f}/5")
            print(f"  Positive: {labels['POSITIVE']}  Negative: {labels['NEGATIVE']}  Neutral: {labels['NEUTRAL']}")

        # Overall
        print("\n" + "=" * 60)
        print("  OVERALL")
        print("=" * 60)
        all_labels = Counter(r["label"] for r in results)
        total = len(results)
        for label in ("POSITIVE", "NEUTRAL", "NEGATIVE"):
            count = all_labels[label]
            bar = "█" * count
            print(f"  {label:<10} {count:>3} / {total}  {bar}")

        # Agreement: sentiment label vs star rating
        agree = sum(
            1 for r in results
            if (r["label"] == "POSITIVE" and r["actual_rating"] >= 4)
            or (r["label"] == "NEGATIVE" and r["actual_rating"] <= 2)
            or (r["label"] == "NEUTRAL" and r["actual_rating"] == 3)
        )
        print(f"\n  Sentiment ↔ Rating agreement: {agree}/{total} ({agree/total*100:.0f}%)")
