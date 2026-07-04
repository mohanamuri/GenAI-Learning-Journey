"""
Sentiment Analyzer — Product Review Pipeline
---------------------------------------------
Demonstrates: text cleaning → sentiment scoring → aggregation → report

Features:
- Multi-method scoring (VADER + TextBlob)
- Batch processing
- Summary statistics
- Actionable output

Run: python main.py
"""

from analyzer import SentimentAnalyzer
from data import SAMPLE_REVIEWS


def main():
    analyzer = SentimentAnalyzer()

    print("=" * 60)
    print("  PRODUCT REVIEW SENTIMENT ANALYZER")
    print("=" * 60)

    results = analyzer.analyze_batch(SAMPLE_REVIEWS)
    analyzer.print_report(results)


if __name__ == "__main__":
    main()
