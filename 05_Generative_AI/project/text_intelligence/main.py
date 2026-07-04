"""
Text Intelligence Pipeline
---------------------------
Combines 3 Hugging Face tasks in one reusable pipeline:
  1. Summarization     — condense long text
  2. Question Answering — extract answers from context
  3. Zero-shot classification — label text without training

Run: python main.py
"""

from pipeline import TextIntelligencePipeline
from data import SAMPLE_ARTICLES


def main():
    print("=" * 65)
    print("  TEXT INTELLIGENCE PIPELINE")
    print("  Summarization + QA + Classification")
    print("=" * 65)

    pipe = TextIntelligencePipeline()

    for article in SAMPLE_ARTICLES:
        print(f"\n{'─' * 65}")
        print(f"Article: {article['title']}")
        print(f"{'─' * 65}")

        # 1. Summarize
        summary = pipe.summarize(article["text"])
        print(f"\nSummary:\n  {summary}")

        # 2. Answer questions
        print("\nQ&A:")
        for question in article["questions"]:
            answer, confidence = pipe.answer(question, article["text"])
            print(f"  Q: {question}")
            print(f"  A: {answer} (confidence: {confidence:.0%})")

        # 3. Classify topic
        topic, scores = pipe.classify(article["text"], article["candidate_labels"])
        print(f"\nTopic: {topic}")
        print(f"Scores: {', '.join(f'{l}={s:.2f}' for l, s in scores[:3])}")


if __name__ == "__main__":
    main()
