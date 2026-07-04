# Author: Mohan Raju Amuri
"""
Hugging Face Pipeline — The Simplest Way to Use Any Model
-----------------------------------------------------------
One-liner: `pipeline("task", model="...")` — download, load, and run any model in 2 lines.

Remember:
- pipeline() is the high-level API — handles tokenization + model + decoding
- First call downloads the model (~hundreds of MB) and caches it in ~/.cache/huggingface
- Tasks: "text-generation", "text-classification", "summarization", "translation",
         "question-answering", "fill-mask", "ner", "zero-shot-classification"
- Default models are small and safe for learning — specify model= for production
- device=0 uses GPU (cuda:0), device="cpu" forces CPU

Don't:
- Don't reinstantiate pipeline() in a loop — it reloads the model every time
- Don't use default models in production — they're chosen for size, not accuracy
- Don't forget: model downloads happen on first run (needs internet + disk space)
- Don't use text-generation for classification tasks — pick the right task

Setup: pip install transformers torch
"""

from transformers import pipeline

# ── 1. Sentiment Classification ───────────────────────────────────────────
print("=== 1. Sentiment Classification ===")
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

reviews = [
    "This product is absolutely amazing! Best purchase ever.",
    "Terrible quality. Broke after one day. Do not buy.",
    "It's okay I guess. Nothing special.",
]
results = classifier(reviews)
for review, result in zip(reviews, results):
    print(f"  [{result['label']:<8} {result['score']:.2f}]  {review[:50]}")

# ── 2. Text Summarization ─────────────────────────────────────────────────
print("\n=== 2. Summarization ===")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

article = """
Machine learning is a subset of artificial intelligence that gives systems the ability
to automatically learn and improve from experience without being explicitly programmed.
Machine learning focuses on the development of computer programs that can access data
and use it to learn for themselves. The process begins with observations or data, such
as examples, direct experience, or instruction, so that computers can learn to make
better decisions in the future.
"""
summary = summarizer(article, max_length=60, min_length=20, do_sample=False)
print(f"Original ({len(article.split())} words):")
print(f"  {article.strip()[:100]}...")
print(f"Summary ({len(summary[0]['summary_text'].split())} words):")
print(f"  {summary[0]['summary_text']}")

# ── 3. Question Answering ─────────────────────────────────────────────────
print("\n=== 3. Question Answering ===")
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = """
Python was created by Guido van Rossum and first released in 1991.
It is a high-level, interpreted programming language known for its clear syntax.
Python is widely used in data science, machine learning, and web development.
"""
questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
]
for q in questions:
    answer = qa(question=q, context=context)
    print(f"  Q: {q}")
    print(f"  A: {answer['answer']} (confidence: {answer['score']:.2f})\n")

# ── 4. Zero-shot Classification (no training needed) ──────────────────────
print("=== 4. Zero-Shot Classification ===")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The new iPhone has a better camera and longer battery life"
labels = ["technology", "sports", "politics", "food", "finance"]
result = zero_shot(text, candidate_labels=labels)

print(f"Text: {text}")
for label, score in zip(result["labels"], result["scores"]):
    bar = "█" * int(score * 20)
    print(f"  {label:<12} {score:.3f}  {bar}")
