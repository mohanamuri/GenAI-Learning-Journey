"""
TextIntelligencePipeline — core logic.
Loads models once, exposes clean interface.
"""

from transformers import pipeline as hf_pipeline


class TextIntelligencePipeline:
    def __init__(self):
        print("Loading models (first run downloads weights)...")

        self._summarizer = hf_pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
        )
        self._qa = hf_pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
        )
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        print("Models ready.\n")

    def summarize(self, text: str, max_length: int = 80, min_length: int = 25) -> str:
        """Condense text to a short summary."""
        # DistilBART needs at least ~50 tokens to summarize well
        if len(text.split()) < 30:
            return text  # too short to summarize

        result = self._summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()

    def answer(self, question: str, context: str) -> tuple[str, float]:
        """Extract answer from context. Returns (answer, confidence)."""
        result = self._qa(question=question, context=context)
        return result["answer"], result["score"]

    def classify(
        self, text: str, candidate_labels: list[str]
    ) -> tuple[str, list[tuple[str, float]]]:
        """Zero-shot topic classification. Returns (top_label, sorted_scores)."""
        result = self._classifier(text, candidate_labels=candidate_labels)
        scores = list(zip(result["labels"], result["scores"]))
        return scores[0][0], scores
