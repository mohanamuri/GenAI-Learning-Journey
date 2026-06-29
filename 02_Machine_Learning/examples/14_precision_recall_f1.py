"""
=========================================================
Topic      : Machine Learning

Lesson     : Precision, Recall & F1-Score

File       : 14_precision_recall_f1.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates Precision, Recall and F1-Score
using Scikit-Learn.

Learning Objectives
-------------------
1. Understand Precision.
2. Understand Recall.
3. Understand F1-Score.
4. Learn why Accuracy alone is not enough.
=========================================================
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Actual Labels
# --------------------------------------------------

y_actual = [
    1, 1, 1, 1,
    0, 0, 0, 0
]

# --------------------------------------------------
# Predicted Labels
# --------------------------------------------------

y_predicted = [
    1, 1, 0, 1,
    0, 1, 0, 0
]


def main():
    """Main Function."""

    print("=" * 60)
    print(" Precision, Recall & F1 Score ")
    print("=" * 60)

    accuracy = accuracy_score(
        y_actual,
        y_predicted,
    )

    precision = precision_score(
        y_actual,
        y_predicted,
    )

    recall = recall_score(
        y_actual,
        y_predicted,
    )

    f1 = f1_score(
        y_actual,
        y_predicted,
    )

    print("\nEvaluation Metrics")
    print("-" * 30)

    print(f"Accuracy  : {accuracy:.2f}")

    print(f"Precision : {precision:.2f}")

    print(f"Recall    : {recall:.2f}")

    print(f"F1 Score  : {f1:.2f}")

    print("\nObservation")
    print("-" * 30)

    print("Accuracy measures overall correctness.")

    print("Precision measures trust in positive predictions.")

    print("Recall measures ability to find positives.")

    print("F1 Score balances Precision and Recall.")


if __name__ == "__main__":
    main()

#python3 14_precision_recall_f1.py