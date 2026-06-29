"""
=========================================================
Topic      : Machine Learning

Lesson     : Confusion Matrix

File       : 13_confusion_matrix.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how to generate a Confusion
Matrix using Scikit-Learn.

Learning Objectives
-------------------
1. Understand TP, TN, FP and FN.
2. Generate a Confusion Matrix.
3. Interpret model predictions.
=========================================================
"""

from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# Actual Labels
# --------------------------------------------------

y_actual = [
    "Pass",
    "Pass",
    "Pass",
    "Fail",
    "Fail",
    "Fail",
]

# --------------------------------------------------
# Predicted Labels
# --------------------------------------------------

y_predicted = [
    "Pass",
    "Pass",
    "Fail",
    "Fail",
    "Pass",
    "Fail",
]


def main():
    """Main Function."""

    print("=" * 60)
    print(" Confusion Matrix ")
    print("=" * 60)

    matrix = confusion_matrix(
        y_actual,
        y_predicted,
        labels=["Pass", "Fail"],
    )

    print("\nConfusion Matrix")
    print("-" * 30)

    print(matrix)

    print("\nMeaning")
    print("-" * 30)

    print("Rows    : Actual Values")
    print("Columns : Predicted Values")

    print("\nLabel Order")
    print("-" * 30)

    print("[Pass, Fail]")

    print("\nObservation")
    print("-" * 30)

    print("Diagonal values are correct predictions.")

    print("Off-diagonal values are incorrect predictions.")

    print("Confusion Matrix explains model errors.")


if __name__ == "__main__":
    main()
    
#python3 13_confusion_matrix.py