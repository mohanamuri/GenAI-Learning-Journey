"""
=========================================================
Topic      : Machine Learning

Lesson     : Cross Validation

File       : 15_cross_validation.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates K-Fold Cross Validation using
Scikit-Learn.

Learning Objectives
-------------------
1. Understand Cross Validation.
2. Evaluate model reliability.
3. Learn why one train-test split is not enough.
=========================================================
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# --------------------------------------------------
# Dataset
# --------------------------------------------------

X = [
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
]

y = [
    "Fail",
    "Fail",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
    "Pass",
]


def main():
    """Main Function."""

    print("=" * 60)
    print(" Cross Validation ")
    print("=" * 60)

    # --------------------------------------------------

    model = DecisionTreeClassifier(
        random_state=42
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
    )

    print("\nIndividual Fold Scores")
    print("-" * 30)

    for index, score in enumerate(scores, start=1):
        print(f"Fold {index} : {score:.2f}")

    print("\nAverage Accuracy")
    print("-" * 30)

    print(f"{scores.mean():.2f}")

    print("\nObservation")
    print("-" * 30)

    print("Cross Validation evaluates the model")
    print("multiple times using different splits.")

    print("Average accuracy is more reliable")

    print("than a single train-test split.")


if __name__ == "__main__":
    main()
    
#python3 15_cross_validation.py