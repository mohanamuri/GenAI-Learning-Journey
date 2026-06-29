"""
=========================================================
Topic      : Machine Learning

Lesson     : Features and Labels

File       : 02_first_dataset.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Introduces Features (X) and Labels (y)
using a simple student dataset.

Learning Objectives
-------------------
1. Understand Features.
2. Understand Labels.
3. Learn how ML datasets are organized.
=========================================================
"""

# ---------------------------------------------
# Historical Dataset
# ---------------------------------------------

students = [
    {"hours": 2, "result": "Fail"},
    {"hours": 3, "result": "Fail"},
    {"hours": 5, "result": "Pass"},
    {"hours": 6, "result": "Pass"},
    {"hours": 8, "result": "Pass"},
]


def separate_features_labels(data):
    """
    Separates Features (X)
    and Labels (y).
    """

    X = []
    y = []

    for student in data:
        X.append(student["hours"])
        y.append(student["result"])

    return X, y


def main():

    print("=" * 60)
    print(" Features and Labels ")
    print("=" * 60)

    X, y = separate_features_labels(students)

    print("\nOriginal Dataset")
    print("-" * 30)

    for student in students:
        print(student)

    print("\nFeatures (X)")
    print("-" * 30)
    print(X)

    print("\nLabels (y)")
    print("-" * 30)
    print(y)

    print("\nObservation")
    print("-" * 30)

    print("Features are the INPUTS.")

    print("Labels are the EXPECTED OUTPUTS.")


if __name__ == "__main__":
    main()