"""
=========================================================
Topic      : Machine Learning

Lesson     : Overfitting vs Underfitting

File       : 12_overfitting_vs_underfitting.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates the concepts of underfitting,
good fitting and overfitting using simple
training and testing accuracy values.

Learning Objectives
-------------------
1. Understand underfitting.
2. Understand overfitting.
3. Learn what generalization means.
4. Interpret training vs testing accuracy.
=========================================================
"""


def evaluate_model(train_accuracy: int, test_accuracy: int) -> str:
    """
    Classifies the model based on training and testing accuracy.
    """

    if train_accuracy < 70 and test_accuracy < 70:
        return "Underfitting"

    if train_accuracy >= 90 and test_accuracy < 80:
        return "Overfitting"

    return "Good Fit"


def main():
    """Main Function."""

    print("=" * 60)
    print(" Overfitting vs Underfitting ")
    print("=" * 60)

    scenarios = [
        ("Model A", 55, 52),
        ("Model B", 93, 91),
        ("Model C", 99, 70),
    ]

    print()

    for name, train, test in scenarios:

        result = evaluate_model(train, test)

        print(f"{name}")
        print("-" * 25)
        print(f"Training Accuracy : {train}%")
        print(f"Testing Accuracy  : {test}%")
        print(f"Evaluation        : {result}")
        print()

    print("Observation")
    print("-" * 30)
    print("Underfitting : Learns too little.")
    print("Good Fit     : Learns useful patterns.")
    print("Overfitting  : Memorizes training data.")
    print("Goal         : Generalize to new data.")


if __name__ == "__main__":
    main()
    
#python3 12_overfitting_vs_underfitting.py