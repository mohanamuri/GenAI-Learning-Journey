"""
=========================================================
Topic      : Machine Learning

Lesson     : Random Forest

File       : 10_random_forest.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates Random Forest Classification
using Scikit-Learn.

Learning Objectives
-------------------
1. Understand Ensemble Learning.
2. Train a Random Forest model.
3. Predict student results.
4. Compare with Decision Tree.
=========================================================
"""

from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Training Dataset
# --------------------------------------------------

X_train = [
    [2],
    [3],
    [5],
    [6],
    [8],
]

y_train = [
    "Fail",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
]

# --------------------------------------------------
# Testing Dataset
# --------------------------------------------------

X_test = [
    [1],
    [4],
    [7],
    [9],
]

y_test = [
    "Fail",
    "Fail",
    "Pass",
    "Pass",
]


def main():
    """Main Function."""

    print("=" * 60)
    print(" Random Forest ")
    print("=" * 60)

    # --------------------------------------------------

    print("\nStep 1 : Create Model")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    print(model)

    # --------------------------------------------------

    print("\nStep 2 : Train Model")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed successfully.")

    # --------------------------------------------------

    print("\nStep 3 : Predict")

    predictions = model.predict(X_test)

    print()

    for hours, prediction, actual in zip(
        X_test,
        predictions,
        y_test,
    ):

        print(
            f"Hours : {hours[0]:>2}"
            f" | Predicted : {prediction:<5}"
            f" | Actual : {actual}"
        )

    # --------------------------------------------------

    print("\nStep 4 : Evaluate")

    accuracy = model.score(
        X_test,
        y_test,
    )

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    # --------------------------------------------------

    print("\nModel Information")
    print("-" * 30)

    print(f"Number of Trees : {model.n_estimators}")

    # --------------------------------------------------

    print("\nObservation")
    print("-" * 30)

    print("Random Forest combines many Decision Trees.")

    print("Each tree votes for the final prediction.")

    print("More trees usually improve stability.")

    print("Random Forest reduces overfitting.")


if __name__ == "__main__":
    main()


#python3 10_random_forest.py