"""
=========================================================
Topic      : Machine Learning

Lesson     : Logistic Regression

File       : 08_logistic_regression.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates Logistic Regression using
Scikit-Learn to classify students as
Pass or Fail based on study hours.

Learning Objectives
-------------------
1. Understand Logistic Regression.
2. Train a classification model.
3. Predict categories.
4. Evaluate classification accuracy.
=========================================================
"""

from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Training Dataset
# --------------------------------------------------

X_train = [
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
]

y_train = [
    "Fail",
    "Fail",
    "Fail",
    "Pass",
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
    [6],
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
    print(" Logistic Regression ")
    print("=" * 60)

    # --------------------------------------------------

    print("\nStep 1 : Create Model")

    model = LogisticRegression()

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

    print("\nObservation")
    print("-" * 30)

    print("Logistic Regression is a Classification algorithm.")

    print("It predicts categories using probability.")

    print("Common use cases include spam detection,")

    print("fraud detection and loan approval.")


if __name__ == "__main__":
    main()
    
#python3 08_logistic_regression.py