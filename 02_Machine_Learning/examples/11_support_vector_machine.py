"""
=========================================================
Topic      : Machine Learning

Lesson     : Support Vector Machine (SVM)

File       : 11_support_vector_machine.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates Support Vector Machine (SVM)
classification using Scikit-Learn.

Learning Objectives
-------------------
1. Understand SVM.
2. Train an SVM classifier.
3. Predict student results.
4. Evaluate classification accuracy.
=========================================================
"""

from sklearn.svm import SVC

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
    print(" Support Vector Machine (SVM) ")
    print("=" * 60)

    # --------------------------------------------------

    print("\nStep 1 : Create Model")

    model = SVC()

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

    print("SVM finds the best boundary between classes.")

    print("It maximizes the margin between categories.")

    #print("Works well on small and medium datasets.")
    print("Different algorithms may produce different predictions.")
    print("Always evaluate a model before selecting it.")

    print("Commonly used for classification tasks.")


if __name__ == "__main__":
    main()
    
#python3 11_support_vector_machine.py