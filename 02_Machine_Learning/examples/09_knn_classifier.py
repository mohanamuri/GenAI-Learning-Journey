"""
=========================================================
Topic      : Machine Learning

Lesson     : K-Nearest Neighbors (KNN)

File       : 09_knn_classifier.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates K-Nearest Neighbors (KNN)
classification using Scikit-Learn.

Learning Objectives
-------------------
1. Understand KNN.
2. Train a KNN classifier.
3. Predict student results.
4. Evaluate model accuracy.
=========================================================
"""

from sklearn.neighbors import KNeighborsClassifier

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
    print(" K-Nearest Neighbors (KNN) ")
    print("=" * 60)

    # --------------------------------------------------

    print("\nStep 1 : Create Model")

    model = KNeighborsClassifier(
        n_neighbors=3
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

    print("\nObservation")
    print("-" * 30)

    print("KNN predicts using nearby data points.")

    print("No decision tree or equation is built.")

    print("Prediction depends on nearest neighbors.")

    print("K = 3 means 3 nearest neighbors are used.")


if __name__ == "__main__":
    main()


#python3 09_knn_classifier.py