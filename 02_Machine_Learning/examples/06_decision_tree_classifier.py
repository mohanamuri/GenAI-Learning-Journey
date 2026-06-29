"""
=========================================================
Topic      : Machine Learning

Lesson     : Decision Tree Classifier

File       : 06_decision_tree_classifier.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how to train and evaluate a
Decision Tree Classifier using Scikit-Learn.

Learning Objectives
-------------------
1. Train a Decision Tree model.
2. Predict unseen data.
3. Evaluate model accuracy.
4. Compare with Rule-Based and MiniMLModel.
=========================================================
"""

from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------
# Training Dataset
# ----------------------------------------------------

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

# ----------------------------------------------------
# Testing Dataset
# ----------------------------------------------------

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

    print("=" * 60)
    print(" Decision Tree Classifier ")
    print("=" * 60)

    # -----------------------------------------

    print("\nStep 1 : Create Model")

    model = DecisionTreeClassifier(
        random_state=42
    )
    #random_state makes the algorithm produce the same result every time.

    print(model)

    # -----------------------------------------

    print("\nStep 2 : Train Model")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed successfully.")

    # -----------------------------------------

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

    # -----------------------------------------

    print("\nStep 4 : Evaluate")

    accuracy = model.score(
        X_test,
        y_test,
    )

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    # -----------------------------------------

    print("\nObservation")
    print("-" * 30)

    print("Decision Tree learned from historical data.")

    print("No business rules were manually written.")

    print("The model discovered its own decision logic.")

    print("Scikit-Learn hides the algorithm complexity.")


if __name__ == "__main__":
    main()
    
    
#python3 06_decision_tree_classifier.py