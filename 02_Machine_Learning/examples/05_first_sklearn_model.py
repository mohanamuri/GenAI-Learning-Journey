"""
=========================================================
Topic      : Machine Learning

Lesson     : First Scikit-Learn Model

File       : 05_first_sklearn_model.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Builds the first real Machine Learning model
using Scikit-Learn.

Learning Objectives
-------------------
1. Learn Scikit-Learn workflow.
2. Understand fit(), predict(), score().
3. Compare with our MiniMLModel.
=========================================================
"""

from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# Training Dataset
# --------------------------------------------------

#Why is X_train written as [[2], [3], [5]] instead of [2, 3, 5]?
#Ans:Because Scikit-Learn expects - 2D Matrix
#Each row represents one sample.Each column represents one feature.
#Each row is one student.Each column is one feature.
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
    [4],
    [7],
    [9],
]

y_test = [
    "Fail",
    "Pass",
    "Pass",
]


def main():

    print("=" * 60)
    print(" First Scikit-Learn Model ")
    print("=" * 60)

    # --------------------------------------

    print("\nStep 1 : Create Model")

    model = DecisionTreeClassifier()

    print(model)

    # --------------------------------------

    print("\nStep 2 : Training")

    model.fit(
        X_train,
        y_train,
    )

    print("Training Completed.")

    # --------------------------------------

    print("\nStep 3 : Prediction")

    predictions = model.predict(X_test)

    for hours, prediction in zip(X_test, predictions):

        print(
            f"Hours : {hours[0]}"
            f" --> {prediction}"
        )

    # --------------------------------------

    print("\nStep 4 : Evaluation")

    accuracy = model.score(
        X_test,
        y_test,
    )

    print(f"\nAccuracy : {accuracy * 100:.2f}%")

    print("\nObservation")
    print("-" * 30)

    print("DecisionTreeClassifier is an Estimator.")

    print("fit() learns from historical data.")

    print("predict() predicts unseen data.")

    print("score() evaluates the model.")


if __name__ == "__main__":
    main()
    
    
#python3 02_Machine_Learning/examples/05_first_sklearn_model.py