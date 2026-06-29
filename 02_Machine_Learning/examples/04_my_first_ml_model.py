"""
=========================================================
Topic      : Machine Learning

Lesson     : My First ML Model

File       : 04_my_first_ml_model.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Builds a tiny Machine Learning model using
pure Python.

This is NOT a real ML algorithm.

The goal is to understand what happens
inside fit(), predict() and score().
=========================================================
It's not a real ML algorithm yet, but it teaches the three most important ideas:
---------------------------------                -------------- 
Training (fit) ==> Learn from historical data.
Prediction (predict) ==>Apply what was learned to unseen data.
Evaluation (score) ==> Measure how well the model performed.

"""

# -------------------------------------------------------
# Training Dataset
# -------------------------------------------------------

X_train = [2, 3, 5, 6, 8]
y_train = ["Fail", "Fail", "Pass", "Pass", "Pass"]

# -------------------------------------------------------
# Testing Dataset
# -------------------------------------------------------

X_test = [4, 7, 9]
y_test = ["Fail", "Pass", "Pass"]


class MiniMLModel:
    """
    A tiny Machine Learning model.

    This is only for learning purposes.
    """

    def __init__(self):
        self.pass_threshold = None

    # -------------------------------------------------

    def fit(self, X, y):
        """
        Learns from historical data.

        Strategy:

        Find the minimum study hours
        where students passed.
        """

        passed_hours = []

        for hours, result in zip(X, y):

            if result == "Pass":
                passed_hours.append(hours)

        self.pass_threshold = min(passed_hours)

        print("\nLearning Completed")
        print("----------------------------")
        print(
            f"Discovered Rule : "
            f"Hours >= {self.pass_threshold}"
        )

    # -------------------------------------------------

    def predict(self, X):
        """
        Predicts results for new students.
        """

        predictions = []

        for hours in X:

            if hours >= self.pass_threshold:
                predictions.append("Pass")
            else:
                predictions.append("Fail")

        return predictions

    # -------------------------------------------------

    def score(self, predicted, actual):
        """
        Calculates prediction accuracy.
        """

        correct = 0

        for p, a in zip(predicted, actual):

            if p == a:
                correct += 1

        accuracy = (correct / len(actual)) * 100

        return accuracy


def main():

    print("=" * 60)
    print(" My First Machine Learning Model ")
    print("=" * 60)

    model = MiniMLModel()

    # ----------------------------------------

    print("\nStep 1 : Training")

    model.fit(X_train, y_train)

    # ----------------------------------------

    print("\nStep 2 : Prediction")

    predictions = model.predict(X_test)

    print("\nTesting Data")

    for hours, prediction in zip(X_test, predictions):

        print(
            f"Study Hours : {hours}"
            f" --> Prediction : {prediction}"
        )

    # ----------------------------------------

    print("\nStep 3 : Evaluation")

    accuracy = model.score(
        predictions,
        y_test,
    )

    print(f"\nActual Labels      : {y_test}")
    print(f"Predicted Labels   : {predictions}")

    print(f"\nAccuracy : {accuracy:.2f}%")

    print("\nObservation")
    print("----------------------------")

    print("fit()     -> Learns from data")

    print("predict() -> Predicts new data")

    print("score()   -> Evaluates accuracy")


if __name__ == "__main__":
    main()