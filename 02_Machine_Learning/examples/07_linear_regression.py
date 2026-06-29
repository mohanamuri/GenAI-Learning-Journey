"""
=========================================================
Topic      : Machine Learning

Lesson     : Linear Regression

File       : 07_linear_regression.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates Linear Regression using Scikit-Learn
to predict employee salary based on years of
experience.

Learning Objectives
-------------------
1. Understand Regression problems.
2. Train a Linear Regression model.
3. Predict continuous values.
4. Evaluate model performance.
=========================================================
"""

from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# Training Dataset
# --------------------------------------------------

X_train = [
    [1],
    [2],
    [3],
    [4],
    [5],
]

y_train = [
    4,
    6,
    8,
    10,
    12,
]

# --------------------------------------------------
# Testing Dataset
# --------------------------------------------------

X_test = [
    [6],
    [7],
    [8],
]

y_test = [
    14,
    16,
    18,
]


def main():
    """Main Function."""

    print("=" * 60)
    print(" Linear Regression ")
    print("=" * 60)

    # --------------------------------------------------

    print("\nStep 1 : Create Model")

    model = LinearRegression()

    print(model)

    # --------------------------------------------------

    print("\nStep 2 : Train Model")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed successfully.")

    # --------------------------------------------------

    print("\nStep 3 : Predict Salary")

    predictions = model.predict(X_test)

    print()

    for experience, prediction, actual in zip(
        X_test,
        predictions,
        y_test,
    ):

        print(
            f"Experience : {experience[0]} Years"
            f" | Predicted Salary : ₹{prediction:.2f} LPA"
            f" | Actual Salary : ₹{actual:.2f} LPA"
        )

    # --------------------------------------------------

    print("\nStep 4 : Evaluate Model")

    accuracy = model.score(
        X_test,
        y_test,
    )

    print(f"\nR² Score : {accuracy:.2f}")

    # --------------------------------------------------

    print("\nModel Details")
    print("-" * 30)

    print(f"Intercept : {model.intercept_:.2f}")

    print(f"Coefficient : {model.coef_[0]:.2f}")

    # --------------------------------------------------

    print("\nObservation")
    print("-" * 30)

    print("Linear Regression predicts continuous values.")

    print("The model learned a linear relationship.")

    print("Higher experience generally leads to higher salary.")

    print("R² Score indicates how well the model fits the data.")


if __name__ == "__main__":
    main()
    

#1)Why is the output 14.00 instead of just 14?
#Ans: Because Linear Regression predicts floating-point (decimal) values, not integers.

#2)What does the R² Score indicate?
#Ans: The R² Score indicates how well the model fits the data. A score closer  to 1 means a better fit, while a score closer to 0 means a poor fit.

#4)Why are coef_ and intercept_ useful?
#Ans: coef_ represents the weight of each feature in the prediction, while 
# intercept_ is the baseline value. 
# Together, they define the linear equation used for predictions.

#python3 07_linear_regression.py

