"""
=========================================================
Topic      : Machine Learning

Lesson     : Train-Test Split

File       : 03_train_test_split.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how a dataset is manually split
into training and testing datasets.

Learning Objectives
-------------------
1. Understand X_train and y_train.
2. Understand X_test and y_test.
3. Learn why testing data should be unseen.
=========================================================
"""

# ---------------------------------------------------
# Historical Dataset
# ---------------------------------------------------

students = [
    {"hours": 2, "result": "Fail"},
    {"hours": 3, "result": "Fail"},
    {"hours": 4, "result": "Fail"},
    {"hours": 5, "result": "Pass"},
    {"hours": 6, "result": "Pass"},
    {"hours": 7, "result": "Pass"},
    {"hours": 8, "result": "Pass"},
    {"hours": 9, "result": "Pass"},
]


def prepare_dataset(data):
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


def manual_train_test_split(X, y):
    """
    Splits the dataset manually.

    Training = First 75%
    Testing = Last 25%
    """

    split_index = int(len(X) * 0.75)

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


def main():

    print("=" * 60)
    print(" Manual Train-Test Split ")
    print("=" * 60)

    X, y = prepare_dataset(students)

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y)

    print("\nOriginal Features (X)")
    print("-" * 30)
    print(X)

    print("\nOriginal Labels (y)")
    print("-" * 30)
    print(y)

    print("\nTraining Features (X_train)")
    print("-" * 30)
    print(X_train)

    print("\nTraining Labels (y_train)")
    print("-" * 30)
    print(y_train)

    print("\nTesting Features (X_test)")
    print("-" * 30)
    print(X_test)

    print("\nTesting Labels (y_test)")
    print("-" * 30)
    print(y_test)

    print("\nObservation")
    print("-" * 30)

    print("Training data is used for learning.")
    print("Testing data is used for evaluation.")


if __name__ == "__main__":
    main()