"""
============================================================
Project : AI Loan Eligibility System (Machine Learning)

File    : loan_model.py

Description
-----------
Creates and trains the Machine Learning model.

============================================================
"""

import joblib

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    train_test_split,
)

from sklearn.metrics import accuracy_score

from config.constants import (
    MODEL_FILE,
    RANDOM_STATE,
    TEST_SIZE,
)

from data.dataset import (
    load_dataset,
    prepare_dataset,
)


def train_model():
    """
    Train Decision Tree model.

    Returns
    -------

    accuracy
    """

    dataframe = load_dataset()

    X, y, encoder = prepare_dataset(
        dataframe
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
        },
        MODEL_FILE,
    )

    return accuracy