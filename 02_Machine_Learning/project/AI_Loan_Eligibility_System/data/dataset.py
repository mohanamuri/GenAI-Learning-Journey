"""
============================================================
Project : AI Loan Eligibility System (Machine Learning)

File    : dataset.py

Description
-----------
Loads and prepares the loan dataset.

============================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.constants import DATASET_FILE


def load_dataset():
    """
    Load dataset from CSV.

    Returns
    -------
    DataFrame
    """

    dataframe = pd.read_csv(DATASET_FILE)

    return dataframe


def prepare_dataset(dataframe):
    """
    Converts categorical values into numbers.

    Returns

    X
    y
    encoder
    """

    encoder = LabelEncoder()

    dataframe["Employment"] = encoder.fit_transform(
        dataframe["Employment"]
    )

    dataframe["LoanApproved"] = dataframe[
        "LoanApproved"
    ].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    X = dataframe.drop(
        columns=["LoanApproved"]
    )

    y = dataframe["LoanApproved"]

    return X, y, encoder