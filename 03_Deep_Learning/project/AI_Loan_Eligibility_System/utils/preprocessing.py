"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : preprocessing.py

Author  : Mohan Raju Amuri

Description
-----------
Contains reusable preprocessing functions.

============================================================
"""

import pandas as pd


def normalize_features(X: pd.DataFrame):
    """
    Normalize numerical features.

    Salary         : Divide by 100000
    Experience     : Divide by 20
    Credit Score   : Divide by 1000
    """

    print("=" * 60)
    print("Normalizing Dataset")
    print("=" * 60)

    X = X.copy()

    X["salary"] = X["salary"] / 100000
    X["experience"] = X["experience"] / 20
    X["credit_score"] = X["credit_score"] / 1000

    print("Normalization Completed Successfully.")

    return X