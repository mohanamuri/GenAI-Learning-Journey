"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : predictor.py

Author  : Mohan Raju Amuri

Description
-----------
Prediction Engine.

============================================================
"""

import pandas as pd

from utils.preprocessing import normalize_features


def predict(model,
            salary,
            experience,
            credit_score):
    """
    Predict Loan Eligibility
    """

    customer = pd.DataFrame({

        "salary": [salary],

        "experience": [experience],

        "credit_score": [credit_score]

    })

    customer = normalize_features(customer)

    probability = model.predict(
        customer,
        verbose=0
    )[0][0]

    status = "Approved"

    if probability < 0.50:
        status = "Rejected"

    return probability, status