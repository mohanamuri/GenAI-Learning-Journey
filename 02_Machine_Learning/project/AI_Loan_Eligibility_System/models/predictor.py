"""
============================================================
Project : AI Loan Eligibility System (Machine Learning)

File    : predictor.py

Description
-----------
Loads the trained model and predicts
loan eligibility.

============================================================
"""

import joblib
import pandas as pd

from config.constants import MODEL_FILE


class LoanEligibilityModel:
    """
    Machine Learning Loan Prediction Model.
    """

    def __init__(self):
        """
        Load the trained model and label encoder.
        """

        saved_model = joblib.load(MODEL_FILE)

        self.model = saved_model["model"]
        self.encoder = saved_model["encoder"]

    def predict(
        self,
        age,
        salary,
        experience,
        credit_score,
        employment,
    ):
        """
        Predict loan eligibility.

        Parameters
        ----------
        age : int
        salary : int
        experience : int
        credit_score : int
        employment : str

        Returns
        -------
        tuple
            (prediction, confidence)
        """

        # Encode employment type
        employment = self.encoder.transform(
            [employment]
        )[0]

        # Create DataFrame with the same feature names
        # used during model training
        features = pd.DataFrame(
            [[
                age,
                salary,
                experience,
                credit_score,
                employment,
            ]],
            columns=[
                "Age",
                "Salary",
                "Experience",
                "CreditScore",
                "Employment",
            ],
        )

        # Predict
        prediction = self.model.predict(features)[0]

        # Prediction probability
        probability = self.model.predict_proba(features)[0]

        confidence = max(probability) * 100

        # Convert numeric prediction to text
        result = (
            "Approved"
            if prediction == 1
            else "Rejected"
        )

        return result, confidence