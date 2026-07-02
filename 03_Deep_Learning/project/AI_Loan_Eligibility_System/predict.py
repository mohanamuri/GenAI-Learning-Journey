"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : predict.py

Author  : Mohan Raju Amuri

Description
-----------
Prediction Entry Point.

============================================================
"""

from config.config import MODEL_PATH

from models.model_loader import load_model

from models.predictor import predict


def main():

    model = load_model(MODEL_PATH)

    salary = 85000

    experience = 9

    credit_score = 780

    probability, status = predict(

        model,

        salary,

        experience,

        credit_score

    )

    print("=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Salary        : {salary}")

    print(f"Experience    : {experience}")

    print(f"Credit Score  : {credit_score}")

    print()

    print(f"Probability   : {probability:.4f}")

    print(f"Loan Status   : {status}")


if __name__ == "__main__":
    main()