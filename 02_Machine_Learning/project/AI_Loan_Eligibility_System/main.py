"""
============================================================
AI Loan Eligibility System

Machine Learning Version

============================================================
"""

from models.predictor import LoanEligibilityModel

from utils.validator import validate_inputs

from utils.report import print_report


def main():

    print("=" * 60)
    print(" AI Loan Eligibility System ")
    print("=" * 60)

    print()

    age = int(input("Age : "))

    salary = int(input("Monthly Salary : "))

    experience = int(
        input("Experience (Years): ")
    )

    credit_score = int(
        input("Credit Score : ")
    )

    employment = input(
        "Employment (Permanent/Contract/Unemployed): "
    )

    employment = validate_inputs(
        age,
        salary,
        experience,
        credit_score,
        employment,
    )

    print()

    print("Loading Machine Learning Model...")

    model = LoanEligibilityModel()

    print("✓ Model Loaded")

    print()

    print("Analyzing Applicant...")

    prediction, confidence = model.predict(
        age,
        salary,
        experience,
        credit_score,
        employment,
    )

    print("✓ Prediction Completed")

    print_report(
        age,
        salary,
        experience,
        credit_score,
        employment,
        prediction,
        confidence,
    )


if __name__ == "__main__":
    main()