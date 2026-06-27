"""
=========================================================
AI Loan Eligibility System

Version : 0.1

Rule-Based AI

Author : Mohan Raju

Repository :
GenAI-Learning-Journey
=========================================================
"""

from validator import Validator
from decision_engine import LoanDecisionEngine
from report import Report


def get_applicant() -> dict:
    """
    Reads applicant details.
    """

    print("=" * 60)
    print(" AI Loan Eligibility System ")
    print("=" * 60)

    applicant = {
        "age": int(input("\nAge : ")),
        "salary": int(input("Monthly Salary : ")),
        "experience": int(input("Experience (Years): ")),
        "credit_score": int(input("Credit Score : ")),
        "employment": input(
            "Employment "
            "(Permanent/Contract/Unemployed): "
        ).strip(),
    }

    return applicant


def main():

    applicant = get_applicant()

    valid = Validator.validate_all(
        applicant["age"],
        applicant["salary"],
        applicant["experience"],
        applicant["credit_score"],
        applicant["employment"],
    )

    if not valid:
        print("\nInvalid Input.")
        return

    engine = LoanDecisionEngine()

    decision, confidence, reasons = engine.evaluate(
        applicant["age"],
        applicant["salary"],
        applicant["experience"],
        applicant["credit_score"],
        applicant["employment"],
    )

    Report.display(
        applicant,
        decision,
        confidence,
        reasons,
    )


if __name__ == "__main__":
    main()