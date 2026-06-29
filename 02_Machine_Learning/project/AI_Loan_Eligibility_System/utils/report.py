"""
============================================================
Loan Eligibility Report
============================================================
"""


def print_report(
    age,
    salary,
    experience,
    credit_score,
    employment,
    prediction,
    confidence,
):
    """
    Display Loan Report.
    """

    print()

    print("=" * 60)
    print(" AI Loan Eligibility Report ")
    print("=" * 60)

    print()

    print("Applicant Details")
    print("-" * 30)

    print(f"Age           : {age}")
    print(f"Salary        : ₹{salary:,}")
    print(f"Experience    : {experience} Years")
    print(f"Credit Score  : {credit_score}")
    print(f"Employment    : {employment}")

    print()

    print("Prediction")
    print("-" * 30)

    print(f"Decision      : {prediction}")
    print(f"Confidence    : {confidence:.2f}%")
    print("Technology    : Decision Tree Classifier")

    print()

    print("Observation")
    print("-" * 30)

    print("Prediction generated using")
    print("a trained Machine Learning model.")

    print("=" * 60)