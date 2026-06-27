"""
=========================================================
Report Module

Displays the final decision in a professional format.
=========================================================
"""


class Report:
    """
    Displays loan evaluation results.
    """

    @staticmethod
    def display(
        applicant: dict,
        decision: str,
        confidence: int,
        reasons: list,
    ) -> None:

        print("\n" + "=" * 60)
        print(" AI Loan Eligibility Report ")
        print("=" * 60)

        print("\nApplicant Details")
        print("-" * 30)

        print(f"Age           : {applicant['age']}")
        print(f"Salary        : ₹{applicant['salary']:,}")
        print(f"Experience    : {applicant['experience']} Years")
        print(f"Credit Score  : {applicant['credit_score']}")
        print(f"Employment    : {applicant['employment']}")

        print("\nDecision")
        print("-" * 30)

        print(decision)

        print("\nConfidence")
        print("-" * 30)

        print(f"{confidence}%")

        print("\nReasons")
        print("-" * 30)

        for reason in reasons:
            print(f"• {reason}")

        print("\nObservation")
        print("-" * 30)

        print(
            "Current version uses Rule-Based AI.\n"
            "Later modules will replace only the\n"
            "Decision Engine with Machine Learning."
        )

        print("=" * 60)