"""
=========================================================
Topic      : AI Fundamentals

Lesson     : AI Ethics

File       : 09_ai_ethics.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how an AI system should
provide explanations for its decisions.

Learning Objectives
-------------------
1. Understand Explainable AI (XAI).
2. Learn the importance of transparency.
3. Demonstrate ethical AI decisions.
=========================================================
"""

MIN_AGE = 18
MIN_SALARY = 30000


def evaluate_loan(age: int, salary: int) -> tuple[str, str]:
    """
    Evaluates loan eligibility and explains the reason.

    Args:
        age (int): Applicant age.
        salary (int): Monthly salary.

    Returns:
        tuple[str, str]:
            Decision and explanation.
    """

    if age < MIN_AGE:
        return (
            "Rejected",
            "Applicant does not meet the minimum age requirement."
        )

    if salary < MIN_SALARY:
        return (
            "Rejected",
            "Salary is below the minimum eligibility criteria."
        )

    return (
        "Approved",
        "Applicant satisfies all eligibility requirements."
    )


def main():
    """
    Main function.
    """

    print("=" * 55)
    print(" Explainable AI - Loan Decision ")
    print("=" * 55)

    age = int(input("\nEnter Applicant Age: ").strip())
    salary = int(input("Enter Monthly Salary: ").strip())

    decision, reason = evaluate_loan(age, salary)

    print("\nDecision")
    print("-" * 20)
    print(decision)

    print("\nReason")
    print("-" * 20)
    print(reason)

    print("\nAI Ethics Reminder")
    print("-" * 20)
    print("AI should explain important decisions clearly.")


if __name__ == "__main__":
    main()