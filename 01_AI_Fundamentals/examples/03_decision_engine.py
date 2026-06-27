"""
=========================================================
Topic      : AI Fundamentals

Lesson     : Decision Engine

File       : 03_decision_engine.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how a traditional rule-based
Decision Engine works using multiple business
rules.

Learning Objectives
-------------------
1. Understand how a Decision Engine works.
2. Learn how multiple rules produce one decision.
3. Understand why traditional systems cannot learn.
4. Prepare for AI-based decision making.
=========================================================
"""

# --------------------------------------------------------
# Constants
# --------------------------------------------------------

USA = 1
INDIA = 2

COUNTRIES = {
    USA: "USA",
    INDIA: "India"
}

VOTING_AGE = {
    USA: 21,
    INDIA: 18
}

MIN_SALARY = 30000


def is_country_valid(country: int) -> bool:
    """
    Checks whether the selected country is valid.

    Args:
        country (int): Country code.

    Returns:
        bool: True if country exists.
    """
    return country in COUNTRIES


def is_adult(country: int, age: int) -> bool:
    """
    Checks whether a person is an adult
    based on country-specific voting age.

    Args:
        country (int): Country code.
        age (int): Person's age.

    Returns:
        bool: True if eligible as an adult.
    """
    return age >= VOTING_AGE[country]


def has_minimum_salary(salary: int) -> bool:
    """
    Checks whether the person meets
    the minimum salary requirement.

    Args:
        salary (int): Monthly salary.

    Returns:
        bool: True if salary meets requirement.
    """
    return salary >= MIN_SALARY


def is_credit_card_eligible(country: int, age: int, salary: int) -> bool:
    """
    Decision Engine.

    Business Rules:
        1. Country must be valid.
        2. Person must be an adult.
        3. Salary must be at least 30,000.

    Args:
        country (int): Country code.
        age (int): Person's age.
        salary (int): Monthly salary.

    Returns:
        bool: True if eligible.
    """

    if not is_country_valid(country):
        return False

    if not is_adult(country, age):
        return False

    if not has_minimum_salary(salary):
        return False

    return True


def main():
    """
    Main function.
    """

    print("\n==============================")
    print(" Credit Card Decision Engine ")
    print("==============================")

    print("\nSelect Country")
    print("1. USA")
    print("2. India")

    country = int(input("\nEnter Country: ").strip())
    age = int(input("Enter Age: ").strip())
    salary = int(input("Enter Monthly Salary: ").strip())

    print("\n------------------------------")

    if not is_country_valid(country):
        print("❌ Invalid Country")
        return

    print(f"Country : {COUNTRIES[country]}")
    print(f"Age     : {age}")
    print(f"Salary  : ₹{salary:,}")

    print("\nDecision")

    if is_credit_card_eligible(country, age, salary):
        print("✅ Credit Card Approved")
    else:
        print("❌ Credit Card Rejected")


if __name__ == "__main__":
    main()