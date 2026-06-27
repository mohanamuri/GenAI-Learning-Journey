"""
=========================================================
Topic      : AI Fundamentals

Lesson     : Pattern Matching

File       : 02_pattern_matching.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Demonstrates how business rules can change
based on different conditions such as country.

Learning Objectives
-------------------
1. Understand conditional rule matching.
2. Learn how business rules vary by region.
3. Prepare for AI-based decision making.
=========================================================
"""

# --------------------------------------------------------
# Constants
# --------------------------------------------------------

USA = 1
INDIA = 2

VOTING_AGE = {
    USA: 21,
    INDIA: 18
}


def is_adult(country: int, age: int) -> bool:
    """
    Determines whether a person is an adult
    based on the country's voting age.

    Args:
        country (int):
            Country code.
            1 = USA
            2 = India

        age (int):
            Age of the person.

    Returns:
        bool:
            True if the person satisfies the
            voting age for the selected country.
            Otherwise False.
    """

    voting_age = VOTING_AGE.get(country)

    if voting_age is None:
        return False

    return age >= voting_age


def main():
    """
    Main function.
    Accepts user input and displays
    whether the person is an adult.
    """

    print("\nSelect Country")
    print("1. USA")
    print("2. India")

    country = int(input("\nEnter country (1 or 2): ").strip())
    age = int(input("Enter age: ").strip())

    if country not in VOTING_AGE:
        print("\n❌ Invalid Country")
        return

    if is_adult(country, age):
        print("\n✅ Adult")
    else:
        print("\n❌ Minor")


if __name__ == "__main__":
    main()