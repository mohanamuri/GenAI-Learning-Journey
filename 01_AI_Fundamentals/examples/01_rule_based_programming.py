"""
=========================================================
Topic      : AI Fundamentals
Lesson     : Rule-Based Programming
File       : 01_rule_based_programming.py

Author     : Mohan Raju

Description
-----------
Demonstrates how traditional rule-based programming
works using a simple age verification example.

Learning Objectives
-------------------
1. Understand rule-based programming.
2. Compare traditional programming with AI.
3. Write clean and readable Python code.
=========================================================
"""


def is_adult(age: int) -> bool:
    """
    Determines whether a person is an adult.

    Args:
        age (int): Person's age.

    Returns:
        bool: True if age is 18 or above, otherwise False.
    """
    return age >= 18


def main():
    """Main function."""

    age = int(input("Enter age: "))

    if is_adult(age):
        print("\n✅ Adult")
    else:
        print("\n❌ Minor")


if __name__ == "__main__":
    main()