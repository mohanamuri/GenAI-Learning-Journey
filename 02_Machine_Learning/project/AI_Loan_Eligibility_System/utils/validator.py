"""
============================================================
Input Validation

============================================================
"""


def validate_inputs(
    age,
    salary,
    experience,
    credit_score,
    employment,
):
    """
    Validate user inputs.
    """

    if age < 18:
        raise ValueError(
            "Age must be at least 18."
        )

    if salary <= 0:
        raise ValueError(
            "Salary must be positive."
        )

    if experience < 0:
        raise ValueError(
            "Experience cannot be negative."
        )

    if not 300 <= credit_score <= 900:
        raise ValueError(
            "Credit Score must be between 300 and 900."
        )

    employment = employment.capitalize()

    valid = [
        "Permanent",
        "Contract",
        "Unemployed",
    ]

    if employment not in valid:
        raise ValueError(
            "Employment must be Permanent, Contract or Unemployed."
        )

    return employment