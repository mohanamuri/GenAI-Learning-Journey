"""
=========================================================
Validator Module

Validates all applicant inputs before
processing the loan request.
=========================================================
"""

from constants import (
    PERMANENT,
    CONTRACT,
    UNEMPLOYED,
)


class Validator:
    """
    Validates applicant information.
    """

    @staticmethod
    def validate_age(age: int) -> bool:
        return age > 0

    @staticmethod
    def validate_salary(salary: int) -> bool:
        return salary >= 0

    @staticmethod
    def validate_experience(experience: int) -> bool:
        return experience >= 0

    @staticmethod
    def validate_credit_score(score: int) -> bool:
        return 300 <= score <= 900

    @staticmethod
    def validate_employment(status: str) -> bool:
        valid_status = (
            PERMANENT,
            CONTRACT,
            UNEMPLOYED,
        )

        return status in valid_status

    @staticmethod
    def validate_all(
        age,
        salary,
        experience,
        score,
        employment,
    ) -> bool:

        return all(
            [
                Validator.validate_age(age),
                Validator.validate_salary(salary),
                Validator.validate_experience(experience),
                Validator.validate_credit_score(score),
                Validator.validate_employment(employment),
            ]
        )