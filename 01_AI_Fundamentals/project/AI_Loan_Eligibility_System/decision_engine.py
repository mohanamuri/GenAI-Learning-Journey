"""
=========================================================
Decision Engine

Contains the business rules for deciding
loan eligibility.

NOTE

This is Version 0.1

The decision is completely Rule-Based.

In Module-02 this file will be replaced
with a Machine Learning model.

The rest of the project will remain
unchanged.
=========================================================
"""

from constants import (
    APPROVED,
    REJECTED,
    MANUAL_REVIEW,
    MINIMUM_AGE,
    MINIMUM_SALARY,
    MINIMUM_EXPERIENCE,
    MINIMUM_CREDIT_SCORE,
    PERMANENT,
)


class LoanDecisionEngine:
    """
    Rule-Based Loan Decision Engine.
    """

    def evaluate(
        self,
        age,
        salary,
        experience,
        credit_score,
        employment,
    ):
        """
        Evaluates loan eligibility.

        Returns

        Decision
        Confidence
        Reason
        """

        reasons = []

        # -------------------------
        # Age
        # -------------------------

        if age < MINIMUM_AGE:
            reasons.append(
                "Minimum age requirement not met."
            )

        # -------------------------
        # Salary
        # -------------------------

        if salary < MINIMUM_SALARY:
            reasons.append(
                "Salary below eligibility criteria."
            )

        # -------------------------
        # Experience
        # -------------------------

        if experience < MINIMUM_EXPERIENCE:
            reasons.append(
                "Insufficient work experience."
            )

        # -------------------------
        # Credit Score
        # -------------------------

        if credit_score < MINIMUM_CREDIT_SCORE:
            reasons.append(
                "Low credit score."
            )

        # -------------------------
        # Employment
        # -------------------------

        if employment != PERMANENT:
            reasons.append(
                "Employment is not permanent."
            )

        # -------------------------
        # Decision
        # -------------------------

        if len(reasons) == 0:
            return (
                APPROVED,
                95,
                ["All eligibility criteria satisfied."],
            )

        if len(reasons) <= 2:
            return (
                MANUAL_REVIEW,
                75,
                reasons,
            )

        return (
            REJECTED,
            55,
            reasons,
        )