"""
=========================================================
Topic      : AI Fundamentals

Project    : AI Loan Eligibility System

File       : constants.py

Author     : Mohan Raju

Repository : GenAI-Learning-Journey

Description
-----------
Stores all configurable values used by the
Loan Eligibility System.

Why?
----
Keeping constants in one place makes the
application easier to maintain.
=========================================================
"""

# -----------------------------
# Eligibility Criteria
# -----------------------------

MINIMUM_AGE = 18
MINIMUM_SALARY = 30000
MINIMUM_EXPERIENCE = 2
MINIMUM_CREDIT_SCORE = 700

# -----------------------------
# Employment Types
# -----------------------------

PERMANENT = "Permanent"
CONTRACT = "Contract"
UNEMPLOYED = "Unemployed"

# -----------------------------
# Decision Labels
# -----------------------------

APPROVED = "Approved"
REJECTED = "Rejected"
MANUAL_REVIEW = "Manual Review"