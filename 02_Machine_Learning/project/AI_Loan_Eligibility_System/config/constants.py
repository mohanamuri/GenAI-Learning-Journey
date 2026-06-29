"""
============================================================
Project : AI Loan Eligibility System (Machine Learning)

File    : constants.py

Description
-----------
Stores project-wide constants.

============================================================
"""

from pathlib import Path

# ----------------------------------------------------------
# Project Paths
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models"

# ----------------------------------------------------------
# Dataset
# ----------------------------------------------------------

DATASET_FILE = DATA_DIR / "loan_dataset.csv"

# ----------------------------------------------------------
# Trained Model
# ----------------------------------------------------------

MODEL_FILE = MODEL_DIR / "loan_eligibility_model.pkl"

# ----------------------------------------------------------
# Random Seed
# ----------------------------------------------------------

RANDOM_STATE = 42

# ----------------------------------------------------------
# Test Size
# ----------------------------------------------------------

TEST_SIZE = 0.20