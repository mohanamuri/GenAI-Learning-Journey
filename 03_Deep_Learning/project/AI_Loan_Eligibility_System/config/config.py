"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : config.py

Author  : Mohan Raju Amuri

Description
-----------
Application level configuration.

============================================================
"""

from pathlib import Path

# --------------------------------------------------------
# Project Directories
# --------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

LOG_DIR = BASE_DIR / "logs"

# --------------------------------------------------------
# Model Configuration
# --------------------------------------------------------

MODEL_NAME = "loan_eligibility_model.keras"

MODEL_PATH = MODEL_DIR / MODEL_NAME

# --------------------------------------------------------
# Dataset Configuration
# --------------------------------------------------------

DATASET_NAME = "loan_dataset.csv"

DATASET_PATH = DATA_DIR / DATASET_NAME

# --------------------------------------------------------
# Training Configuration
# --------------------------------------------------------

RANDOM_SEED = 42

EPOCHS = 30

BATCH_SIZE = 16

LEARNING_RATE = 0.001