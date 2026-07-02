"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : dataset.py

Author  : Mohan Raju Amuri

Description
-----------
Loads and preprocesses the dataset.

============================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path):
    """
    Load dataset from CSV file.
    """

    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = pd.read_csv(dataset_path)

    print(f"Total Records : {len(dataset)}")

    return dataset


def preprocess_dataset(dataset):
    """
    Split dataset into Features and Target.
    """

    print("\nPreprocessing Dataset")
    print("-" * 60)

    X = dataset.drop("approved", axis=1)

    y = dataset["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training Records : {len(X_train)}")
    print(f"Testing Records  : {len(X_test)}")

    return X_train, X_test, y_train, y_test