"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : train.py

Author  : Mohan Raju Amuri

Description
-----------
Training entry point.

============================================================
"""

from config.config import (
    DATASET_PATH,
    MODEL_PATH,
    EPOCHS,
    BATCH_SIZE
)

from data.dataset import (
    load_dataset,
    preprocess_dataset
)

from utils.preprocessing import normalize_features

from models.train_model import (
    build_model,
    train_model,
    save_model
)


def main():

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    dataset = load_dataset(DATASET_PATH)

    # ---------------------------------------------
    # Split Dataset
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = preprocess_dataset(dataset)

    # ---------------------------------------------
    # Normalize
    # ---------------------------------------------

    X_train = normalize_features(X_train)

    X_test = normalize_features(X_test)

    # ---------------------------------------------
    # Build Model
    # ---------------------------------------------

    model = build_model()

    # ---------------------------------------------
    # Train Model
    # ---------------------------------------------

    train_model(
        model,
        X_train,
        y_train,
        EPOCHS,
        BATCH_SIZE
    )

    # ---------------------------------------------
    # Save Model
    # ---------------------------------------------

    save_model(
        model,
        MODEL_PATH
    )


if __name__ == "__main__":
    main()