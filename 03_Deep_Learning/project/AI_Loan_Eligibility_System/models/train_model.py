"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : train_model.py

Author  : Mohan Raju Amuri

Description
-----------
Creates, trains and saves the Deep Learning model.

============================================================
"""

import tensorflow as tf


def build_model():
    """
    Build Neural Network Architecture
    """

    print("=" * 60)
    print("Building Neural Network")
    print("=" * 60)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(3,)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Model Created Successfully.\n")

    return model


def train_model(model,
                X_train,
                y_train,
                epochs,
                batch_size):
    """
    Train the Neural Network
    """

    print("=" * 60)
    print("Training Started")
    print("=" * 60)

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    print("\nTraining Completed Successfully.\n")

    return history


def save_model(model, model_path):
    """
    Save trained model
    """

    model.save(model_path)

    print("=" * 60)
    print("Model Saved Successfully")
    print("=" * 60)

    print(model_path)