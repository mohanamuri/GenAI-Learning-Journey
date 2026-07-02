"""
============================================================

Project : AI Loan Eligibility System

Module  : Deep Learning

File    : model_loader.py

Author  : Mohan Raju Amuri

Description
-----------
Loads the trained TensorFlow model.

============================================================
"""

import tensorflow as tf


def load_model(model_path):
    """
    Load trained Deep Learning model.
    """

    print("=" * 60)
    print("Loading Trained Model")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)

    print("Model Loaded Successfully.\n")

    return model