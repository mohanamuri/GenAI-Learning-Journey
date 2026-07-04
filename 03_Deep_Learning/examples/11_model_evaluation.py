# Author: Mohan Raju Amuri
"""
============================================================
Example 11 : Model Evaluation

Module      : 03 - Deep Learning

Chapter     : 11 - Model Evaluation

Framework   : TensorFlow + Keras

Author      : Mohan Raju Amuri

Description
-----------
This example trains a Neural Network and evaluates
its performance using Accuracy.

Learning Outcomes
-----------------
✓ Train a Neural Network
✓ Evaluate Model
✓ Understand Accuracy
✓ Generate Predictions
✓ Compare Prediction vs Actual

============================================================
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 11 : Model Evaluation ")
print("=" * 60)

# --------------------------------------------------------
# Training Data
# --------------------------------------------------------

X_train = np.array([
    [50000,8,750],
    [30000,3,600],
    [80000,12,820],
    [25000,2,500],
    [70000,10,780],
    [20000,1,450],
    [90000,15,850],
    [40000,5,650]
], dtype=np.float32)

y_train = np.array([
    1,0,1,0,1,0,1,0
], dtype=np.float32)

# Normalize

X_train[:,0] /= 100000
X_train[:,1] /= 20
X_train[:,2] /= 1000

# --------------------------------------------------------
# Build Model
# --------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------------
# Train
# --------------------------------------------------------

model.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=0
)

print("\nModel Trained Successfully")

# --------------------------------------------------------
# Evaluate
# --------------------------------------------------------

loss, accuracy = model.evaluate(
    X_train,
    y_train,
    verbose=0
)

print("\nEvaluation Results")
print("-" * 40)

print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy:.4f}")

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------

predictions = model.predict(
    X_train,
    verbose=0
)

print("\nPrediction Comparison")
print("-" * 60)

print(f"{'Actual':<10}{'Predicted Probability'}")

for actual, pred in zip(y_train, predictions):

    print(
        f"{int(actual):<10}{pred[0]:.4f}"
    )

print("\nExample Completed Successfully.")
#python 11_model_evaluation.py