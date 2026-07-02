"""
============================================================
Example 10 : Model Training

Module      : 03 - Deep Learning

Chapter     : 10 - Model Training

Framework   : TensorFlow + Keras

Author      : Mohan Raju Amuri

Description
-----------
This example trains a simple Neural Network using
sample loan eligibility data.

Learning Outcomes
-----------------
✓ Build a Neural Network
✓ Compile the Model
✓ Train using model.fit()
✓ Observe Loss and Accuracy
✓ Understand Epochs

============================================================
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 10 : Model Training ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Training Dataset
# --------------------------------------------------------
#
# Features
# Salary
# Experience
# Credit Score
#
# Label
# 1 = Approved
# 0 = Rejected
#
# --------------------------------------------------------

X = np.array([
    [50000, 8, 750],
    [30000, 3, 600],
    [80000, 12, 820],
    [25000, 2, 500],
    [70000, 10, 780],
    [20000, 1, 450],
    [90000, 15, 850],
    [40000, 5, 650]
], dtype=np.float32)

y = np.array([
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
], dtype=np.float32)

print("\nTraining Samples")
print("-" * 40)
print(len(X))

# --------------------------------------------------------
# Step 2 : Normalize Data
# --------------------------------------------------------

X[:, 0] = X[:, 0] / 100000.0
X[:, 1] = X[:, 1] / 20.0
X[:, 2] = X[:, 2] / 1000.0

print("\nNormalized Training Data")
print("-" * 40)
print(X)

# --------------------------------------------------------
# Step 3 : Build Model
# --------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# --------------------------------------------------------
# Step 4 : Compile Model
# --------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Compiled Successfully")

# --------------------------------------------------------
# Step 5 : Train Model
# --------------------------------------------------------

print("\nTraining Started...")
print("-" * 40)

history = model.fit(
    X,
    y,
    epochs=20,
    verbose=1
)

print("\nTraining Completed.")

# --------------------------------------------------------
# Step 6 : Model Summary
# --------------------------------------------------------

print("\nModel Summary")
print("-" * 40)

model.summary()

# --------------------------------------------------------
# Step 7 : Make Prediction
# --------------------------------------------------------

new_customer = np.array([
    [65000, 9, 770]
], dtype=np.float32)

new_customer[:,0] /= 100000.0
new_customer[:,1] /= 20.0
new_customer[:,2] /= 1000.0

prediction = model.predict(new_customer, verbose=0)

print("\nPrediction")
print("-" * 40)
print(prediction)

if prediction[0][0] >= 0.5:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")

print("\nExample Completed Successfully.")

#python 10_model_training.py