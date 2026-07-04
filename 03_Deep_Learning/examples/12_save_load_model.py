# Author: Mohan Raju Amuri
"""
============================================================
Example 12 : Save and Load Model

Module      : 03 - Deep Learning

Chapter     : 12 - Save and Load Model

Framework   : TensorFlow + Keras

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how to:

1. Train a Neural Network
2. Save the trained model
3. Load the saved model
4. Make predictions using the loaded model

Learning Outcomes
-----------------
✓ Save Keras Models
✓ Load Saved Models
✓ Reuse Models
✓ Understand Production Workflow

============================================================
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 12 : Save and Load Model ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Training Data
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
# Step 2 : Build Model
# --------------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# --------------------------------------------------------
# Step 3 : Compile
# --------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------------
# Step 4 : Train
# --------------------------------------------------------

print("\nTraining Model...")
print("-" * 40)

model.fit(
    X_train,
    y_train,
    epochs=30,
    verbose=0
)

print("Training Completed.")

# --------------------------------------------------------
# Step 5 : Save Model
# --------------------------------------------------------

MODEL_PATH = "loan_eligibility_model.keras"

model.save(MODEL_PATH)

print("\nModel Saved Successfully")
print("-" * 40)
print(MODEL_PATH)

# --------------------------------------------------------
# Step 6 : Load Model
# --------------------------------------------------------

loaded_model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel Loaded Successfully")

# --------------------------------------------------------
# Step 7 : Prediction
# --------------------------------------------------------

customer = np.array([
    [65000,9,770]
], dtype=np.float32)

customer[:,0] /= 100000
customer[:,1] /= 20
customer[:,2] /= 1000

prediction = loaded_model.predict(
    customer,
    verbose=0
)

print("\nPrediction")
print("-" * 40)
print(prediction)

if prediction[0][0] >= 0.5:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")

print("\nExample Completed Successfully.")
#python 12_save_load_model.py