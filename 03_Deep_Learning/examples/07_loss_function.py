# Author: Mohan Raju Amuri
"""
============================================================
Example 07 : Loss Function

Module      : 03 - Deep Learning

Chapter     : 06 - Loss Function

Framework   : Pure Python + TensorFlow

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how a Neural Network measures
its prediction error using a Loss Function.

We will calculate the error manually and then compare it
with TensorFlow's implementation.

Learning Outcomes
-----------------
✓ Understand Prediction vs Actual Value
✓ Calculate Error Manually
✓ Calculate Mean Squared Error (MSE)
✓ Use TensorFlow Loss Functions
✓ Understand why Loss is important

============================================================
"""

# --------------------------------------------------------
# Step 1 : Import TensorFlow
# --------------------------------------------------------

import tensorflow as tf

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 07 : Loss Function ")
print("=" * 60)

# --------------------------------------------------------
# Step 2 : Actual and Predicted Values
# --------------------------------------------------------
#
# Example
#
# Actual Loan Eligibility Score
# Predicted Loan Eligibility Score
#
# --------------------------------------------------------

actual = 100.0
prediction = 85.0

print("\nActual Value")
print("-" * 40)
print(actual)

print("\nPredicted Value")
print("-" * 40)
print(prediction)

# --------------------------------------------------------
# Step 3 : Manual Error
# --------------------------------------------------------

error = actual - prediction

print("\nPrediction Error")
print("-" * 40)
print(error)

# --------------------------------------------------------
# Step 4 : Manual Mean Squared Error
# --------------------------------------------------------

manual_loss = error ** 2

print("\nManual Mean Squared Error")
print("-" * 40)
print(manual_loss)

# --------------------------------------------------------
# Step 5 : TensorFlow Mean Squared Error
# --------------------------------------------------------

y_true = tf.constant([actual], dtype=tf.float32)
y_pred = tf.constant([prediction], dtype=tf.float32)

mse = tf.keras.losses.MeanSquaredError()

tf_loss = mse(y_true, y_pred)

print("\nTensorFlow Mean Squared Error")
print("-" * 40)
print(tf_loss.numpy())

# --------------------------------------------------------
# Step 6 : Comparison
# --------------------------------------------------------

print("\nComparison")
print("-" * 40)

print(f"Manual Loss      : {manual_loss}")
print(f"TensorFlow Loss  : {tf_loss.numpy()}")

print("\nExample Completed Successfully.")

#python 07_loss_function.py