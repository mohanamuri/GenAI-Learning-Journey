"""
============================================================
Example 06 : Forward Propagation

Module      : 03 - Deep Learning

Chapter     : 05 - Forward Propagation

Framework   : TensorFlow

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how data flows through a
simple Neural Network using Forward Propagation.

Network Architecture

Input Layer (3 Features)
        ↓
Hidden Layer (2 Neurons)
        ↓
Output Layer (1 Neuron)

Learning Outcomes
-----------------
✓ Understand Forward Propagation
✓ Learn Hidden Layers
✓ Matrix Multiplication
✓ Activation Functions
✓ Final Prediction

============================================================
"""

import tensorflow as tf

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 06 : Forward Propagation ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Customer Input
# --------------------------------------------------------
#
# Features
# Salary
# Experience
# Credit Score
#
# Shape = (1,3)
# --------------------------------------------------------

inputs = tf.constant([[50000.0, 8.0, 750.0]])

print("\nInput Layer")
print("-" * 40)
print(inputs)

# --------------------------------------------------------
# Step 2 : Hidden Layer Weights
# --------------------------------------------------------
#
# Shape = (3,2)
#
# 3 Inputs
# 2 Hidden Neurons
#
# --------------------------------------------------------

hidden_weights = tf.constant([
    [0.40, 0.20],
    [0.30, 0.60],
    [0.50, 0.10]
])

hidden_bias = tf.constant([[100.0, 50.0]])

print("\nHidden Layer Weights")
print("-" * 40)
print(hidden_weights)

# --------------------------------------------------------
# Step 3 : Hidden Layer Calculation
# --------------------------------------------------------

hidden_output = tf.matmul(inputs, hidden_weights)

hidden_output = hidden_output + hidden_bias

# --------------------------------------------------------
# Step 4 : ReLU Activation
# --------------------------------------------------------

hidden_output = tf.nn.relu(hidden_output)

print("\nHidden Layer Output")
print("-" * 40)
print(hidden_output)

# --------------------------------------------------------
# Step 5 : Output Layer Weights
# --------------------------------------------------------
#
# Shape = (2,1)
#
# 2 Hidden Neurons
# 1 Output Neuron
#
# --------------------------------------------------------

output_weights = tf.constant([
    [0.70],
    [0.40]
])

output_bias = tf.constant([[25.0]])

print("\nOutput Layer Weights")
print("-" * 40)
print(output_weights)

# --------------------------------------------------------
# Step 6 : Output Layer Calculation
# --------------------------------------------------------

final_output = tf.matmul(hidden_output, output_weights)

final_output = final_output + output_bias

# --------------------------------------------------------
# Step 7 : Sigmoid Activation
# --------------------------------------------------------

prediction = tf.nn.sigmoid(final_output)

print("\nPrediction")
print("-" * 40)
print(prediction)

# --------------------------------------------------------
# Step 8 : Business Decision
# --------------------------------------------------------

print("\nBusiness Decision")
print("-" * 40)

if prediction.numpy()[0][0] > 0.5:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")

print("\nExample Completed Successfully.")

#python 06_forward_propagation.py