"""
============================================================
Example 04 : Perceptron using TensorFlow

Module      : 03 - Deep Learning

Chapter     : 03 - Perceptron

Framework   : TensorFlow

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how to implement a simple
Perceptron using TensorFlow Tensors.

The mathematics is exactly the same as Example 03.

The only difference is that we now use TensorFlow
instead of normal Python variables.

Learning Outcomes
-----------------
✓ Create Input Tensors
✓ Create Weight Tensors
✓ Create Bias
✓ Perform Matrix Multiplication
✓ Calculate Perceptron Output

### Please note and think like,
1)Pure Python Mathematics = TensorFlow Mathematics
 The framework changed.The mathematics did not.
          
2) Neouron / Perception ==> Always Returns a Number. But Businees dont want NUmbers
                          They Want Decisions. So, we need to convert the number to Decision. This is done by Activation Function. 

4)Neouron / Perception --> Like A Calculator.
  Activation Function --> Like A Decision Maker.
============================================================
"""

# --------------------------------------------------------
# Step 1 : Import TensorFlow
# --------------------------------------------------------

import tensorflow as tf

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 04 : Perceptron using TensorFlow ")
print("=" * 60)

# --------------------------------------------------------
# Step 2 : Create Input Tensor
# --------------------------------------------------------
#
# Customer Details
#
# Salary        = 50000
# Experience    = 8
# Credit Score  = 750
#
# Shape = (1,3)
#
# One customer
# Three features
#
# --------------------------------------------------------

inputs = tf.constant(
    [[50000.0, 8.0, 750.0]]
)

print("\nInput Tensor")
print(inputs)

print("\nInput Shape")
print(inputs.shape)

# --------------------------------------------------------
# Step 3 : Create Weight Tensor
# --------------------------------------------------------
#
# Every input has one weight.
#
# Salary        -> 0.40
# Experience    -> 0.30
# Credit Score  -> 0.50
#
# Shape = (3,1)
#
# Three weights
# One output neuron
#
# --------------------------------------------------------

weights = tf.constant(
    [
        [0.40],
        [0.30],
        [0.50]
    ]
)

print("\nWeight Tensor")
print(weights)

print("\nWeight Shape")
print(weights.shape)

# --------------------------------------------------------
# Step 4 : Create Bias
# --------------------------------------------------------

bias = tf.constant(100.0)

print("\nBias")
print(bias)

# --------------------------------------------------------
# Step 5 : Matrix Multiplication
# --------------------------------------------------------
#
# Formula
#
# Output
#
# =
#
# (Inputs × Weights)
#
# +
#
# Bias
#
# --------------------------------------------------------

weighted_sum = tf.matmul(inputs, weights)

print("\nWeighted Sum")
print(weighted_sum)

# --------------------------------------------------------
# Step 6 : Add Bias
# --------------------------------------------------------

output = weighted_sum + bias

print("\nFinal Output")
print(output)

print("\nExample Completed Successfully.")

#python 04_perceptron.py