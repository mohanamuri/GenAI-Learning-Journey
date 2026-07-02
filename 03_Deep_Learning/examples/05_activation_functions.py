"""
============================================================
Example 05 : Activation Functions

Module      : 03 - Deep Learning

Chapter     : 04 - Activation Functions

Framework   : Pure Python

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how Activation Functions
convert a neuron's numerical output into a meaningful
decision.

We will implement:

1. Business Rule
2. Step Function
3. Sigmoid Function
4. ReLU Function

Learning Outcomes
-----------------
✓ Understand why Activation Functions are needed
✓ Learn Step Function
✓ Learn Sigmoid Function
✓ Learn ReLU Function
✓ Compare different outputs

============================================================
"""

import math

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 05 : Activation Functions ")
print("=" * 60)

# --------------------------------------------------------
# Neuron Output
# --------------------------------------------------------

output = 20477.4

print("\nNeuron Output")
print("-" * 40)
print(output)

# ========================================================
# Part 1 : Business Rule
# ========================================================

print("\n1. Business Rule")
print("-" * 40)

if output > 20000:
    print("Loan Status : Approved")
else:
    print("Loan Status : Rejected")

# ========================================================
# Part 2 : Step Function
# This is the first activation function.
# ========================================================

print("\n2. Step Function")
print("-" * 40)

threshold = 20000

step_output = 1 if output > threshold else 0

print("Threshold :", threshold)
print("Output    :", step_output)

# ========================================================
# Part 3 : Sigmoid Function
# Converts any value into a probability.
# Sigmoid - is used for binary classification
# ========================================================

print("\n3. Sigmoid Function")
print("-" * 40)

# Using a smaller value because Sigmoid overflows
# with very large numbers like 20477.4

x = 2

sigmoid = 1 / (1 + math.exp(-x))

# math.exp() --> Calculates the exponential value (eˣ), used in the Sigmoid function

print("Input      :", x)
print("Probability:", round(sigmoid, 4))

# ========================================================
# Part 4 : ReLU Functionv (Rectified Linear Unit)
# ReLu - activation function is commonly used in hidden layers
# ========================================================

print("\n4. ReLU Function")
print("-" * 40)

relu_input = -5

relu = max(0, relu_input)

print("Input  :", relu_input)
print("Output :", relu)

relu_input = 15

relu = max(0, relu_input)

#ReLU (Rectified Linear Unit) is an activation function 
# That results the input directly if it is positive; 
#        otherwise(if negative), it will results zero.

print("\nInput  :", relu_input)
print("Output :", relu)

print("\nExample Completed Successfully.")

#python 05_activation_functions.py