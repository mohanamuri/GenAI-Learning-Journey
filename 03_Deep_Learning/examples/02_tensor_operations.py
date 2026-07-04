# Author: Mohan Raju Amuri
"""
=========================================================
Example 02 : Tensor Operations

Module      : 03 - Deep Learning

Framework   : TensorFlow

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates basic Tensor operations.

Learning Outcomes
-----------------
✓ Create Tensors
✓ Addition
✓ Subtraction
✓ Multiplication
✓ Matrix Multiplication
✓ Tensor Shape

=========================================================
"""

# --------------------------------------------------------
# Step 1 : Import TensorFlow
# --------------------------------------------------------

import tensorflow as tf

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 02 : Tensor Operations ")
print("=" * 60)

# --------------------------------------------------------
# Step 2 : Create Two Tensors
# --------------------------------------------------------

tensor1 = tf.constant([[1, 2],
                       [3, 4]])

tensor2 = tf.constant([[5, 6],
                       [7, 8]])

print("\nTensor 1")
print(tensor1)

print("\nTensor 2")
print(tensor2)

# --------------------------------------------------------
# Step 3 : Addition
# --------------------------------------------------------

print("\nAddition")
print(tensor1 + tensor2)

# --------------------------------------------------------
# Step 4 : Subtraction
# --------------------------------------------------------

print("\nSubtraction")
print(tensor2 - tensor1)

# --------------------------------------------------------
# Step 5 : Element-wise Multiplication
# --------------------------------------------------------

print("\nElement-wise Multiplication")
print(tensor1 * tensor2)

# --------------------------------------------------------
# Step 6 : Matrix Multiplication
# --------------------------------------------------------

print("\nMatrix Multiplication")
print(tf.matmul(tensor1, tensor2))

# --------------------------------------------------------
# Step 7 : Shape
# --------------------------------------------------------

print("\nShape of Tensor 1")
print(tensor1.shape)

print("\nShape of Tensor 2")
print(tensor2.shape)

print("\nExample Completed Successfully.")

#python 02_tensor_operations.py