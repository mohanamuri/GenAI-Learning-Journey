"""
=========================================================
Example 01

First Tensor 
Tensor is the basic building block of Deep Learning.

Module : 03 - Deep Learning

Author : Mohan Raju Amuri
=========================================================
"""

import tensorflow as tf

print("=" * 60)
print(" Deep Learning - First Tensor ")
print("=" * 60)

# Create a tensor
#Meaning: TensorFlow created a Tensor object.
#Tensor is the basic building block of Deep Learning.
tensor = tf.constant([10, 20, 30, 40, 50])

print("\nTensor:")
print(tensor)

print("\nTensor Type:")
print(type(tensor))

#shape=(5,) meaning One Dimension --> 5 Values
print("\nTensor Shape:")
print(tensor.shape)

print("\nTensor Data Type:")
print(tensor.dtype)

print("\nTensor converted to NumPy:")
print(tensor.numpy())

print("\nExample Completed Successfully")

# --------------------------------------------------------
# Tensor Output Explanation
# --------------------------------------------------------
#
# Example Output:
#
# tf.Tensor([10 20 30 40 50], shape=(5,), dtype=int32)
#
# Breakdown:
#
# tf.Tensor
# ----------
# Indicates that this object is a TensorFlow Tensor.
# A Tensor is TensorFlow's primary data structure used
# to store and process numerical data.
#
# [10 20 30 40 50]
# ----------------
# These are the actual values stored inside the Tensor.
#
# shape=(5,)
# ----------
# Shape describes the structure of the Tensor.
#
# (5,)
#  ↓
# One-dimensional Tensor (Vector)
# containing 5 elements.
#
# Examples:
#
# (5,)      -> 1D Tensor with 5 values
# (2,3)     -> 2D Tensor with 2 rows and 3 columns
# (3,4,2)   -> 3D Tensor
#
# dtype=int32
# -----------
# Data Type of every element.
#
# int32
#   ↓
# 32-bit Integer
#
# Other common TensorFlow data types:
#
# int64
# float32
# float64
# bool
# string
#
# TensorFlow performs mathematical operations based
# on the Tensor's data type.
#
# Visual Representation:
#
# Tensor
#
# 10   20   30   40   50
#
# Shape : (5,)
# Rank  : 1
#
# Everything in Deep Learning eventually becomes a Tensor.
#
# Images
# Text
# Audio
# Video
#
# are all converted into Tensors before being processed
# by a Neural Network.
#
# --------------------------------------------------------

#python 01_first_tensor.py