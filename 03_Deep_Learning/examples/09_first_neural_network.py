# Author: Mohan Raju Amuri
"""
============================================================
Example 09 : First Neural Network

Module      : 03 - Deep Learning

Chapter     : 09 - Building First Neural Network

Framework   : TensorFlow + Keras

Author      : Mohan Raju Amuri

Description
-----------
This example builds the first real Neural Network
using TensorFlow and Keras.

Architecture
------------

Input Layer (3 Features)

        ↓

Hidden Layer (4 Neurons, ReLU)

        ↓

Output Layer (1 Neuron, Sigmoid)

Learning Outcomes
-----------------
✓ Create Sequential Model
✓ Add Dense Layers
✓ Compile Model
✓ Understand Model Summary
✓ Prepare for Model Training

============================================================
"""

import tensorflow as tf

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 09 : First Neural Network ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Create Model
# --------------------------------------------------------

model = tf.keras.Sequential()

print("\nStep 1 : Empty Sequential Model Created")

# --------------------------------------------------------
# Step 2 : Hidden Layer
# --------------------------------------------------------

model.add(
    tf.keras.layers.Dense(
        units=4,
        activation="relu",
        input_shape=(3,)
    )
)

print("Step 2 : Hidden Layer Added")

# --------------------------------------------------------
# Step 3 : Output Layer
# --------------------------------------------------------

model.add(
    tf.keras.layers.Dense(
        units=1,
        activation="sigmoid"
    )
)

print("Step 3 : Output Layer Added")

# --------------------------------------------------------
# Step 4 : Compile Model
# --------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Step 4 : Model Compiled")

# --------------------------------------------------------
# Step 5 : Display Model Summary
# --------------------------------------------------------

print("\nModel Summary")
print("-" * 60)

model.summary()

print("\nExample Completed Successfully.")

#python 09_first_neural_network.py
