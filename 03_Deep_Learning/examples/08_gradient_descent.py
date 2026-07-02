"""
============================================================
Example 08 : Gradient Descent

Module      : 03 - Deep Learning

Chapter     : 07 - Gradient Descent

Framework   : Pure Python

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates the basic idea of Gradient
Descent using a simple weight update process.

The objective is to reduce the prediction error by
updating the model's weight step by step.

Learning Outcomes
-----------------
✓ Understand Gradient Descent
✓ Learn Weight Updates
✓ Learn Learning Rate
✓ Observe Loss Reduction
✓ Prepare for TensorFlow Optimizers

============================================================
"""

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 08 : Gradient Descent ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Initial Values
# --------------------------------------------------------

actual = 100

weight = 0.50

learning_rate = 0.05

print("\nInitial Weight")
print("-" * 40)
print(weight)

# --------------------------------------------------------
# Step 2 : Gradient Descent Loop
# --------------------------------------------------------

print("\nTraining Progress")
print("-" * 70)

print(f"{'Epoch':<8}{'Prediction':<15}{'Loss':<15}{'Weight'}")

for epoch in range(1, 11):

    # Prediction
    prediction = actual * weight

    # Error
    error = actual - prediction

    # Mean Squared Error
    loss = error ** 2

    print(f"{epoch:<8}{prediction:<15.2f}{loss:<15.2f}{weight:.4f}")

    # ----------------------------------------------------
    # Simple Weight Update
    #
    # Increase the weight slightly when the
    # prediction is lower than the actual value.
    #
    # This is NOT TensorFlow's real Gradient Descent.
    # It is only a simple simulation to understand
    # the concept.
    # ----------------------------------------------------

    weight = weight + learning_rate * (error / actual)

print("\nFinal Weight")
print("-" * 40)
print(round(weight,4))

print("\nExample Completed Successfully.")

#python 08_gradient_descent.py