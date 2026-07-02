"""
============================================================
Example 03 : First Artificial Neuron

Module      : 03 - Deep Learning

Chapter     : 03 - FirstNeuron (Perceptron) using Pure Python

Framework   : Pure Python

Author      : Mohan Raju Amuri

Description
-----------
This example demonstrates how a single Artificial
Neuron works using simple mathematics.

The neuron receives multiple inputs, multiplies
each input by its corresponding weight, adds a bias,
and produces an output.

Learning Outcomes
-----------------
✓ Understand Inputs
✓ Understand Weights
✓ Understand Bias
✓ Calculate Neuron Output
✓ Prepare for Perceptron and TensorFlow

============================================================
"""

print("=" * 60)
print(" Module 03 - Deep Learning ")
print(" Example 03 : First Artificial Neuron ")
print("=" * 60)

# --------------------------------------------------------
# Step 1 : Input Values
# --------------------------------------------------------

salary = 50000
experience = 8
credit_score = 750

# --------------------------------------------------------
# Step 2 : Assign Weights
# --------------------------------------------------------

weight_salary = 0.40
weight_experience = 0.30
weight_credit = 0.50

# --------------------------------------------------------
# Step 3 : Bias
# --------------------------------------------------------

bias = 100

# --------------------------------------------------------
# Step 4 : Weighted Sum
# --------------------------------------------------------

output = (
    (salary * weight_salary)
    + (experience * weight_experience)
    + (credit_score * weight_credit)
    + bias
)

# --------------------------------------------------------
# Display Inputs
# --------------------------------------------------------

print("\nInput Values")
print("-" * 40)
print(f"Salary        : {salary}")
print(f"Experience    : {experience}")
print(f"Credit Score  : {credit_score}")

# --------------------------------------------------------
# Display Weights
# --------------------------------------------------------

print("\nWeights")
print("-" * 40)
print(f"Salary Weight      : {weight_salary}")
print(f"Experience Weight  : {weight_experience}")
print(f"Credit Weight      : {weight_credit}")

# --------------------------------------------------------
# Display Bias
# --------------------------------------------------------

print("\nBias")
print("-" * 40)
print(bias)

# --------------------------------------------------------
# Display Formula
# --------------------------------------------------------

print("\nNeuron Formula")
print("-" * 40)
print("Output = (Input × Weight) + Bias")

# --------------------------------------------------------
# Display Calculation
# --------------------------------------------------------

print("\nNeuron Output")
print("-" * 40)
print(output)

print("\nExample Completed Successfully.")

#python 03_first_neuron.py