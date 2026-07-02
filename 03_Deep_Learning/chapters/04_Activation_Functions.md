# Activation Functions

> **Module:** 03 - Deep Learning
>
> **Chapter:** 04
>
> **Difficulty:** ⭐⭐⭐☆☆
>
> **Reading Time:** 20 Minutes
>
> **Framework:** Pure Python → TensorFlow → Keras
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What an Activation Function is
- Why Activation Functions are required
- Why a Perceptron alone is not enough
- Different types of Activation Functions
- When to use ReLU, Sigmoid and Softmax
- How Activation Functions are used in TensorFlow and Keras

---

# 📚 Prerequisites

Before reading this chapter, you should already understand:

- Tensor
- Artificial Neural Networks
- Perceptron

Examples completed:

- ✅ 03_first_neuron.py
- ✅ 04_perceptron.py

---

# 📖 Overview

In the previous chapter, our Perceptron produced a numerical value.

Example:

```
20477.4
```

But businesses do not need numbers.

They need decisions.

```
Approved

Rejected
```

Something must convert mathematical output into a meaningful prediction.

That "something" is called an **Activation Function**.

---

# ❓ Why Do We Need Activation Functions?

Imagine a bank employee calculates a customer's score.

```
Customer Score

↓

20477.4
```

Does this automatically mean

```
Loan Approved?
```

No.

Someone must interpret the score.

Similarly,

a Perceptron calculates a value,

but an Activation Function decides what that value means.

---

# 🌍 Real-World Analogy

Think about school exams.

A student scores:

```
82 Marks
```

The school does not simply display

```
82
```

Instead it decides

```
Pass
```

or

```
Distinction
```

The marks are the calculation.

The result is the decision.

Activation Functions perform a similar role.

---

# 🔄 Evolution of Decision Making

```
Business Rule

↓

Step Function

↓

Sigmoid

↓

ReLU

↓

Softmax
```

As AI evolved,

Activation Functions became more powerful and flexible.

---

# 🧠 What is an Activation Function?

An Activation Function is a mathematical function applied after the neuron calculates its weighted sum.

Its job is to transform the neuron's output into a useful value.

Without Activation Functions,

a Neural Network becomes only a series of mathematical calculations.

---

# 🏗 Architecture

```
Inputs

↓

Weights

↓

Bias

↓

Weighted Sum

↓

Activation Function

↓

Prediction
```

This is the complete workflow of one Artificial Neuron.

---

# 🧮 Mathematics

The Perceptron calculates:

```
Weighted Sum

=

(Input × Weight)

+

Bias
```

Then

```
Prediction

=

Activation Function

(

Weighted Sum

)
```

Think of the Activation Function as the final decision maker.

---

# 📌 Types of Activation Functions

The most common Activation Functions are:

- Step Function
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- Softmax

We will focus on the most widely used ones.

---

# 1️⃣ Step Function

The oldest Activation Function.

Rule:

```
If Output > Threshold

↓

1

Else

↓

0
```

Useful for simple binary decisions.

---

# Example

```
Input

250

↓

1
```

```
Input

20

↓

0
```

---

# 2️⃣ Sigmoid Function

Sigmoid converts any value into a probability.

Output range:

```
0

↓

1
```

Example

```
0.92
```

Meaning

```
92% Probability
```

Commonly used for Binary Classification.

---

# 3️⃣ ReLU (Rectified Linear Unit)

ReLU is the most popular Activation Function used in modern Deep Learning.

Rule

```
If x > 0

↓

x

Else

↓

0
```

Example

```
Input

-10

↓

0
```

```
Input

25

↓

25
```

Simple.

Fast.

Efficient.

---

# 4️⃣ Softmax

Softmax is used for Multi-Class Classification.

Example

```
Dog

92%

Cat

6%

Horse

2%
```

All probabilities add up to

```
100%
```

---

# 📊 Comparison

| Activation Function | Output | Common Usage |
|---------------------|--------|--------------|
| Step | 0 or 1 | Basic Decision |
| Sigmoid | 0 to 1 | Binary Classification |
| ReLU | 0 to ∞ | Hidden Layers |
| Softmax | Probability Distribution | Multi-Class Classification |

---

# 💻 Python Connection

In **Example 05**, we implemented:

Business Rule

```python
if output > 20000:
    print("Approved")
```

Step Function

```python
step = 1 if output > threshold else 0
```

Sigmoid

```python
sigmoid = 1 / (1 + math.exp(-x))
```

ReLU

```python
relu = max(0, x)
```

Each example demonstrates a different way of converting calculations into decisions.

---

# ⚙ TensorFlow / Keras Connection

Later, we will use built-in Activation Functions.

Example

```python
Dense(
    64,
    activation="relu"
)
```

Output Layer

```python
Dense(
    1,
    activation="sigmoid"
)
```

Keras performs the same mathematical functions automatically.

---

# 🏦 Connection to Our AI Loan Eligibility System

```
Customer Details

↓

Perceptron

↓

Weighted Sum

↓

Activation Function

↓

Loan Approved

or

Loan Rejected
```

Without an Activation Function,

our model cannot produce a meaningful business prediction.

---

# 💡 AI Engineer Notes

Modern Neural Networks almost always use:

```
Hidden Layers

↓

ReLU
```

Binary Classification

```
↓

Sigmoid
```

Multi-Class Classification

```
↓

Softmax
```

Choosing the correct Activation Function is an important design decision.

---

# ⚠ Common Mistakes

❌ Activation Functions are optional.

✔ Every practical Neural Network uses Activation Functions.

---

❌ Sigmoid is used everywhere.

✔ Today ReLU is the standard choice for Hidden Layers.

---

❌ ReLU produces probabilities.

✔ ReLU simply removes negative values.

---

❌ Softmax is used for Binary Classification.

✔ Softmax is used for Multi-Class Classification.

---

# 🎤 Interview Questions

## Q1. What is an Activation Function?

A mathematical function that converts a neuron's output into a useful prediction.

---

## Q2. Why is an Activation Function required?

To introduce non-linearity and enable meaningful decisions.

---

## Q3. Which Activation Function is commonly used in Hidden Layers?

ReLU.

---

## Q4. Which Activation Function is used for Binary Classification?

Sigmoid.

---

## Q5. Which Activation Function is used for Multi-Class Classification?

Softmax.

---

## Q6. Why is ReLU so popular?

Because it is simple, computationally efficient and helps Deep Neural Networks train effectively.

---

# 🧠 Interview in 30 Seconds

```
Purpose

↓

Decision Making

Most Popular

↓

ReLU

Binary Classification

↓

Sigmoid

Multi-Class

↓

Softmax

Remember

Neuron Calculates

↓

Activation Decides
```

---

# ⭐ Must Remember

✅ Activation Function converts calculations into decisions.

✅ ReLU is used in Hidden Layers.

✅ Sigmoid is used for Binary Classification.

✅ Softmax is used for Multi-Class Classification.

✅ Every modern Neural Network uses Activation Functions.

---

# 📝 Summary

In this chapter you learned:

- Why Activation Functions exist
- The limitations of a Perceptron
- Step Function
- Sigmoid
- ReLU
- Softmax
- Where each function is used
- How they connect to TensorFlow and Keras

You are now ready to understand how multiple neurons work together inside a Neural Network.

---

# 📖 Related Example

✅ Example 05 – Activation Functions

---

# ➡ Next Chapter

# Forward Propagation

Until now,

we have worked with a single Artificial Neuron.

But a real Neural Network contains many neurons working together.

How does information travel through all these neurons?

The answer is **Forward Propagation**.