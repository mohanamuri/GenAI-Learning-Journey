# Perceptron

> **Module:** 03 - Deep Learning
>
> **Chapter:** 03
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Reading Time:** 20 Minutes
>
> **Framework:** Pure Python → TensorFlow
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What a Perceptron is
- Why it was invented
- How it makes decisions
- The role of Inputs, Weights and Bias
- The Perceptron equation
- How we implemented it using Python and TensorFlow
- Why every modern Neural Network starts with a Perceptron

---

# 📚 Prerequisites

Before reading this chapter, you should already understand:

- Tensor
- Tensor Operations
- Artificial Neural Networks

Examples completed:

- ✅ 01_first_tensor.py
- ✅ 02_tensor_operations.py
- ✅ 03_first_neuron.py

---

# 📖 Overview

A Neural Network is made up of many Artificial Neurons.

The simplest Artificial Neuron is called a **Perceptron**.

Think of the Perceptron as a tiny decision-making unit.

It receives several inputs, performs a mathematical calculation, and produces one output.

Although modern AI models contain billions of neurons, every one of them follows the same basic principle introduced by the Perceptron.

---

# ❓ Why Do We Need a Perceptron?

Imagine a bank wants to decide whether to approve a loan.

The bank looks at several pieces of information:

- Salary
- Experience
- Credit Score
- Existing Loans

The decision is based on all these values together.

Similarly, a Perceptron combines multiple inputs and produces one decision.

---

# 🏛 History

In **1958**, psychologist **Frank Rosenblatt** introduced the Perceptron.

It was the first mathematical model designed to imitate how a biological neuron makes decisions.

Although simple, the Perceptron became the foundation of modern Deep Learning.

---

# 🌍 Real-World Analogy

Imagine a loan officer.

```
Customer

↓

Salary

↓

Experience

↓

Credit Score

↓

Loan Officer

↓

Decision
```

The loan officer does not use only one factor.

Instead, they combine all available information before making a decision.

A Perceptron works in exactly the same way.

---

# 🧠 What is a Perceptron?

A Perceptron is the simplest Artificial Neuron.

It receives multiple inputs.

Each input has a weight.

The weighted values are combined.

A bias is added.

The result becomes the neuron's output.

---

# 🏗 Perceptron Architecture

```
                 Inputs

Salary -----------------------------

Experience -------------------------\

Credit Score ------------------------>  Perceptron  -----> Output

Employment -------------------------/

Age --------------------------------

```

Many Inputs

↓

One Output

---

# 🔄 Evolution of AI

```
Human Decision

↓

Business Rules

↓

Machine Learning

↓

Perceptron

↓

Neural Network

↓

Deep Learning

↓

Generative AI

↓

AI Agents
```

The Perceptron is the first building block in this evolution.

---

# 🧩 Components of a Perceptron

Every Perceptron contains four important components.

## 1️⃣ Inputs

Inputs are the information received by the neuron.

Example:

```
Salary

Experience

Credit Score
```

These values come from the dataset.

---

## 2️⃣ Weights

Every input has a weight.

Weights represent importance.

Example

```
Salary

↓

0.40

------------------------

Experience

↓

0.30

------------------------

Credit Score

↓

0.50
```

A larger weight means that input has a greater influence on the final decision.

---

## 3️⃣ Bias

Bias is an additional value added to the weighted sum.

Think of Bias as a fine-tuning adjustment.

Without Bias, the model becomes less flexible.

---

## 4️⃣ Output

After combining all weighted inputs and adding Bias, the Perceptron produces one output.

Example

```
20477.4
```

Later, an Activation Function converts this value into a useful prediction.

---

# 🧮 Mathematics

The Perceptron follows a very simple equation.

```
Output

=

(Input × Weight)

+

Bias
```

Expanded form

```
Output

=

(x₁ × w₁)

+

(x₂ × w₂)

+

(x₃ × w₃)

+

Bias
```

Where

| Symbol | Meaning |
|----------|----------|
| x | Input |
| w | Weight |
| b | Bias |
| y | Output |

---

# 🧠 Visual Representation

```
Inputs

↓

Multiply by Weights

↓

Add Bias

↓

Weighted Sum

↓

Output
```

In the next chapter, we'll add one more step.

```
Weighted Sum

↓

Activation Function

↓

Prediction
```

---

# 💻 Python Connection

In **Example 03**, we manually built a Perceptron.

```python
output = (
    (salary * weight_salary)
    + (experience * weight_experience)
    + (credit_score * weight_credit)
    + bias
)
```

This is the Perceptron equation written in Python.

No Deep Learning framework was required.

---

# ⚙ TensorFlow Connection

In **Example 04**, we implemented the same equation using TensorFlow.

```python
weighted_sum = tf.matmul(inputs, weights)

output = weighted_sum + bias
```

The mathematics did not change.

Only the implementation changed.

TensorFlow performs the calculations efficiently using Tensors.

---

# 🏦 Connection to Our AI Loan Eligibility System

```
Customer Details

↓

Salary

Experience

Credit Score

↓

Perceptron

↓

Weighted Sum

↓

Output

↓

(Next Chapter)

Activation Function

↓

Loan Approved / Rejected
```

This is exactly how our Deep Learning model begins.

---

# 💡 AI Engineer Notes

A Perceptron **does not understand** loans.

It only understands numbers.

Everything is converted into numerical values before the calculations begin.

Modern AI systems such as ChatGPT, Gemini and Claude perform the same type of mathematical operations, but on a much larger scale using millions or billions of neurons.

---

# ⚠ Common Mistakes

❌ A Perceptron is a complete Neural Network.

✔ A Perceptron is a single Artificial Neuron.

---

❌ Every input is equally important.

✔ Each input has its own weight.

---

❌ Bias is optional.

✔ Bias improves the flexibility of the model.

---

❌ TensorFlow changes the mathematics.

✔ TensorFlow only changes the implementation.

The mathematics remains the same.

---

# 🎤 Interview Questions

## Q1. What is a Perceptron?

A Perceptron is the simplest Artificial Neuron that performs weighted calculations to produce an output.

---

## Q2. Who invented the Perceptron?

Frank Rosenblatt in 1958.

---

## Q3. What are the four components of a Perceptron?

- Inputs
- Weights
- Bias
- Output

---

## Q4. Why are Weights important?

Weights determine the importance of each input.

---

## Q5. Why is Bias required?

Bias allows the model to shift its decision boundary and improve flexibility.

---

## Q6. Which TensorFlow function performed the weighted calculation?

```python
tf.matmul()
```

---

# 🧠 Interview in 30 Seconds

```
Definition

↓

First Artificial Neuron

Inventor

↓

Frank Rosenblatt

Year

↓

1958

Formula

↓

Output

=

Input × Weight + Bias

Purpose

↓

Weighted Decision

Framework Used

↓

Pure Python

↓

TensorFlow
```

---

# ⭐ Must Remember

✅ Perceptron = First Artificial Neuron

✅ Invented by Frank Rosenblatt

✅ Inputs have different Weights

✅ Bias fine-tunes the decision

✅ Formula = Input × Weight + Bias

✅ TensorFlow does not change the mathematics

✅ Every Neural Network starts with a Perceptron

---

# 📝 Summary

In this chapter you learned:

- Why the Perceptron was invented
- The components of a Perceptron
- The mathematical equation
- Python implementation
- TensorFlow implementation
- Connection to our AI Loan Eligibility System

You now understand the fundamental building block of every Neural Network.

---

# 📖 Related Examples

✅ Example 03 – First Artificial Neuron (Pure Python)

✅ Example 04 – Perceptron using TensorFlow

---

# ➡ Next Chapter

# Activation Functions

The Perceptron produces a numerical output.

But how does a computer convert that number into a meaningful decision like:

```
Approved

or

Rejected?
```

The answer is the **Activation Function**, which transforms mathematical output into intelligent predictions.