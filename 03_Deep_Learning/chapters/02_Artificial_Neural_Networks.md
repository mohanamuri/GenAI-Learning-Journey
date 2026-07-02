# Artificial Neural Networks (ANN)

> **Module:** 03 - Deep Learning
>
> **Reading Time:** 20 Minutes
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What an Artificial Neural Network (ANN) is.
- Why Neural Networks were invented.
- How ANNs are inspired by the human brain.
- The basic structure of a Neural Network.
- Why ANNs are the foundation of Deep Learning.

---

# 📖 History & Motivation

Let's go back to our Machine Learning module.

We solved problems like:

- Loan Approval
- House Price Prediction
- Customer Churn

These worked well because the data was structured.

Now imagine a different problem.

```text
Can a computer recognize this?

        🐶
```

Traditional Machine Learning struggled.

Why?

Because engineers had to manually define features.

```text
Long Nose

↓

Fur

↓

Tail

↓

Four Legs

↓

Prediction
```

Researchers asked:

> What if a computer could automatically discover these features?

This led to **Artificial Neural Networks**.

---

# 🧠 Why is it called a Neural Network?

The idea comes from the **human brain**.

Our brain contains billions of neurons.

Each neuron receives information,

processes it,

and passes it to other neurons.

A Neural Network follows the same idea.

---

# Human Brain

```text
Eyes

↓

Brain Neurons

↓

Decision

↓

Dog
```

---

# Artificial Neural Network

```text
Image

↓

Artificial Neurons

↓

Prediction

↓

Dog
```

Notice that this is only **inspiration**.

Artificial Neural Networks are **not copies of the human brain**.

They are simplified mathematical models.

---

# What is an Artificial Neural Network?

An Artificial Neural Network (ANN) is a collection of interconnected artificial neurons.

Each neuron receives information,

performs a mathematical calculation,

and passes the result to the next neuron.

Many neurons working together learn complex patterns.

---

# Basic Structure

```text
Input Layer

↓

Hidden Layer

↓

Hidden Layer

↓

Output Layer
```

This is the basic architecture of almost every Neural Network.

---

# Input Layer

The Input Layer receives the data.

Example

Loan Eligibility

```text
Age

Salary

Experience

Credit Score

Employment
```

These become the network inputs.

The Input Layer does **not** make decisions.

It simply passes information forward.

---

# Hidden Layers

The Hidden Layers perform the learning.

Example

```text
Input

↓

Neuron

↓

Neuron

↓

Neuron

↓

Output
```

This is where patterns are discovered.

The more hidden layers,

the more complex patterns the model can learn.

---

# Output Layer

The Output Layer produces the final prediction.

Examples

Classification

```text
Approved

Rejected
```

Regression

```text
Salary = ₹15 LPA
```

Image Classification

```text
Dog
```

Everything ends here.

---

# Complete Neural Network

```text
Age

Salary

Experience

Credit Score

Employment

        │

        ▼

Input Layer

        │

        ▼

Hidden Layer

        │

        ▼

Hidden Layer

        │

        ▼

Output Layer

        │

        ▼

Approved
```

This is exactly how our Deep Learning Loan Eligibility System will work.

---

# Why Hidden Layers Matter

Imagine recognizing a face.

Layer 1 may learn:

```text
Edges
```

Layer 2 may learn:

```text
Eyes

Nose
```

Layer 3 may learn:

```text
Complete Face
```

Each layer builds on the previous one.

This automatic feature learning is the strength of Deep Learning.

---

# ANN vs Machine Learning

Machine Learning

```text
Data

↓

Human selects features

↓

Algorithm

↓

Prediction
```

ANN

```text
Data

↓

Neural Network

↓

Learns Features

↓

Prediction
```

No manual feature engineering is required.

---

# Real-World Applications

Artificial Neural Networks are used in:

- Face Recognition
- Voice Recognition
- ChatGPT
- Google Translate
- Self-Driving Cars
- Medical Diagnosis
- Recommendation Systems

Almost every modern AI application starts with Neural Networks.

---

# 🧠 Think Like an AI Engineer

Do not think of an ANN as "magic."

Think of it as:

```text
Many Small Decision Makers

↓

Working Together

↓

Learning Patterns

↓

Making Predictions
```

Each neuron performs a tiny task.

Together,

they solve complex problems.

---

# 💼 AI Engineer Note

The Neural Network itself is only the beginning.

In the coming chapters,

we will learn:

- Neuron
- Perceptron
- Activation Function
- Forward Propagation
- Backpropagation
- Gradient Descent

These are the mechanisms that make Neural Networks learn.

---

# 🆚 Machine Learning vs Neural Networks

| Machine Learning | Neural Networks |
|------------------|-----------------|
| Manual Features | Automatic Features |
| Simpler Models | Complex Models |
| Smaller Datasets | Large Datasets |
| Easier to Explain | Harder to Explain |
| Faster Training | Slower Training |

---

# ⚠ Common Mistakes

❌ Neural Networks copy the human brain.

✔ They are inspired by the brain.

---

❌ More hidden layers always mean better performance.

✔ More layers increase complexity but also require more data and computation.

---

❌ Neural Networks only work for images.

✔ They are used for text, speech, video, and structured data as well.

---

# 🎤 Interview Questions

## Q1. What is an Artificial Neural Network?

An ANN is a collection of interconnected artificial neurons that learn patterns from data.

---

## Q2. Why are they called Neural Networks?

Because they are inspired by the way neurons in the human brain communicate.

---

## Q3. What are the three main layers?

- Input Layer
- Hidden Layer(s)
- Output Layer

---

## Q4. Which layer performs most of the learning?

The Hidden Layers.

---

## Q5. Why are Neural Networks powerful?

Because they automatically learn useful features from data.

---

# ⭐ Must Remember

✅ ANN is inspired by the human brain.

✅ Input Layer receives data.

✅ Hidden Layers learn patterns.

✅ Output Layer makes predictions.

✅ ANN is the foundation of Deep Learning.

---

# 📝 Summary

In this chapter you learned:

- Why Artificial Neural Networks were invented.
- Their inspiration from the human brain.
- The three main layers.
- How they process information.
- Why they power modern AI.

---

# ➡ Next Chapter

## Perceptron

A Neural Network is made up of many neurons.

But what exactly is a neuron?

How does one neuron make a decision?

In the next chapter, we'll build and understand the **Perceptron**, the smallest building block of every Neural Network.