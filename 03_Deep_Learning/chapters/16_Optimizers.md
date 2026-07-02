# Chapter 16 - Optimizers

---

# Learning Objectives

After completing this chapter, you will understand:

- What an Optimizer is
- Why Optimizers are needed
- How Optimizers work
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent
- Adam Optimizer
- RMSprop
- Adagrad
- AdamW
- Choosing the Right Optimizer
- Best Practices

---

# Introduction

Training a Neural Network involves continuously updating its weights to reduce prediction errors.

The component responsible for updating these weights is called the **Optimizer**.

Without an optimizer, the neural network would never learn.

---

# What is an Optimizer?

An Optimizer is an algorithm that adjusts the weights of a neural network to minimize the loss function.

Workflow

```
Prediction

↓

Loss

↓

Gradient

↓

Optimizer

↓

Update Weights

↓

Better Prediction
```

Every training step ends with the optimizer modifying the model's weights.

---

# Why Do We Need Optimizers?

Suppose our model predicts

```
Loan Approved

Probability = 0.35
```

Actual Answer

```
Approved
```

The prediction is incorrect.

The optimizer changes the weights so that the next prediction becomes more accurate.

This process repeats thousands of times during training.

---

# Gradient Descent

The simplest optimizer is Gradient Descent.

Idea

```
Current Error

↓

Move Towards Lower Error

↓

Repeat
```

Imagine standing on top of a mountain.

Your goal is to reach the lowest point.

Each step you take reduces your height.

Similarly,

Gradient Descent reduces the loss after every iteration.

---

# Types of Gradient Descent

There are three common approaches.

---

## Batch Gradient Descent

Uses the entire dataset before updating weights.

```
Entire Dataset

↓

One Update
```

Advantages

- Stable

Disadvantages

- Slow on large datasets.

---

## Stochastic Gradient Descent (SGD)

Updates weights after every training sample.

```
One Sample

↓

Update

↓

Next Sample

↓

Update
```

Advantages

- Faster learning

Disadvantages

- Noisy updates.

TensorFlow

```python
optimizer="sgd"
```

---

## Mini-Batch Gradient Descent

The most common approach.

```
Dataset

↓

Small Batch

↓

Update

↓

Next Batch

↓

Update
```

This combines the advantages of Batch and SGD.

TensorFlow uses Mini-Batch Gradient Descent internally when you specify `batch_size`.

---

# Adam Optimizer

Adam stands for

```
Adaptive Moment Estimation
```

It automatically adjusts the learning rate during training.

Advantages

- Fast convergence
- Stable
- Works well for most problems
- Default choice for many projects

TensorFlow

```python
optimizer="adam"
```

This is the optimizer we used in our Loan Eligibility project.

---

# RMSprop

RMSprop adapts the learning rate based on recent gradients.

Advantages

- Good for recurrent neural networks
- Faster convergence than basic SGD

TensorFlow

```python
optimizer="rmsprop"
```

---

# Adagrad

Adagrad automatically decreases the learning rate for frequently updated parameters.

Advantages

- Useful for sparse data

Disadvantages

- Learning rate can become too small over time.

TensorFlow

```python
optimizer="adagrad"
```

---

# AdamW

AdamW is an improved version of Adam.

It separates weight decay from gradient updates.

Advantages

- Better generalization
- Widely used in Transformers and LLMs

Today, many large AI models use AdamW.

---

# Optimizer Comparison

| Optimizer | Speed | Stability | Typical Use |
|-----------|-------|-----------|-------------|
| SGD | Medium | Medium | Simple models |
| Mini-Batch | High | High | Most Deep Learning |
| Adam | Very High | Very High | Default choice |
| RMSprop | High | High | RNNs |
| Adagrad | Medium | Medium | Sparse features |
| AdamW | Very High | Excellent | Transformers, LLMs |

---

# TensorFlow Example

```python
model.compile(

optimizer="adam",

loss="binary_crossentropy",

metrics=["accuracy"]

)
```

Changing the optimizer requires modifying only one line of code.

---

# Choosing the Right Optimizer

General recommendations

| Problem | Optimizer |
|----------|-----------|
| Beginner Projects | Adam |
| Image Classification | Adam |
| NLP | AdamW |
| Transformers | AdamW |
| RNN/LSTM | RMSprop |
| Simple Experiments | SGD |

If unsure,

start with **Adam**.

---

# Our Production Project

Current

```python
optimizer="adam"
```

Future experiments

```python
optimizer="sgd"
```

```python
optimizer="rmsprop"
```

```python
optimizer="adam"
```

We will compare:

- Accuracy
- Loss
- Training Time

to understand the impact of each optimizer.

---

# Best Practices

- Start with Adam.
- Tune learning rate before changing optimizers.
- Compare multiple optimizers during experimentation.
- Monitor convergence.
- Avoid changing many hyperparameters at once.

---

# Summary

Optimizers are responsible for updating neural network weights during training.

They help the model minimize the loss function and improve prediction accuracy.

Adam is the most commonly used optimizer because it combines speed, stability, and ease of use.

---

# Key Takeaways

- Optimizers update weights.
- Gradient Descent is the foundation.
- Mini-Batch is widely used.
- Adam is the default optimizer.
- AdamW dominates modern Transformer and LLM training.
- Optimizer choice affects training speed and model performance.

---

# Common Mistakes

- Changing optimizers without understanding the learning rate.
- Assuming one optimizer is always the best.
- Ignoring convergence.
- Comparing optimizers without using the same dataset.

---

# Interview Questions

1. What is an Optimizer?
2. Why do we need an Optimizer?
3. Explain Gradient Descent.
4. Difference between Batch, SGD and Mini-Batch Gradient Descent?
5. Why is Adam popular?
6. What is RMSprop?
7. What is Adagrad?
8. What is AdamW?
9. Which optimizer would you choose for an LLM?
10. Which optimizer did you use in your project and why?

---

# Hands-on Exercise

Modify our Loan Eligibility project.

Experiment with:

- Adam
- SGD
- RMSprop

Compare:

- Accuracy
- Loss
- Training Time

Record your observations.

---

# AI Engineering Perspective

Choosing an optimizer is not about selecting the newest algorithm—it is about selecting one that matches the problem, dataset, and model architecture.

In production AI systems, optimizer selection is part of **hyperparameter tuning**, where engineers systematically compare different configurations to improve performance.

For many enterprise applications, **Adam** is an excellent starting point, while **AdamW** has become the standard for training modern Transformer-based models and large language models.

---

# 🧠 Connections

Previous Chapter

↓

Dropout & Regularization

↓

Current Chapter

Optimizers

↓

Next Chapter

CNN Introduction