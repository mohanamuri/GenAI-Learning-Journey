# Chapter 14 - Overfitting & Underfitting

---

# Learning Objectives

After completing this chapter, you will understand:

- What Overfitting is
- What Underfitting is
- Why they occur
- Bias vs Variance
- Identifying Overfitting
- Identifying Underfitting
- Prevention Techniques
- Early Stopping
- Regularization
- Best Practices

---

# Introduction

The goal of Machine Learning and Deep Learning is not to memorize training data.

The goal is to **learn patterns** that work well on new, unseen data.

Two common problems prevent this:

- Underfitting
- Overfitting

Finding the right balance between them is one of the biggest challenges in AI.

---

# The Ideal Model

```
Training Data

↓

Learn Patterns

↓

Generalize

↓

New Data

↓

Correct Prediction
```

A good model should perform well on both training and unseen data.

---

# Underfitting

Underfitting occurs when the model is **too simple** to learn the underlying patterns.

```
Training Accuracy

↓

Low

Testing Accuracy

↓

Low
```

The model performs poorly everywhere.

---

# Example

Suppose we want to predict loan approval.

Inputs:

- Salary
- Experience
- Credit Score

If we build a neural network with only one neuron, it may not capture the complex relationships among these features.

Result:

```
Poor Learning

↓

Poor Predictions
```

---

# Causes of Underfitting

- Too few neurons
- Too few layers
- Too few epochs
- Poor feature selection
- Very high learning rate

---

# Signs of Underfitting

- High training loss
- Low training accuracy
- High testing loss
- Low testing accuracy

The model never learns properly.

---

# Overfitting

Overfitting occurs when the model memorizes the training data instead of learning general patterns.

```
Training Accuracy

↓

Very High

Testing Accuracy

↓

Low
```

The model performs well on known data but poorly on unseen data.

---

# Example

Imagine a student memorizing previous exam answers.

During practice:

```
100%
```

During the real exam:

```
55%
```

The student memorized instead of understanding.

Neural networks behave similarly.

---

# Causes of Overfitting

- Too many neurons
- Too many layers
- Excessive epochs
- Small datasets
- Complex models

---

# Signs of Overfitting

Training

```
Accuracy

98%
```

Testing

```
Accuracy

72%
```

Large differences between training and testing performance indicate overfitting.

---

# Visual Representation

```
Training Accuracy

|

|                  /
|                 /
|                /
|               /
|______________/

Testing Accuracy

|

|           /\
|          /  \
|         /    \
|________/      \____
```

Initially both improve.

Eventually the testing performance starts decreasing while training keeps improving.

That is overfitting.

---

# Bias vs Variance

Underfitting

↓

High Bias

↓

Low Variance

---

Overfitting

↓

Low Bias

↓

High Variance

---

Ideal Model

↓

Balanced Bias

Balanced Variance

---

# Comparison

| Underfitting | Good Fit | Overfitting |
|--------------|----------|-------------|
| Low Accuracy | High Accuracy | High Training Accuracy |
| Poor Learning | Good Learning | Memorization |
| High Bias | Balanced | High Variance |

---

# How to Reduce Underfitting

- Increase neurons
- Add hidden layers
- Train longer
- Improve features
- Reduce learning rate

---

# How to Reduce Overfitting

- Increase dataset size
- Use Dropout
- Early Stopping
- Regularization
- Data Augmentation
- Simpler architecture

We will study Dropout and Regularization in the next chapter.

---

# Early Stopping

Instead of training for a fixed number of epochs,

monitor validation performance.

```
Epoch

↓

Validation Loss Stops Improving

↓

Stop Training
```

TensorFlow provides an EarlyStopping callback.

---

# Real-World Example

Medical Diagnosis

Overfitting

↓

Model memorizes hospital-specific data.

↓

Fails in another hospital.

This can lead to incorrect diagnoses.

Therefore,

generalization is far more important than memorization.

---

# Our Production Project

Our Loan Eligibility System currently uses a small synthetic dataset.

As we improve the dataset and increase the model complexity, we must monitor:

- Training Accuracy
- Validation Accuracy
- Testing Accuracy

to ensure the model generalizes well.

---

# Best Practices

- Monitor validation performance.
- Avoid unnecessarily complex models.
- Use adequate training data.
- Apply regularization techniques.
- Evaluate using unseen data.

---

# Summary

A successful neural network should learn patterns rather than memorize data.

Underfitting indicates insufficient learning, while overfitting indicates excessive memorization.

The goal is to build models that generalize well to unseen data.

---

# Key Takeaways

- Underfitting = Poor Learning
- Overfitting = Memorization
- Generalization is the objective.
- Balance Bias and Variance.
- Monitor validation performance.
- Prevent overfitting before deployment.

---

# Common Mistakes

- Training for too many epochs.
- Ignoring validation accuracy.
- Using very small datasets.
- Building unnecessarily large networks.
- Evaluating only on training data.

---

# Interview Questions

1. What is Underfitting?
2. What is Overfitting?
3. How do you identify Overfitting?
4. How do you reduce Underfitting?
5. How do you reduce Overfitting?
6. What is Bias?
7. What is Variance?
8. What is Generalization?
9. What is Early Stopping?
10. Why is validation accuracy important?

---

# Hands-on Exercise

Modify our project.

Experiment with:

- Epochs = 10
- Epochs = 100
- Hidden Layer = 4 Neurons
- Hidden Layer = 32 Neurons

Compare:

- Training Accuracy
- Testing Accuracy
- Loss

Observe when overfitting starts to appear.

---

# AI Engineering Perspective

One of the biggest responsibilities of an AI Engineer is ensuring that models perform well in production—not just during training.

Production systems continuously monitor model quality. If a deployed model starts showing signs of overfitting to historical data or poor performance on new data (often due to changing data patterns), it must be retrained or redesigned.

Building a model that generalizes well is more valuable than achieving perfect training accuracy.

---

# 🧠 Connections

Previous Chapter

↓

Model Evaluation

↓

Current Chapter

Overfitting & Underfitting

↓

Next Chapter

Dropout & Regularization