# Chapter 13 - Model Evaluation

---

# Learning Objectives

After completing this chapter, you will understand:

- What Model Evaluation is
- Why Model Evaluation is important
- Training vs Testing Data
- Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC Score
- Best Practices

---

# Introduction

Training a neural network is only the beginning.

The real question is:

> **How well does the model perform on new, unseen data?**

The process of measuring a model's performance is called **Model Evaluation**.

Without evaluation, we cannot determine whether a model is useful in real-world applications.

---

# Why Model Evaluation?

Imagine a student.

Training is like studying for an exam.

Evaluation is writing the exam.

A student who only studies but never takes an exam cannot prove their knowledge.

Similarly,

A neural network must be tested using unseen data.

---

# Training vs Testing Data

A dataset is normally divided into two parts.

```
Entire Dataset

↓

Training Data (80%)

↓

Testing Data (20%)
```

Training Data

- Used for learning.

Testing Data

- Used only for evaluation.

Our production project follows this approach using:

```python
train_test_split()
```

---

# Why Not Test on Training Data?

Suppose a student memorizes all exam answers.

They score

```
100%
```

But when asked different questions,

they fail.

Similarly,

Testing on training data gives misleading results.

Always evaluate using unseen data.

---

# Evaluation Workflow

```
Dataset

↓

Split

↓

Train

↓

Predict

↓

Compare

↓

Calculate Metrics

↓

Performance Report
```

---

# Accuracy

Accuracy is the percentage of correct predictions.

Formula

```
Accuracy

=

Correct Predictions

/

Total Predictions
```

Example

```
100 Predictions

↓

92 Correct

↓

Accuracy = 92%
```

TensorFlow

```python
metrics=["accuracy"]
```

---

# Confusion Matrix

A Confusion Matrix shows where predictions are correct and incorrect.

```
                Actual

           Yes      No

Pred Yes   TP      FP

Pred No    FN      TN
```

Where

TP

True Positive

Correctly predicted Positive.

FP

False Positive

Incorrectly predicted Positive.

FN

False Negative

Incorrectly predicted Negative.

TN

True Negative

Correctly predicted Negative.

---

# Precision

Precision answers

> **When the model predicts "Approved", how often is it correct?**

Formula

```
TP

/

(TP + FP)
```

High Precision

↓

Few False Positives.

---

# Recall

Recall answers

> **Out of all actual approved loans, how many did the model correctly identify?**

Formula

```
TP

/

(TP + FN)
```

High Recall

↓

Few False Negatives.

---

# F1 Score

Sometimes,

Accuracy alone is misleading.

F1 Score combines

- Precision
- Recall

into one metric.

Formula

```
2

×

Precision

×

Recall

/

(Precision + Recall)
```

Higher is better.

---

# ROC Curve

ROC

Receiver Operating Characteristic

Shows how well the classifier separates two classes.

```
Good Model

↗

Bad Model

──────
```

The closer the curve is to the top-left corner,

the better the model.

---

# AUC Score

Area Under the Curve

Measures overall classification performance.

```
1.0

Perfect

0.5

Random Guess
```

The closer to

```
1
```

the better.

---

# TensorFlow Evaluation

Example

```python
loss, accuracy = model.evaluate(

X_test,

y_test

)
```

Returns

```
Loss

Accuracy
```

---

# Our Production Project

Evaluation Flow

```
Dataset

↓

Split

↓

Train

↓

Evaluate

↓

Save Model

↓

Deploy
```

We already implemented:

```python
model.evaluate()
```

inside our project.

---

# Important Metrics

| Metric | Purpose |
|----------|----------|
| Accuracy | Overall correctness |
| Precision | Correct positive predictions |
| Recall | Finds actual positives |
| F1 Score | Balance between Precision & Recall |
| ROC | Classification quality |
| AUC | Overall classifier performance |

---

# When Accuracy Is Misleading

Suppose

```
1000 Customers

990 Rejected

10 Approved
```

A model predicts

```
Rejected

for everyone.
```

Accuracy

```
99%
```

Looks excellent.

But

The model never approves anyone.

This is why Precision, Recall and F1 Score are important.

---

# Best Practices

- Keep separate testing data.
- Never evaluate using training data.
- Use multiple evaluation metrics.
- Monitor both loss and accuracy.
- Compare different models.

---

# Summary

Model Evaluation measures how well a trained model performs on unseen data.

Good evaluation ensures the model is reliable before deployment.

Accuracy alone is not sufficient; Precision, Recall, F1 Score and AUC provide deeper insight into model performance.

---

# Key Takeaways

- Training and Evaluation are different.
- Use unseen data.
- Accuracy is not always enough.
- Confusion Matrix explains prediction quality.
- Precision and Recall solve class imbalance issues.
- Evaluate before deployment.

---

# Common Mistakes

- Testing on training data.
- Using only Accuracy.
- Ignoring False Positives.
- Ignoring False Negatives.
- Deploying without evaluation.

---

# Interview Questions

1. Why do we evaluate a model?
2. Why split training and testing data?
3. What is Accuracy?
4. What is Precision?
5. What is Recall?
6. What is F1 Score?
7. What is a Confusion Matrix?
8. What is ROC?
9. What is AUC?
10. Why can Accuracy be misleading?

---

# Hands-on Exercise

Modify our project.

After prediction,

calculate:

- Accuracy
- Precision
- Recall
- F1 Score

using **Scikit-Learn**.

Observe the differences.

---

# AI Engineering Perspective

In production AI systems, evaluation is much more than printing an accuracy score.

Teams continuously monitor model performance using validation datasets, business metrics, and real-world feedback. If performance drops over time due to changing data patterns (data drift), the model may need retraining.

Model evaluation is therefore not a one-time task but an ongoing part of the AI lifecycle.

---

# 🧠 Connections

Previous Chapter

↓

Building Your First Neural Network

↓

Current Chapter

Model Evaluation

↓

Next Chapter

Overfitting & Underfitting