# Model Evaluation

> **Module:** 02 - Machine Learning
>
> **Reading Time:** 18 Minutes
>
> **Difficulty:** ⭐⭐⭐☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- Why model evaluation is important.
- What Accuracy means.
- Why Accuracy alone is not enough.
- What a Confusion Matrix is.
- What Precision, Recall, and F1 Score mean.
- How AI Engineers compare Machine Learning models.

---

# 🤔 Why Should You Learn This?

Suppose you build a Machine Learning model.

It reports:

```text
Accuracy = 99%
```

Can you immediately say:

> My model is excellent.

No.

A high accuracy does not always mean a good model.

The quality of a model depends on **what mistakes it makes**.

---

# 🏦 Real Business Example

Imagine a bank receives

```text
1000 Loan Applications
```

Out of them

```text
990 Approved

10 Rejected
```

Now imagine a very simple model.

It predicts

```text
Approve Everyone
```

Result

```text
990 Correct

10 Wrong

Accuracy = 99%
```

Looks amazing.

But the bank approved every risky customer.

Business Result

```text
Huge Financial Loss
```

This is why Accuracy alone is dangerous.

---

# 📊 Model Evaluation Workflow

```text
Train Model

        │

        ▼

Predict

        │

        ▼

Compare with Actual Results

        │

        ▼

Evaluation Metrics
```

Evaluation tells us how good the model really is.

---

# What is Accuracy?

Accuracy tells us

```text
Correct Predictions

-------------------

Total Predictions
```

Example

```text
100 Predictions

↓

90 Correct

↓

Accuracy = 90%
```

Simple.

But not always enough.

---

# Confusion Matrix

A Confusion Matrix shows

**what kinds of mistakes** the model made.

```text
                 Actual

             Yes      No

Pred Yes     TP       FP

Pred No      FN       TN
```

Where

TP

True Positive

Correct Positive Prediction

---

TN

True Negative

Correct Negative Prediction

---

FP

False Positive

Incorrect Positive Prediction

---

FN

False Negative

Incorrect Negative Prediction

---

# Loan Approval Example

Suppose

```text
Actual

Approved

Model

Approved
```

Correct

↓

True Positive

---

Suppose

```text
Actual

Rejected

Model

Approved
```

Wrong

↓

False Positive

The bank approved someone who should have been rejected.

---

Suppose

```text
Actual

Approved

Model

Rejected
```

↓

False Negative

A genuine customer lost the opportunity.

---

# Precision

Precision answers:

```text
Of all customers
the model approved,

how many were actually eligible?
```

Formula

```text
TP

-------------

TP + FP
```

High Precision means

Few incorrect approvals.

---

# Recall

Recall answers:

```text
Of all eligible customers,

how many did the model correctly identify?
```

Formula

```text
TP

-------------

TP + FN
```

High Recall means

Few eligible customers are missed.

---

# F1 Score

Sometimes

Precision is high.

Recall is low.

Or vice versa.

F1 Score balances both.

```text
Precision

+

Recall

↓

Balanced Score
```

When both Precision and Recall matter,

F1 Score is a good metric.

---

# Accuracy vs Precision vs Recall

| Metric | Answers |
|---------|----------|
| Accuracy | Overall Correct Predictions |
| Precision | How many predicted positives were correct? |
| Recall | How many actual positives were found? |
| F1 Score | Balance between Precision and Recall |

---

# Which Metric Should You Use?

Loan Approval

↓

Precision

Banks don't want to approve risky customers.

---

Disease Detection

↓

Recall

Doctors don't want to miss sick patients.

---

General Classification

↓

Accuracy

When classes are balanced.

---

Precision & Recall Important

↓

F1 Score

---

# Our AI Loan Eligibility Project

When evaluating our model,

we used

```python
accuracy_score()
```

In real banking systems,

additional metrics such as

- Precision
- Recall
- F1 Score

are also evaluated before deployment.

---

# 🧠 Think Like an AI Engineer

Never ask

```text
What is my Accuracy?
```

Instead ask

```text
What mistakes is my model making?
```

This is how experienced AI Engineers think.

---

# 💼 AI Engineer Note

Real companies rarely deploy models using only Accuracy.

They compare multiple evaluation metrics before selecting a model.

Evaluation is one of the most important stages in the Machine Learning lifecycle.

---

# ⚠ Common Mistakes

❌ High Accuracy always means a good model.

✔ Not necessarily.

---

❌ Accuracy is enough.

✔ Evaluate Precision, Recall, and F1 Score as well.

---

❌ Ignore business impact.

✔ Always choose evaluation metrics based on business requirements.

---

# 🎤 Interview Questions

## Q1. What is Model Evaluation?

Model Evaluation measures how well a Machine Learning model performs on unseen data.

---

## Q2. What is Accuracy?

Accuracy is the percentage of correct predictions out of the total predictions.

---

## Q3. Why is Accuracy sometimes misleading?

Because it may hide important prediction errors, especially in imbalanced datasets.

---

## Q4. What is Precision?

Precision measures how many predicted positive cases are actually positive.

---

## Q5. What is Recall?

Recall measures how many actual positive cases were correctly identified.

---

## Q6. What is F1 Score?

F1 Score is the harmonic mean of Precision and Recall.

It balances both metrics.

---

# ⭐ Must Remember

✅ Evaluate every model.

✅ Accuracy alone is not enough.

✅ Precision reduces false approvals.

✅ Recall reduces missed positives.

✅ F1 Score balances Precision and Recall.

---

# 📝 Summary

In this chapter you learned:

- Why Model Evaluation is important.
- Accuracy and its limitations.
- Confusion Matrix.
- Precision.
- Recall.
- F1 Score.
- Choosing the right evaluation metric.

---

# ➡ Next Chapter

## Cross Validation

Suppose one Train-Test Split gives

```text
Accuracy = 92%
```

Can we completely trust it?

Not always.

Different dataset splits can produce different results.

The solution is

```text
Cross Validation
```

which evaluates the model multiple times using different data splits.