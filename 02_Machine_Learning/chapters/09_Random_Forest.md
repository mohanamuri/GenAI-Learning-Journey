# Random Forest

> **Module:** 02 - Machine Learning
>
> **Reading Time:** 12 Minutes
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What Random Forest is.
- Why it performs better than a single Decision Tree.
- How Random Forest makes predictions.
- When to use it.
- Its advantages and limitations.

---

# 🤔 Why Should You Learn This?

Suppose a bank has one experienced loan officer.

The officer makes good decisions.

But sometimes,

the officer makes mistakes.

Now imagine instead of asking **one officer**,

you ask **100 experienced officers**.

Then,

you choose the answer that most officers agree on.

Usually,

that decision is more reliable.

This is the basic idea behind Random Forest.

---

# 🧠 What is Random Forest?

Random Forest is a Machine Learning algorithm that combines **many Decision Trees**.

Instead of depending on one tree,

it asks multiple trees to make predictions.

The final prediction is decided by **majority voting**.

---

# 🌳 Random Forest Workflow

```text
Historical Data

        │

        ▼

Decision Tree 1

Decision Tree 2

Decision Tree 3

...

Decision Tree 100

        │

        ▼

Majority Voting

        │

        ▼

Final Prediction
```

Think of it as asking many experts before making a decision.

---

# 🏦 Our Loan Eligibility Project

Suppose three trees predict:

```text
Tree 1

↓

Approved

Tree 2

↓

Approved

Tree 3

↓

Rejected
```

Voting result:

```text
Approved

2 Votes

Rejected

1 Vote
```

Final Decision

```text
Approved
```

This is why Random Forest is generally more reliable than a single Decision Tree.

---

# Our Python Example

We implemented:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

Our program displayed:

```text
Number of Trees : 100
```

Scikit-Learn creates **100 trees by default**.

Each tree learns slightly different patterns.

---

# Why Does It Work Better?

A single Decision Tree may learn an incorrect pattern.

Random Forest reduces this risk.

```text
One Tree

↓

Can Make Mistakes

100 Trees

↓

Mistakes Cancel Out

↓

Better Prediction
```

This makes Random Forest more stable.

---

# Decision Tree vs Random Forest

| Decision Tree | Random Forest |
|---------------|---------------|
| One Tree | Many Trees |
| Simple | More Powerful |
| Easy to Explain | Harder to Explain |
| Can Overfit | Less Overfitting |
| Faster | Slightly Slower |

---

# When Should You Use Random Forest?

Choose Random Forest when:

- Higher accuracy is required.
- Overfitting is a concern.
- The dataset is medium or large.
- Explainability is less important than performance.

Examples:

- Loan Approval
- Fraud Detection
- Credit Risk Analysis
- Customer Churn Prediction

---

# When Should You NOT Use It?

Avoid Random Forest when:

- You need very simple explanations.
- Model interpretability is the top priority.
- Extremely fast predictions are required.

In such situations,

Decision Tree may be a better choice.

---

# 🧠 Think Like an AI Engineer

Ask yourself:

```text
Do I want

Simple Explanation?

↓

Decision Tree


Higher Accuracy?

↓

Random Forest
```

The choice depends on the business requirement.

---

# 💼 AI Engineer Note

Many production systems start with a Decision Tree.

If accuracy is not sufficient,

the next algorithm engineers often try is Random Forest.

It is one of the most popular classification algorithms in industry.

---

# ⚠ Common Mistakes

❌ Random Forest is one Decision Tree.

✔ It is a collection of many Decision Trees.

---

❌ More trees always mean better results.

✔ More trees improve stability, but after a point they increase computation with little improvement.

---

❌ Random Forest never overfits.

✔ It greatly reduces overfitting, but it can still overfit in some situations.

---

# 🎤 Interview Questions

## Q1. What is Random Forest?

Random Forest is an ensemble Machine Learning algorithm that combines multiple Decision Trees to improve prediction accuracy.

---

## Q2. Why is Random Forest better than a Decision Tree?

Because it combines predictions from multiple trees, reducing overfitting and improving stability.

---

## Q3. How does Random Forest make its final prediction?

Each Decision Tree votes.

The majority vote becomes the final prediction.

---

## Q4. Why did we use 100 trees?

Scikit-Learn uses 100 trees by default.

This usually provides a good balance between accuracy and performance.

---

## Q5. Is Random Forest used for Classification or Regression?

Both.

- RandomForestClassifier → Classification
- RandomForestRegressor → Regression

---

# ⭐ Must Remember

✅ Random Forest combines many Decision Trees.

✅ Final prediction is based on voting.

✅ More stable than a single Decision Tree.

✅ Reduces overfitting.

✅ Widely used in real-world ML projects.

---

# 📝 Summary

In this chapter you learned:

- What Random Forest is.
- Why it performs better than a Decision Tree.
- How voting works.
- When to use it.
- Its advantages and limitations.

---

# ➡ Next Chapter

## Support Vector Machine (SVM)

Decision Trees divide data using a series of questions.

Random Forest uses many Decision Trees.

But what if we could separate two classes using the **best possible boundary**?

That is exactly what Support Vector Machine (SVM) does.