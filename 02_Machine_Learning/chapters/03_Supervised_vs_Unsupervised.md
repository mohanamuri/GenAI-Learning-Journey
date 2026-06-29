# Supervised Learning vs Unsupervised Learning

> **Module:** 02 - Machine Learning
>
> **Reading Time:** 10 Minutes
>
> **Difficulty:** ⭐☆☆☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What Supervised Learning is.
- What Unsupervised Learning is.
- The differences between them.
- When to choose each approach.

---

# 🤔 Why Should You Learn This?

Whenever you start a Machine Learning project, the first question is:

> **Do I have the correct answers (labels)?**

Your answer decides which type of Machine Learning to use.

---

# 📊 Decision Flow

```text
Do you have Labels?

       │
  ┌────┴────┐
  │         │
 Yes        No
  │         │
  ▼         ▼
Supervised  Unsupervised
Learning    Learning
```

Remember this diagram.

---

# 🧠 What is Supervised Learning?

In Supervised Learning,

the model learns using:

- Input Data
- Correct Answers (Labels)

The model studies the relationship between inputs and outputs.

---

# Example

Loan Dataset

| Salary | Credit Score | Loan |
|---------|--------------|------|
| 40000 | 750 | Approved |
| 25000 | 620 | Rejected |
| 70000 | 800 | Approved |

Notice something.

We already know the correct answer.

```text
Approved

Rejected
```

These are called **Labels**.

---

# 📊 Supervised Learning

```text
Input Data

+

Correct Answers

↓

Model Learns

↓

Prediction
```

The model learns from examples.

---

# Real-World Examples

- Loan Approval
- Spam Detection
- House Price Prediction
- Disease Prediction
- Student Pass/Fail Prediction

---

# 🏦 Our Project

Our AI Loan Eligibility System is a **Supervised Learning** problem.

Why?

Because our historical dataset already contains the answer.

Example

| Salary | Credit Score | Loan |
|---------|--------------|------|
| 50000 | 720 | Approved |
| 30000 | 600 | Rejected |

The model learns from these previous decisions.

---

# 🧠 What is Unsupervised Learning?

In Unsupervised Learning,

we provide only input data.

There are **no labels**.

The model tries to discover hidden patterns by itself.

---

# Example

Customer Data

| Customer | Age | Income |
|-----------|-----|--------|
| A | 22 | 25000 |
| B | 24 | 28000 |
| C | 55 | 90000 |
| D | 58 | 95000 |

No one tells the model:

- Young Customers
- Senior Customers

The model automatically groups similar customers.

---

# 📊 Unsupervised Learning

```text
Input Data

↓

Model Finds Patterns

↓

Groups Similar Data
```

No correct answers are provided.

---

# Real-World Examples

- Customer Segmentation
- Product Grouping
- Market Basket Analysis
- Fraud Pattern Discovery
- Data Clustering

---

# 📋 Comparison

| Supervised Learning | Unsupervised Learning |
|---------------------|-----------------------|
| Has Labels | No Labels |
| Learns from Correct Answers | Finds Hidden Patterns |
| Predicts Outcomes | Discovers Groups |
| Used for Prediction | Used for Exploration |

---

# Common Algorithms

### Supervised Learning

- Decision Tree
- Random Forest
- Linear Regression
- Logistic Regression
- SVM
- KNN
- Naive Bayes

---

### Unsupervised Learning

- K-Means Clustering
- DBSCAN
- Hierarchical Clustering

We will learn these in later modules.

---

# 🧠 Think Like an AI Engineer

Before selecting an algorithm, ask:

```text
Do I know the correct answer?

↓

YES

↓

Supervised Learning


Do I only have raw data?

↓

YES

↓

Unsupervised Learning
```

This simple question saves a lot of confusion.

---

# 💼 AI Engineer Note

Around 80–90% of business Machine Learning projects use **Supervised Learning** because companies usually have historical data with known outcomes.

Examples:

- Loan Approved / Rejected
- Fraud / Not Fraud
- Customer Left / Stayed
- Email Spam / Not Spam

Unsupervised Learning is commonly used for discovering patterns when labels are unavailable.

---

# ⚠ Common Mistakes

❌ Supervised Learning means the developer supervises the model.

✔ "Supervised" means the dataset contains correct answers (labels).

---

❌ Unsupervised Learning is less important.

✔ It is extremely useful for discovering hidden patterns.

---

❌ Every ML problem is supervised.

✔ Many real-world problems involve unlabeled data.

---

# 🎤 Interview Questions

## Q1. What is Supervised Learning?

Supervised Learning is a type of Machine Learning where the model learns using input data and corresponding correct answers (labels).

---

## Q2. What is Unsupervised Learning?

Unsupervised Learning is a type of Machine Learning where the model receives only input data and discovers hidden patterns without labels.

---

## Q3. Is our AI Loan Eligibility System supervised or unsupervised?

Supervised Learning.

Because the historical dataset already contains loan decisions.

---

## Q4. Name three Supervised Learning algorithms.

- Decision Tree
- Random Forest
- Logistic Regression

---

## Q5. Name one Unsupervised Learning algorithm.

K-Means Clustering.

---

# ⭐ Must Remember

✅ Supervised Learning → Labels Available.

✅ Unsupervised Learning → No Labels.

✅ Supervised Learning predicts outcomes.

✅ Unsupervised Learning discovers patterns.

✅ Our Loan Eligibility Project uses Supervised Learning.

---

# 📝 Summary

In this chapter you learned:

- What Supervised Learning is.
- What Unsupervised Learning is.
- Their differences.
- Real-world examples.
- How to decide which one to use.

---

# ➡ Next Chapter

## Classification vs Regression

Now that you understand Supervised Learning,

the next question is:

```text
What type of prediction do I need?

        │
  ┌─────┴─────┐
  │           │
Category    Number
  │           │
  ▼           ▼
Classification
            Regression
```

This decision determines which Machine Learning algorithm you should choose.