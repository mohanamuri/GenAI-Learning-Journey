# Classification vs Regression

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

- What Classification is.
- What Regression is.
- Their differences.
- How to choose the correct Machine Learning algorithm.

---

# 🤔 Why Should You Learn This?

Once you've decided your problem is **Supervised Learning**, the next question is:

> **What type of prediction do I need?**

There are only two possibilities.

```text
Prediction

│

├── Category

└── Number
```

Your answer decides which algorithm to use.

---

# 📊 Decision Flow

```text
Need Prediction

        │
   ┌────┴────┐
   │         │
Category   Number
   │         │
   ▼         ▼
Classification
         Regression
```

This is one of the most important diagrams in Machine Learning.

---

# 🧠 What is Classification?

Classification predicts a **category** or **label**.

The output is one of a fixed number of choices.

Examples

```text
Approved / Rejected

Spam / Not Spam

Pass / Fail

Yes / No

Fraud / Genuine
```

The answer belongs to a category.

---

# 🏦 Our Project

Our AI Loan Eligibility System predicts:

```text
Approved

OR

Rejected
```

These are categories.

Therefore,

our project is a **Classification** problem.

---

# More Classification Examples

| Problem | Prediction |
|----------|------------|
| Loan Approval | Approved / Rejected |
| Spam Detection | Spam / Not Spam |
| Disease Detection | Positive / Negative |
| Student Result | Pass / Fail |
| Face Recognition | Person Name |

All of these predict categories.

---

# 🧠 What is Regression?

Regression predicts a **continuous numerical value**.

Examples

```text
Salary

House Price

Temperature

Sales

Rainfall
```

The answer is a number.

---

# Example

Suppose we predict salary.

```text
Experience

↓

Salary

↓

₹15.75 LPA
```

The prediction is not limited to fixed categories.

It can be any value.

---

# More Regression Examples

| Problem | Prediction |
|----------|------------|
| Salary Prediction | ₹18 LPA |
| House Price | ₹75 Lakhs |
| Temperature | 32.5°C |
| Monthly Sales | ₹15 Crores |

These are continuous values.

---

# 📊 Comparison

| Classification | Regression |
|----------------|------------|
| Predicts Categories | Predicts Numbers |
| Output is Fixed | Output is Continuous |
| Loan Approval | Salary Prediction |
| Spam Detection | House Price |
| Pass / Fail | Temperature |

---

# Common Algorithms

## Classification

- Decision Tree
- Random Forest
- Logistic Regression
- SVM
- KNN
- Naive Bayes

---

## Regression

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Notice something.

Some algorithms have both

Classification

and

Regression versions.

---

# 🧠 Think Like an AI Engineer

Before choosing an algorithm,

ask yourself one simple question.

```text
What am I predicting?

        │

   ┌────┴────┐

Category   Number

   │         │

Decision   Linear
Tree       Regression
```

Never start with the algorithm.

Start with the prediction type.

---

# 🏦 Our AI Loan Eligibility Project

Let's classify our project.

Business Problem

```text
Loan Approval
```

Prediction

```text
Approved

Rejected
```

Prediction Type

```text
Category
```

Machine Learning Type

```text
Classification
```

Algorithm

```text
Decision Tree Classifier
```

Everything now makes sense.

---

# 💼 AI Engineer Note

Many beginners memorize algorithms.

Experienced engineers first identify the prediction type.

Only then do they choose an algorithm.

This saves time and leads to better model selection.

---

# ⚠ Common Mistakes

❌ Salary Prediction is Classification.

✔ Salary is a numerical value.

It is Regression.

---

❌ Loan Approval is Regression.

✔ Loan Approval predicts categories.

It is Classification.

---

❌ Logistic Regression is used for Regression.

✔ Despite its name,

Logistic Regression is a **Classification algorithm**.

This is a very common interview question.

---

# 🎤 Interview Questions

## Q1. What is Classification?

Classification predicts categories or labels.

Example:

Approved / Rejected.

---

## Q2. What is Regression?

Regression predicts continuous numerical values.

Example:

Salary Prediction.

---

## Q3. Is our AI Loan Eligibility System Classification or Regression?

Classification.

Because it predicts Approved or Rejected.

---

## Q4. Is Logistic Regression a Regression algorithm?

No.

It is a Classification algorithm.

---

## Q5. Give three examples of Classification.

- Loan Approval
- Spam Detection
- Disease Detection

---

## Q6. Give three examples of Regression.

- Salary Prediction
- House Price Prediction
- Temperature Prediction

---

# ⭐ Must Remember

✅ Classification predicts categories.

✅ Regression predicts numbers.

✅ Our Loan Eligibility System is Classification.

✅ Linear Regression predicts continuous values.

✅ Logistic Regression is used for Classification.

---

# 📝 Summary

In this chapter you learned:

- What Classification is.
- What Regression is.
- Their differences.
- Which algorithms belong to each category.
- How to identify the correct prediction type.

---

# ➡ Next Chapter

## Train-Test Split

Once you've selected an algorithm,

the next question is:

```text
Should I train the model
using all the data?

        │

        ▼

No
```

Why?

Because we need unseen data to evaluate how well the model performs.

In the next chapter, we'll learn why we split data into **Training** and **Testing** datasets.