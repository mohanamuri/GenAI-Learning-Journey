# K-Nearest Neighbors (KNN)

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

- What K-Nearest Neighbors (KNN) is.
- How KNN makes predictions.
- What the value of **K** means.
- When to use KNN.
- Its advantages and limitations.

---

# 🤔 Why Should You Learn This?

Imagine you move into a new city.

You don't know whether a neighborhood is safe.

What would you do?

Most people ask nearby residents.

If most neighbors say,

```text
Safe
```

you'll probably think it's safe.

If most say,

```text
Unsafe
```

you'll likely avoid it.

KNN follows exactly the same idea.

---

# 🧠 What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a Machine Learning algorithm used for **Classification** and **Regression**.

Instead of learning rules during training,

it compares a new data point with the most similar existing data points.

The prediction is based on the majority of its nearest neighbors.

---

# 📊 KNN Workflow

```text
Historical Data

        │

        ▼

Find Nearest Neighbors

        │

        ▼

Majority Vote

        │

        ▼

Prediction
```

Unlike Decision Tree,

KNN does not build a tree.

It simply looks for similar examples.

---

# Example

Suppose a new loan applicant arrives.

Nearby customers are:

```text
Customer 1

Approved

Customer 2

Approved

Customer 3

Rejected
```

Majority Vote

```text
Approved

2 Votes

Rejected

1 Vote
```

Prediction

```text
Approved
```

---

# What Does "K" Mean?

K represents the number of neighbors considered.

Example

```text
K = 3

↓

Check 3 nearest neighbors
```

```text
K = 5

↓

Check 5 nearest neighbors
```

Choosing the right K is important.

---

# Our Python Example

We implemented:

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

Although we call **fit()**,

KNN does not build a traditional model.

It mainly stores the training data.

The real work happens during prediction.

---

# Why Does KNN Work?

The basic assumption is:

> Similar data points usually belong to the same category.

For example,

customers with similar:

- Salary
- Experience
- Credit Score

often receive similar loan decisions.

---

# Choosing the Value of K

Small K

```text
K = 1
```

Advantages

- Very flexible.

Disadvantages

- Sensitive to noise.

---

Large K

```text
K = 15
```

Advantages

- More stable.

Disadvantages

- May ignore important local patterns.

There is no perfect K.

It depends on the dataset.

---

# When Should You Use KNN?

Choose KNN when:

- Dataset is small.
- Similarity matters.
- Relationships are simple.
- Quick model building is required.

Examples

- Movie Recommendation
- Student Performance
- Product Classification
- Basic Medical Diagnosis

---

# When Should You NOT Use It?

Avoid KNN when:

- Dataset is very large.
- Prediction speed is important.
- Many features exist.
- Data contains significant noise.

KNN becomes slower as the dataset grows.

---

# 🆚 Algorithm Comparison

| If you need... | Choose... |
|----------------|-----------|
| Human-readable Rules | Decision Tree |
| Better Accuracy | Random Forest |
| Best Decision Boundary | SVM |
| Similarity-Based Prediction | KNN |

---

# 🧠 Think Like an AI Engineer

Ask yourself:

```text
Can similar customers
be expected to have
similar outcomes?

↓

Yes

↓

KNN may work well.
```

Remember,

KNN learns through similarity,

not through decision rules.

---

# 💼 AI Engineer Note

One interesting fact about KNN:

Training is very fast.

Prediction is relatively slower.

Why?

Because the algorithm searches for nearest neighbors every time a new prediction is made.

---

# ⚠ Common Mistakes

❌ KNN builds a complex model.

✔ KNN mainly stores the training data.

---

❌ Larger K is always better.

✔ Choosing K requires experimentation.

---

❌ KNN is suitable for huge datasets.

✔ Prediction becomes slower as the dataset grows.

---

# 🎤 Interview Questions

## Q1. What is KNN?

K-Nearest Neighbors is a Machine Learning algorithm that predicts outcomes based on the majority class of the nearest data points.

---

## Q2. What does K represent?

K represents the number of nearest neighbors considered during prediction.

---

## Q3. Why is KNN called a lazy learning algorithm?

Because it performs very little work during training.

Most computation happens during prediction.

---

## Q4. Is KNN used for Classification or Regression?

Both.

- KNeighborsClassifier → Classification
- KNeighborsRegressor → Regression

---

## Q5. What is one limitation of KNN?

Prediction becomes slower as the dataset size increases.

---

# ⭐ Must Remember

✅ KNN predicts using similar data.

✅ K = Number of nearest neighbors.

✅ Small K may overfit.

✅ Large K may underfit.

✅ Training is fast, prediction is slower.

---

# 📝 Summary

In this chapter you learned:

- What KNN is.
- How it works.
- What K represents.
- When to use it.
- Its advantages and limitations.

---

# ➡ Next Chapter

## Naive Bayes

Decision Tree learns rules.

Random Forest uses voting.

SVM finds the best boundary.

KNN finds similar neighbors.

But what if we predicted using **probability**?

That is exactly how **Naive Bayes** works.