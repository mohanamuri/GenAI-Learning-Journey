# Decision Tree

> **Module:** 02 - Machine Learning
>
> **Reading Time:** 15 Minutes
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What a Decision Tree is.
- Why Decision Trees are popular.
- How they learn from data.
- Why we selected Decision Tree for our project.
- When to use and when not to use Decision Trees.

---

# 🤔 Why Should You Learn This?

Imagine you work for a bank.

A customer applies for a loan.

The bank needs to answer one question.

```text
Approve

or

Reject
```

Instead of manually writing hundreds of IF-ELSE rules,

can the computer discover those rules by itself?

Yes.

That is exactly what a Decision Tree does.

---

# 🧠 What is a Decision Tree?

A Decision Tree is a Machine Learning algorithm used mainly for **Classification** and **Regression**.

It learns a series of questions from historical data.

Each question splits the data into smaller groups until a final prediction is reached.

Think of it as asking a sequence of intelligent questions.

---

# 🌳 Why is it called a Tree?

Because its structure looks like an upside-down tree.

```text
                Credit Score >= 700?

                 /             \

              Yes              No

              │                │

     Salary >= 30000?      Reject

         /      \

      Yes        No

      │          │

   Approve    Reject
```

Every question is called a **Node**.

Every branch represents a possible answer.

The last node is called a **Leaf Node**.

---

# 📊 Decision Tree Workflow

```text
Historical Data

        │

        ▼

Find Best Question

        │

        ▼

Split Dataset

        │

        ▼

Repeat

        │

        ▼

Prediction
```

Notice something.

We never wrote these questions manually.

The algorithm discovered them.

---

# 🏦 Our AI Loan Eligibility Project

Module 01

```text
IF Salary >= 30000

AND Credit Score >= 700

↓

Approve
```

Developer wrote the rule.

---

Module 02

```text
Historical Loan Data

↓

Decision Tree

↓

Learns Rules

↓

Approve
```

No IF-ELSE statements were manually written.

---

# 🧠 How Does It Learn?

Imagine this dataset.

| Salary | Credit Score | Loan |
|---------|--------------|------|
| 50000 | 750 | Approved |
| 30000 | 650 | Rejected |
| 70000 | 800 | Approved |
| 25000 | 620 | Rejected |

The algorithm searches for the question that best separates Approved and Rejected customers.

Example

```text
Credit Score >= 700?
```

This creates two groups.

Then it repeats the process until the groups become as pure as possible.

---

# Python Example

We already implemented this.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

Remember

```text
fit()

↓

Learn

predict()

↓

Predict
```

---

# Why Did We Choose Decision Tree?

For our project,

Decision Tree was a good choice because:

- Easy to understand.
- Easy to explain.
- Works well for classification.
- Handles numerical and categorical data.
- Produces decision rules similar to human thinking.

It is an excellent algorithm for beginners.

---

# Advantages

✅ Easy to visualize.

✅ Easy to explain.

✅ No complex mathematics required.

✅ Handles nonlinear relationships.

✅ Fast for small and medium datasets.

---

# Limitations

Decision Trees can:

- Overfit the training data.
- Become very large.
- Change significantly with small changes in data.

These limitations led to the development of **Random Forest**.

---

# 📊 Decision Tree vs Rule-Based AI

| Rule-Based AI | Decision Tree |
|---------------|---------------|
| Humans write rules | Model learns rules |
| Difficult to maintain | Learns automatically |
| Fixed logic | Learns from data |
| Code changes required | Retrain the model |

This is the biggest advantage of Machine Learning.

---

# 🧠 Think Like an AI Engineer

When should you choose Decision Tree?

Choose it when:

- The problem is Classification.
- Explainability is important.
- Business users want understandable decisions.
- Dataset size is small or medium.

---

Avoid Decision Tree when:

- The dataset is extremely large.
- High accuracy is required.
- The model overfits.

In such cases,

Random Forest is often a better choice.

---

# 💼 AI Engineer Note

Banks and healthcare organizations often prefer algorithms that can be explained.

If a customer asks:

> Why was my loan rejected?

A Decision Tree can often provide understandable reasoning.

Explainability is one of its biggest strengths.

---

# ⚠ Common Mistakes

❌ Decision Trees always give the best accuracy.

✔ They are simple and interpretable, but not always the most accurate.

---

❌ Decision Trees cannot handle numerical data.

✔ They handle both numerical and categorical features.

---

❌ Decision Trees never overfit.

✔ Overfitting is one of their main limitations.

---

# 🎤 Interview Questions

## Q1. What is a Decision Tree?

A Decision Tree is a Machine Learning algorithm that learns decision rules from historical data to make predictions.

---

## Q2. Is Decision Tree used for Classification or Regression?

Both.

DecisionTreeClassifier is used for Classification.

DecisionTreeRegressor is used for Regression.

---

## Q3. Why did we use Decision Tree in our AI Loan Eligibility Project?

Because:

- It is easy to understand.
- Easy to explain.
- Suitable for classification.
- Produces human-readable decision rules.

---

## Q4. What is the biggest advantage of a Decision Tree?

Explainability.

It is one of the easiest Machine Learning algorithms to understand.

---

## Q5. What is the biggest disadvantage?

It can overfit the training data.

---

# ⭐ Must Remember

✅ Decision Tree learns rules automatically.

✅ Suitable for Classification and Regression.

✅ Easy to understand.

✅ Easy to explain.

✅ Can overfit.

---

# 📝 Summary

In this chapter you learned:

- What a Decision Tree is.
- How it works.
- Why it is useful.
- Why we selected it for our project.
- Its advantages and limitations.

---

# ➡ Next Chapter

## Linear Regression

Decision Tree predicts categories such as:

```text
Approved

Rejected
```

But what if we need to predict:

```text
Salary

House Price

Temperature
```

These are numbers.

For that, we use **Linear Regression**.
