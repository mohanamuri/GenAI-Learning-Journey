# Overfitting vs Underfitting

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

- What Overfitting is.
- What Underfitting is.
- Why both are problems.
- How to identify them.
- How to reduce them.

---

# 🤔 Why Should You Learn This?

Imagine a student preparing for an exam.

One student memorizes every question from last year's paper.

Another student studies only two chapters.

Which student will perform better in a new exam?

Probably neither.

Machine Learning models face the same problem.

---

# 🧠 What is Overfitting?

Overfitting happens when a model **memorizes** the training data instead of learning general patterns.

It performs extremely well on training data,

but poorly on new, unseen data.

---

# Overfitting Example

```text
Training Accuracy

99%

↓

Testing Accuracy

68%
```

The model looks excellent during training,

but fails in the real world.

---

# Visual Representation

```text
Training Data

↓

Model Memorizes

↓

Very High Training Accuracy

↓

Poor Testing Accuracy
```

The model remembers the answers instead of learning.

---

# 🧠 What is Underfitting?

Underfitting happens when a model is **too simple** to learn the patterns in the data.

It performs poorly on both training and testing data.

---

# Underfitting Example

```text
Training Accuracy

58%

↓

Testing Accuracy

55%
```

The model failed to learn useful information.

---

# Visual Representation

```text
Training Data

↓

Model Learns Too Little

↓

Low Training Accuracy

↓

Low Testing Accuracy
```

The model never understood the problem.

---

# Comparison

| Overfitting | Underfitting |
|-------------|--------------|
| Memorizes Data | Learns Too Little |
| High Training Accuracy | Low Training Accuracy |
| Poor Testing Accuracy | Poor Testing Accuracy |
| Complex Model | Very Simple Model |

---

# Real World Example

Imagine our AI Loan Eligibility System.

### Overfitting

```text
Training Customers

↓

99% Accuracy

↓

New Customers

↓

70% Accuracy
```

The model memorized past customers.

---

### Underfitting

```text
Training Customers

↓

60% Accuracy

↓

New Customers

↓

58% Accuracy
```

The model never learned enough.

---

# Why Do These Problems Occur?

### Overfitting

- Very complex model.
- Small dataset.
- Too much training.
- Noise in the data.

---

### Underfitting

- Model is too simple.
- Not enough features.
- Insufficient training.
- Important patterns are ignored.

---

# How to Reduce Overfitting?

Common techniques include:

- Collect more training data.
- Use Cross Validation.
- Reduce model complexity.
- Prune Decision Trees.
- Use Random Forest instead of a single Decision Tree.

---

# How to Reduce Underfitting?

Possible solutions include:

- Use a better algorithm.
- Add more useful features.
- Train the model properly.
- Increase model complexity when appropriate.

---

# 🧠 Think Like an AI Engineer

The goal is **not** the highest training accuracy.

The goal is:

```text
Good Training Accuracy

+

Good Testing Accuracy

↓

Good Generalization
```

A model should perform well on **new data**, not just the training data.

---

# 💼 AI Engineer Note

Production systems are evaluated on **real customer data**.

A model with:

```text
99% Training Accuracy
```

may still fail if its testing accuracy is poor.

Always compare training and testing performance.

---

# ⚠ Common Mistakes

❌ Higher Training Accuracy always means a better model.

✔ Not necessarily.

---

❌ Overfitting means the model learned well.

✔ It memorized instead of learning.

---

❌ Underfitting is acceptable.

✔ Underfitting means the model failed to learn the required patterns.

---

# 🎤 Interview Questions

## Q1. What is Overfitting?

Overfitting occurs when a model memorizes the training data and performs poorly on unseen data.

---

## Q2. What is Underfitting?

Underfitting occurs when a model is too simple to learn the patterns in the data.

---

## Q3. How can you reduce Overfitting?

- More training data
- Cross Validation
- Simpler models
- Random Forest
- Pruning

---

## Q4. How can you reduce Underfitting?

- Better algorithms
- More features
- Better training
- Increased model complexity

---

## Q5. Which is worse?

Both are undesirable.

The goal is a model that generalizes well to new data.

---

# ⭐ Must Remember

✅ Overfitting = Memorizes.

✅ Underfitting = Learns Too Little.

✅ Training Accuracy alone is not enough.

✅ Compare Training and Testing performance.

✅ Generalization is the real goal.

---

# 📝 Summary

In this chapter you learned:

- What Overfitting is.
- What Underfitting is.
- Why they occur.
- How to reduce them.
- Why AI Engineers care about generalization.

---

# ➡ Next Chapter

## AI Loan Eligibility Project

In this chapter, we'll review the complete Machine Learning project that we built during this module.

```text
Dataset

↓

Train Model

↓

Save Model

↓

Load Model

↓

Predict Loan Eligibility

↓

Generate Report
```

You'll see how all the Machine Learning concepts come together in one real-world application.