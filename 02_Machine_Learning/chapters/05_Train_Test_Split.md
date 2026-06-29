# Train-Test Split

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

- Why we split data.
- What Training Data is.
- What Testing Data is.
- Why we should never train on the entire dataset.
- How Train-Test Split works in Scikit-Learn.

---

# 🤔 Why Should You Learn This?

Imagine a teacher gives students the **question paper** before the exam.

On exam day,

every student scores **100%**.

Does that mean they truly learned?

**No.**

They simply memorized the answers.

Machine Learning behaves the same way.

---

# 🧠 Why Can't We Train on the Entire Dataset?

Suppose we have:

```text
1000 Loan Applications
```

If we train using all 1000 records,

the model has already seen every example.

Now ask it to predict one of those same records.

Of course,

it will perform well.

But that does **not** tell us whether it can predict **new customers**.

---

# 📊 Train-Test Split

We divide the dataset into two parts.

```text
Complete Dataset

        │

        ▼

┌─────────────────────────────┐
│                             │
│ 80% Training Data           │
│                             │
├─────────────────────────────┤
│                             │
│ 20% Testing Data            │
│                             │
└─────────────────────────────┘
```

Training Data teaches the model.

Testing Data checks what the model learned.

---

# 🧠 Training Data

Training Data is used for learning.

Python

```python
model.fit(X_train, y_train)
```

Remember

```text
fit()

↓

Learn
```

The model discovers patterns from the training dataset.

---

# 🧠 Testing Data

Testing Data is **never shown** during training.

Python

```python
predictions = model.predict(X_test)
```

This is like a final exam.

The model predicts answers it has never seen before.

---

# 🏦 Our Loan Eligibility Project

Suppose we have

```text
1000 Customers
```

We split them like this.

```text
800 Customers

↓

Training

↓

Model Learns


200 Customers

↓

Testing

↓

Model Predicts
```

This tells us how well the model performs on new applicants.

---

# 📊 Why 80% and 20%?

A common choice is:

```text
80%

Training

20%

Testing
```

Other ratios are also used.

Examples

- 70 : 30
- 75 : 25
- 90 : 10

The choice depends on the dataset size.

---

# Python Example

We already used this in our examples.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Understanding the Parameters

### test_size

```python
test_size=0.2
```

Means

```text
20%

Testing Data
```

The remaining

```text
80%

Training Data
```

---

### random_state

```python
random_state=42
```

This ensures the split remains the same every time you run the program.

Without it,

every execution may create a different split.

This makes debugging and comparing results difficult.

---

# 📊 Visual Example

```text
Dataset

100 Records

│

├── 80 Records

│       ↓

│   Model Learns

│

└── 20 Records

        ↓

 Model Evaluation
```

---

# 🧠 Think Like an AI Engineer

Never ask:

> "How accurate is my model on training data?"

Instead ask:

> "How well does my model perform on unseen data?"

That is what matters in real projects.

---

# 💼 AI Engineer Note

In production,

users never send the same data that the model was trained on.

They send **new data**.

Testing data simulates real-world usage.

That is why Train-Test Split is essential.

---

# ⚠ Common Mistakes

❌ Train using all available data.

✔ Keep some data aside for testing.

---

❌ Evaluate using Training Data.

✔ Evaluate using Testing Data.

---

❌ Ignore random_state.

✔ Use a fixed random_state for reproducible results.

---

# 🎤 Interview Questions

## Q1. Why do we split the dataset?

To evaluate the model on unseen data.

---

## Q2. What is Training Data?

Training Data is used to teach the Machine Learning model.

---

## Q3. What is Testing Data?

Testing Data is used to evaluate the model after training.

---

## Q4. What does test_size=0.2 mean?

20% of the dataset is reserved for testing.

80% is used for training.

---

## Q5. Why do we use random_state=42?

It ensures the dataset is split the same way every time, making results reproducible.

---

# ⭐ Must Remember

✅ Training Data teaches the model.

✅ Testing Data evaluates the model.

✅ Never train on the entire dataset.

✅ test_size defines the testing percentage.

✅ random_state ensures reproducible results.

---

# 📝 Summary

In this chapter you learned:

- Why Train-Test Split is important.
- The difference between Training and Testing data.
- How train_test_split() works.
- Why random_state is useful.
- How our Loan Eligibility project uses Train-Test Split.

---

# ➡ Next Chapter

## Decision Tree

Now that we know how to prepare the dataset,

it's time to learn our first Machine Learning algorithm.

```text
Training Data

↓

Decision Tree

↓

Learns Decision Rules

↓

Prediction
```

In the next chapter, we'll understand how a Decision Tree learns without writing a single IF-ELSE rule.