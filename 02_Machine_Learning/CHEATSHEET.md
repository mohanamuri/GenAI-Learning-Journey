# Machine Learning - Cheat Sheet

> **Module 02 Quick Revision**

**Author:** Mohan Raju Amuri

---

# What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn patterns from data without being explicitly programmed.

Simply put,

> **Instead of writing rules, we provide data.**

---

# Why Machine Learning?

Traditional Programming

```text
Rules
    +
Data
    │
    ▼
Output
```

Machine Learning

```text
Historical Data
        +
Correct Answers
        │
        ▼
Model Learns
        │
        ▼
Prediction
```

---

# Machine Learning Workflow

```text
Business Problem
        │
        ▼
Collect Data
        │
        ▼
Prepare Data
        │
        ▼
Choose Algorithm
        │
        ▼
Train Model
        │
        ▼
Evaluate Model
        │
        ▼
Deploy Model
        │
        ▼
Monitor & Improve
```

---

# Types of Machine Learning

## Supervised Learning

✔ Has Input Data

✔ Has Correct Answers (Labels)

Examples

- Loan Approval
- Spam Detection
- House Price Prediction

---

## Unsupervised Learning

✔ Has Input Data

❌ No Correct Answers

Examples

- Customer Segmentation
- Market Basket Analysis
- Fraud Pattern Discovery

---

# Classification vs Regression

## Classification

Predict Categories

Examples

```text
Approved / Rejected

Spam / Not Spam

Pass / Fail
```

---

## Regression

Predict Numbers

Examples

```text
Salary

House Price

Temperature

Sales
```

---

# Common Algorithms

| Algorithm | Used For |
|------------|-----------|
| Decision Tree | Classification |
| Logistic Regression | Classification |
| Random Forest | Classification |
| SVM | Classification |
| KNN | Classification |
| Naive Bayes | Classification |
| Linear Regression | Regression |

---

# Scikit-Learn Workflow

```python
model = Algorithm()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

score = model.score(X_test, y_test)
```

Remember

```text
fit()

↓

Learn

predict()

↓

Predict

score()

↓

Evaluate
```

---

# Train-Test Split

```text
Dataset

│

├── Training Data (80%)

└── Testing Data (20%)
```

Train

↓

Learn

Test

↓

Evaluate

---

# Model Evaluation

Common Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation

Never trust Accuracy alone.

---

# Cross Validation

Instead of testing once,

test multiple times.

```text
Fold 1

Fold 2

Fold 3

↓

Average Accuracy
```

More reliable.

---

# Overfitting

```text
Training Accuracy

99%

Testing Accuracy

60%
```

Model memorized data.

---

# Underfitting

```text
Training Accuracy

55%

Testing Accuracy

50%
```

Model failed to learn.

---

# AI Loan Eligibility System

Module 01

```text
Applicant

↓

IF-ELSE Rules

↓

Decision
```

Module 02

```text
Applicant

↓

Decision Tree

↓

Prediction
```

Only the Decision Engine changed.

---

# Choosing an Algorithm

| Problem | Algorithm |
|----------|-----------|
| Pass / Fail | Decision Tree |
| Loan Approval | Decision Tree |
| Spam Detection | Naive Bayes |
| House Price | Linear Regression |
| Salary Prediction | Linear Regression |
| Customer Segmentation | K-Means |

---

# AI Engineer Thinking

Don't ask

> Which algorithm is best?

Ask

> Which algorithm is suitable for this problem?

---

# Interview Tips

Remember

✔ Machine Learning learns from data.

✔ Different algorithms solve different problems.

✔ Classification predicts categories.

✔ Regression predicts numbers.

✔ Evaluate before deployment.

---

# Must Remember

✅ ML is a subset of AI.

✅ Data is more important than algorithms.

✅ fit() = Learn

✅ predict() = Predict

✅ score() = Evaluate

✅ Accuracy alone is not enough.

---

# Machine Learning Journey

```text
Machine Learning

↓

Supervised Learning

↓

Classification

↓

Regression

↓

Model Evaluation

↓

Deployment

↓

Monitoring
```

---

**Author**

**Mohan Raju Amuri**