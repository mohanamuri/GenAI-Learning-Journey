# Gradient Descent

> **Module:** 03 - Deep Learning
>
> **Chapter:** 07
>
> **Difficulty:** ⭐⭐⭐⭐☆
>
> **Reading Time:** 30 Minutes
>
> **Framework:** TensorFlow / Keras
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this chapter, you will understand:

- What Gradient Descent is
- Why it is required
- How Neural Networks reduce error
- Learning Rate
- Local Minimum
- Global Minimum
- How TensorFlow optimizes models

---

# 📚 Prerequisites

You should understand:

- Perceptron
- Activation Functions
- Forward Propagation
- Loss Function

---

# 📖 Overview

The Loss Function tells us how wrong the model is.

But it does not tell us how to improve.

Gradient Descent is the algorithm that improves the model.

Its job is simple:

**Reduce the Loss by updating the model's weights.**

---

# ❓ Why Do We Need Gradient Descent?

Imagine your GPS says:

```
Destination

↓

10 km Away
```

The GPS doesn't just tell you the distance.

It also tells you

```
Turn Left

↓

Go Straight

↓

Turn Right
```

Similarly,

Loss tells us **how wrong** the model is.

Gradient Descent tells us **how to improve** it.

---

# 🌍 Real World Analogy

Imagine standing on top of a hill.

```
          Peak
            ▲
           / \
          /   \
         /     \
        /       \
       ▼
```

Your goal is to reach the lowest point.

You don't jump.

You take one small step at a time.

Eventually,

you reach the bottom.

That is exactly how Gradient Descent works.

---

# 🧠 What is Gradient Descent?

Gradient Descent is an optimization algorithm.

It updates the model's weights so that the Loss becomes smaller.

```
High Loss

↓

Update Weights

↓

Lower Loss

↓

Repeat
```

---

# 🏗 Learning Cycle

```
Prediction

↓

Loss

↓

Gradient Descent

↓

Update Weights

↓

Better Prediction

↓

Smaller Loss
```

This process repeats many times during training.

---

# 📌 Learning Rate

Gradient Descent takes **small steps**.

The size of each step is called the **Learning Rate**.

---

## Small Learning Rate

```
🙂

Small Step

↓

Slow Learning
```

---

## Large Learning Rate

```
🏃

Huge Step

↓

May Skip Best Solution
```

---

## Balanced Learning Rate

```
🚶

Steady Steps

↓

Good Learning
```

---

# Visual Representation

```
High Loss

↓

Step

↓

Lower Loss

↓

Step

↓

Lower Loss

↓

Step

↓

Minimum Loss
```

---

# 🧮 Simple Idea

```
Old Weight

↓

Small Update

↓

New Weight
```

The process continues until the model reaches a good solution.

---

# Local vs Global Minimum

```
Loss

▲

|

|        /\

|       /  \

|  /\  /    \____

|_/  \/           \____

+---------------------------->

```

Small valleys are **Local Minima**.

The lowest valley is the **Global Minimum**.

Gradient Descent tries to reach the best possible solution.

---

# 💻 Python Connection

Imagine

```
Weight

↓

0.40
```

Gradient Descent updates it.

```
0.40

↓

0.38

↓

0.36

↓

0.35
```

Each update attempts to reduce the Loss.

---

# ⚙ TensorFlow Connection

TensorFlow automatically performs Gradient Descent using Optimizers.

Example

```python
optimizer = tf.keras.optimizers.SGD()
```

Later we will also learn:

- Adam
- RMSprop
- Adagrad

---

# 🏦 Connection to Our Loan Eligibility System

```
Customer Data

↓

Prediction

↓

Loss

↓

Gradient Descent

↓

Better Weights

↓

Better Prediction
```

The model improves after every training iteration.

---

# 🌍 Enterprise Perspective

ChatGPT,

Gemini,

Claude,

Llama

all use optimization algorithms to improve model weights during training.

The models are much larger,

but the learning principle remains the same.

---

# 💡 AI Engineer Insight

Gradient Descent is **not** the Neural Network.

It is the **teacher** that continuously improves the Neural Network.

Without it,

the model would make the same mistakes forever.

---

# ⚠ Common Mistakes

❌ Gradient Descent calculates Loss.

✔ Loss is calculated first.

Gradient Descent uses that Loss.

---

❌ Gradient Descent makes predictions.

✔ Forward Propagation makes predictions.

---

❌ Bigger Learning Rate is always better.

✔ Large Learning Rates may overshoot the optimal solution.

---

# 🎤 Interview Questions

## Q1

What is Gradient Descent?

An optimization algorithm that reduces Loss by updating model weights.

---

## Q2

Why is Gradient Descent required?

To improve model predictions by reducing error.

---

## Q3

What is Learning Rate?

The step size used when updating weights.

---

## Q4

Which TensorFlow component performs Gradient Descent?

Optimizers.

---

## Q5

Does Gradient Descent run during prediction?

No.

It runs during training.

---

# 🧠 Interview in 30 Seconds

```
Gradient Descent

↓

Optimization

↓

Reduce Loss

↓

Update Weights

↓

Better Prediction

↓

Learning Rate

↓

Small Steps
```

---

# ⭐ Must Remember

✅ Gradient Descent reduces Loss.

✅ It updates model weights.

✅ Learning Rate controls the step size.

✅ Forward Propagation predicts.

✅ Loss measures error.

✅ Gradient Descent improves the model.

---

# 📝 Summary

In this chapter you learned:

- Why Gradient Descent exists
- Learning Rate
- Weight updates
- Optimization
- TensorFlow Optimizers
- Enterprise usage

Gradient Descent is the algorithm that teaches a Neural Network how to improve.

---

# 📖 Related Example

➡ Example 08 – Gradient Descent

---

# ➡ Next Chapter

# Backpropagation

Gradient Descent knows that the model made a mistake.

But how does it know **which weights should change**?

The answer is **Backpropagation**.