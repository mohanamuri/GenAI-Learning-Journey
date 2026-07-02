# Chapter 09 - Training Neural Networks

---

# Learning Objectives

After completing this chapter, you will understand:

- What training means in Deep Learning
- Epochs
- Batches
- Iterations
- Forward Propagation
- Backpropagation
- Weight Updates
- Convergence
- Training Workflow
- Best Practices

---

# Introduction

Building a neural network is only the beginning.

A newly created neural network contains **random weights** and therefore makes random predictions.

Training is the process of adjusting these weights so that the neural network learns patterns from historical data and produces accurate predictions.

Without training, even the best neural network architecture is useless.

---

# What Happens During Training?

Training is a repeated learning cycle.

```
Training Data
       │
       ▼
Forward Propagation
       │
       ▼
Prediction
       │
       ▼
Calculate Loss
       │
       ▼
Backpropagation
       │
       ▼
Update Weights
       │
       ▼
Repeat
```

This process continues until the model minimizes its prediction error.

---

# Training Workflow

During training, TensorFlow repeatedly performs the following steps:

1. Read training data.
2. Perform Forward Propagation.
3. Calculate Loss.
4. Perform Backpropagation.
5. Compute Gradients.
6. Update Weights.
7. Repeat until all epochs complete.

---

# Epoch

An **Epoch** means one complete pass through the entire training dataset.

Example

Dataset contains:

```
1000 Records
```

One epoch means

```
Read all 1000 records once.
```

If

```
Epochs = 10
```

The dataset is processed

```
10 times.
```

---

# Batch

Large datasets are not processed all at once.

Instead, they are divided into smaller groups called **batches**.

Example

```
1000 Samples

↓

Batch Size = 100

↓

10 Batches
```

Each batch updates the neural network once.

---

# Iteration

An iteration means processing **one batch**.

Example

```
Dataset = 1000

Batch Size = 100

Iterations per Epoch = 10
```

Formula

```
Iterations

=

Total Samples

/

Batch Size
```

---

# Example

Dataset

```
500 Records
```

Batch Size

```
25
```

Epochs

```
30
```

Iterations

```
500 / 25 = 20
```

Total Updates

```
20 × 30

=

600 Weight Updates
```

This explains why your training output displayed multiple updates for every epoch.

---

# Why Multiple Epochs?

One pass is usually not enough.

Each epoch helps the model learn better.

```
Epoch 1

↓

Poor Prediction

↓

Epoch 10

↓

Better Prediction

↓

Epoch 30

↓

Good Prediction
```

---

# Convergence

Training should continue until the model stabilizes.

This point is called **Convergence**.

Signs of convergence:

- Loss stops decreasing.
- Accuracy stops improving.
- Model becomes stable.

---

# Training in TensorFlow

```python
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16
)
```

TensorFlow performs the entire learning process automatically.

---

# Our Production Project

In our AI Loan Eligibility System:

```
Dataset

↓

Normalize

↓

Train Model

↓

Save Model

↓

Load Model

↓

Prediction
```

This is the same workflow followed in enterprise AI systems.

---

# Best Practices

- Normalize input features.
- Use appropriate batch sizes.
- Monitor loss and accuracy.
- Save trained models.
- Evaluate on unseen data.
- Avoid overtraining.

---

# Summary

Training is the learning phase of a neural network.

During training, the model repeatedly processes the dataset, computes prediction errors, updates its weights, and gradually improves its predictions.

---

# Key Takeaways

- Training adjusts weights.
- Epoch = One complete pass through the dataset.
- Batch = Small subset of data.
- Iteration = One batch processed.
- Multiple epochs improve learning.
- Training stops after convergence.

---

# Common Mistakes

- Using too few epochs.
- Using too many epochs.
- Very large batch sizes.
- Forgetting to normalize data.
- Training and testing on the same dataset.

---

# Interview Questions

1. What is training in Deep Learning?
2. What is an epoch?
3. What is batch size?
4. What is an iteration?
5. What happens during `model.fit()`?
6. Why do we normalize data before training?
7. What is convergence?

---

# Hands-on Exercise

Modify `train.py` and experiment with:

- Epochs = 10
- Epochs = 50
- Batch Size = 8
- Batch Size = 32

Observe:

- Training time
- Accuracy
- Loss

Record your observations.

---

# AI Engineering Perspective

Training is usually performed offline using historical data.

Once training is complete, the trained model is deployed for inference. Production systems rarely retrain the model for every prediction; instead, they load the saved model and serve predictions to users. Separating **training** and **inference** is a fundamental design principle in AI engineering.