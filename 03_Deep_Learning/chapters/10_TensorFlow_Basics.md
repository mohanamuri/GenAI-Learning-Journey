# Chapter 10 - TensorFlow Basics

---

# Learning Objectives

After completing this chapter, you will understand:

- What TensorFlow is
- Why TensorFlow is popular
- TensorFlow Architecture
- Tensor
- Variables
- Constants
- Tensor Operations
- Automatic Differentiation
- Eager Execution
- Keras Integration
- Best Practices

---

# Introduction

TensorFlow is Google's open-source Deep Learning framework used to build, train, evaluate, and deploy Artificial Intelligence models.

It provides everything needed to develop AI applications, from simple neural networks to large language models (LLMs).

Today, TensorFlow powers applications in:

- Healthcare
- Banking
- Manufacturing
- Autonomous Vehicles
- Recommendation Systems
- Computer Vision
- Natural Language Processing

---

# Why TensorFlow?

Before TensorFlow, implementing neural networks required manually writing mathematical equations.

TensorFlow automates:

- Matrix calculations
- Gradient computation
- Weight updates
- GPU utilization
- Model serialization
- Deployment

This allows engineers to focus on solving business problems instead of low-level mathematics.

---

# TensorFlow Architecture

```

Python Code

↓

TensorFlow API

↓

Keras API

↓

Tensor Operations

↓

CPU / GPU

↓

Prediction

```

Most developers interact only with the Python and Keras layers.

TensorFlow manages everything underneath.

---

# Core Components

TensorFlow consists of several important building blocks.

```

Tensor

↓

Variable

↓

Operations

↓

Model

↓

Training

↓

Inference

```

---

# Tensor

A Tensor is the fundamental data structure in TensorFlow.

Everything in TensorFlow is represented as a Tensor.

Examples

```
Number

↓

Scalar Tensor
```

```
List

↓

Vector Tensor
```

```
Table

↓

Matrix Tensor
```

```
Multi-dimensional Data

↓

Higher-Dimensional Tensor
```

Example

```python
tensor = tf.constant([10,20,30])
```

Output

```
tf.Tensor([10 20 30], shape=(3,), dtype=int32)
```

---

# Constant

A Constant cannot change after creation.

Example

```python
learning_rate = tf.constant(0.001)
```

Use constants for values that remain fixed.

Examples:

- Learning Rate
- PI
- Mathematical Constants

---

# Variable

Variables can change during training.

Example

```python
weights = tf.Variable([[0.2],[0.5]])
```

Neural network weights are stored as Variables because TensorFlow updates them after every training step.

---

# Tensor Operations

TensorFlow supports mathematical operations directly on tensors.

Examples

Addition

```python
tf.add(a,b)
```

Subtraction

```python
tf.subtract(a,b)
```

Multiplication

```python
tf.multiply(a,b)
```

Matrix Multiplication

```python
tf.matmul(a,b)
```

Transpose

```python
tf.transpose(a)
```

These operations are highly optimized for CPUs and GPUs.

---

# Eager Execution

TensorFlow executes code immediately.

Example

```python
x = tf.constant(10)

print(x)
```

Output appears instantly.

Older TensorFlow versions required building a computational graph before execution.

TensorFlow 2.x enables eager execution by default, making debugging much easier.

---

# Automatic Differentiation

One of TensorFlow's most powerful features is Automatic Differentiation.

Instead of manually computing derivatives,

TensorFlow automatically calculates gradients using:

```
GradientTape
```

Example

```python
with tf.GradientTape() as tape:
    ...
```

This feature is essential for Backpropagation.

---

# Keras Integration

Keras is TensorFlow's high-level API.

Instead of writing thousands of mathematical operations,

you can build neural networks using:

```python
model = tf.keras.Sequential()
```

Keras makes Deep Learning development much faster.

---

# TensorFlow Workflow

```
Dataset

↓

Tensor

↓

Neural Network

↓

Training

↓

Evaluation

↓

Save Model

↓

Prediction

```

This is the workflow we implemented in our production project.

---

# Common TensorFlow APIs

Creating Tensor

```python
tf.constant()
```

Variable

```python
tf.Variable()
```

Random Numbers

```python
tf.random.normal()
```

Tensor Shape

```python
tensor.shape
```

Tensor Type

```python
tensor.dtype
```

Convert Tensor

```python
tensor.numpy()
```

Model

```python
tf.keras.Sequential()
```

Dense Layer

```python
tf.keras.layers.Dense()
```

Training

```python
model.fit()
```

Prediction

```python
model.predict()
```

Evaluation

```python
model.evaluate()
```

Save

```python
model.save()
```

Load

```python
tf.keras.models.load_model()
```

---

# Best Practices

- Keep TensorFlow updated.
- Use TensorFlow 2.x.
- Normalize input data.
- Save trained models.
- Separate training and inference.
- Use GPU when available.
- Use Keras unless low-level TensorFlow is required.

---

# Summary

TensorFlow is a complete Deep Learning ecosystem.

It provides everything required to build, train, evaluate, save, load, and deploy AI models.

TensorFlow handles complex mathematical operations internally, allowing engineers to focus on designing intelligent systems.

---

# Key Takeaways

- TensorFlow is Google's Deep Learning framework.
- Tensor is the basic data structure.
- Variables store trainable parameters.
- Constants never change.
- Keras simplifies TensorFlow.
- Automatic Differentiation enables Backpropagation.
- Eager Execution simplifies debugging.

---

# Common Mistakes

- Confusing Tensor and Variable.
- Using TensorFlow 1.x tutorials.
- Forgetting to normalize data.
- Retraining instead of loading saved models.
- Ignoring model evaluation.

---

# Interview Questions

1. What is TensorFlow?
2. Why is TensorFlow popular?
3. What is a Tensor?
4. Difference between Tensor and Variable?
5. What is Eager Execution?
6. What is Automatic Differentiation?
7. What is GradientTape?
8. Why does TensorFlow use Keras?
9. Difference between TensorFlow and PyTorch?
10. What happens internally during model.fit()?

---

# Hands-on Exercise

Experiment with:

- Create Scalar Tensor
- Create Vector Tensor
- Create Matrix Tensor
- Create Tensor using tf.random.normal()
- Print shape and datatype
- Convert Tensor to NumPy

---

# AI Engineering Perspective

TensorFlow is much more than a Deep Learning library.

It supports model training, inference, deployment, TensorBoard visualization, distributed training, TensorFlow Lite (mobile), TensorFlow Serving (production), and integration with cloud platforms.

As AI Engineers, we typically use TensorFlow for building and training models, then package those models into APIs, containers, and production systems.