# TensorFlow

> **Framework:** TensorFlow
>
> **Module:** 03 - Deep Learning
>
> **Author:** Mohan Raju Amuri

---

# 📖 What is TensorFlow?

TensorFlow is an **open-source Deep Learning framework** developed by **Google**.

It provides tools to:

- Build Neural Networks
- Train AI models
- Evaluate models
- Save and load models
- Deploy AI applications

Think of TensorFlow as the engine that powers Deep Learning.

---

# 🎯 Why was TensorFlow Created?

Before TensorFlow,

developers had to manually write complex mathematical operations such as:

- Matrix Multiplication
- Gradient Calculation
- Weight Updates
- Optimization

As Neural Networks became larger,

this became extremely difficult.

Google introduced TensorFlow to simplify Deep Learning development.

---

# 🚀 Why Are We Using TensorFlow?

Our goal is to build Deep Learning models without manually implementing all mathematical operations.

TensorFlow provides ready-made components for:

- Tensors
- Layers
- Models
- Optimizers
- Loss Functions
- Training

This allows us to focus on solving business problems instead of writing low-level mathematics.

---

# 📅 History

| Year | Event |
|------|-------|
| 2015 | TensorFlow released by Google |
| 2017 | TensorFlow became one of the most popular Deep Learning frameworks |
| Today | Used worldwide in research and production |

---

# 👨‍💻 Developed By

**Google**

TensorFlow was created by the Google Brain team.

---

# 🌍 Where is TensorFlow Used?

TensorFlow is used in:

- Image Recognition
- Object Detection
- Chatbots
- Speech Recognition
- Language Translation
- Medical Diagnosis
- Fraud Detection
- Recommendation Systems

---

# 🏗 TensorFlow Architecture

```text
Python Program

        │

        ▼

TensorFlow

        │

        ▼

Neural Network

        │

        ▼

CPU / GPU / TPU

        │

        ▼

Prediction
```

---

# 📦 Installation

```bash
pip install tensorflow
```

Verify installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

# 📂 Important TensorFlow Modules

```text
tensorflow

│

├── tf.constant()

├── tf.Variable()

├── tf.math

├── tf.random

├── tf.data

├── tf.keras

└── tf.saved_model
```

---

# ⭐ Most Important Classes

| Class | Purpose |
|---------|----------|
| tf.Tensor | Stores data |
| tf.Variable | Trainable parameter |
| tf.keras.Sequential | Build Neural Networks |
| tf.keras.Model | Base model class |
| tf.keras.layers.Dense | Fully connected layer |

---

# ⭐ Most Important Functions

| Function | Purpose |
|----------|----------|
| tf.constant() | Create Tensor |
| tf.Variable() | Create trainable variable |
| tf.matmul() | Matrix multiplication |
| tf.reshape() | Change tensor shape |
| tf.rank() | Tensor dimension |
| tf.size() | Number of elements |
| tf.cast() | Convert data type |

---

# 💻 Example

```python
import tensorflow as tf

numbers = tf.constant([10,20,30])

print(numbers)
```

Output

```text
tf.Tensor([10 20 30], shape=(3,), dtype=int32)
```

---

# 🏦 Where Are We Using TensorFlow?

In Module 03,

TensorFlow will power our:

```text
AI Loan Eligibility System

↓

Neural Network

↓

Prediction
```

TensorFlow replaces Scikit-Learn's Decision Tree model from Module 02.

---

# 🆚 TensorFlow vs NumPy

| NumPy | TensorFlow |
|---------|------------|
| Arrays | Tensors |
| CPU | CPU + GPU + TPU |
| Numerical Computing | Deep Learning |
| No Neural Networks | Neural Networks |

---

# 🎤 Interview Questions

### Q1. What is TensorFlow?

TensorFlow is Google's open-source Deep Learning framework.

---

### Q2. Who developed TensorFlow?

Google.

---

### Q3. What is the basic object in TensorFlow?

Tensor.

---

### Q4. What is TensorFlow mainly used for?

Building and training Neural Networks.

---

### Q5. Does TensorFlow require a GPU?

No.

It works on CPU and GPU.

---

# ⚠ Common Mistakes

❌ TensorFlow replaces Python.

✔ TensorFlow is a Python library.

---

❌ TensorFlow only works on GPU.

✔ It supports CPU, GPU and TPU.

---

❌ TensorFlow is only for Deep Learning researchers.

✔ It is widely used by AI Engineers and Data Scientists.

---

# 🧠 AI Engineer Memory Box

```text
Framework

TensorFlow

-------------------------

Created By

Google

-------------------------

Released

2015

-------------------------

Main Object

Tensor

-------------------------

Main API

tf.keras

-------------------------

Used For

Deep Learning

-------------------------

Remember

Tensor

↓

Neuron

↓

Layer

↓

Model

↓

Training

↓

Prediction
```

---

# ⭐ Must Remember

✅ TensorFlow is Google's Deep Learning framework.

✅ Tensor is the basic data structure.

✅ tf.keras is the most important module.

✅ TensorFlow automates Neural Network training.

✅ TensorFlow powers modern AI applications.

---

# 📖 References

Official Documentation

https://www.tensorflow.org/


| Function        | Purpose               | Must Remember                |
| --------------- | --------------------- | ---------------------------- |
| `tf.constant()` | Create Tensor         | Most frequently used         |
| `tf.matmul()`   | Matrix Multiplication | Used in every Neural Network |
| `.shape`        | Tensor Shape          | Interview question           |
| `.dtype`        | Data Type             | Important for model training |
| `.numpy()`      | Convert to NumPy      | Frequently used              |



| Function          | Purpose            | Remember                            |
| ----------------- | ------------------ | ----------------------------------- |
| `tf.nn.relu()`    | ReLU Activation    | Most common hidden layer activation |
| `tf.nn.sigmoid()` | Sigmoid Activation | Binary Classification               |



| Class                              | Purpose        | Remember               |
| ---------------------------------- | -------------- | ---------------------- |
| `tf.keras.losses.MeanSquaredError` | Calculates MSE | Common Regression Loss |
