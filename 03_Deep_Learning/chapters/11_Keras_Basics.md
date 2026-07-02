# Chapter 11 - Keras Basics

---

# Learning Objectives

After completing this chapter, you will understand:

- What Keras is
- Why Keras exists
- Keras Architecture
- Sequential API
- Functional API
- Layers
- Model Compilation
- Model Training
- Model Evaluation
- Model Saving
- Best Practices

---

# Introduction

Keras is TensorFlow's high-level Deep Learning API.

It allows developers to build neural networks using simple, readable Python code instead of writing complex mathematical equations.

Today, almost every TensorFlow-based Deep Learning application is developed using Keras.

---

# Why Keras?

Without Keras, building a neural network requires manually implementing:

- Matrix multiplication
- Weight initialization
- Forward propagation
- Backpropagation
- Gradient calculation
- Optimizer logic
- Loss computation

Keras hides this complexity behind a clean API.

Instead of hundreds of lines of mathematical code, you can build a neural network in just a few lines.

---

# Keras Architecture

```
Python Application
        │
        ▼
Keras API
        │
        ▼
TensorFlow Backend
        │
        ▼
CPU / GPU
```

Keras focuses on simplicity, while TensorFlow performs the heavy mathematical computations.

---

# Why We Selected Keras

Throughout this AI Engineering Handbook, we will build many AI applications.

We chose Keras because it provides:

- Simple syntax
- Fast development
- Excellent documentation
- Seamless TensorFlow integration
- Production readiness
- Strong community support

For most Deep Learning projects, Keras is the recommended choice.

---

# Sequential API

The Sequential API is the simplest way to build a neural network.

Example

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

Data flows through the layers one after another.

```
Input

↓

Layer 1

↓

Layer 2

↓

Output
```

We used this approach in our Loan Eligibility System.

---

# Functional API

The Functional API is used when models become more complex.

Example

```
Input

├─────────────┐
│             │
▼             ▼
Layer A    Layer B
│             │
└──────┬──────┘
       ▼
    Output
```

Use the Functional API for:

- Multiple inputs
- Multiple outputs
- Residual connections
- Complex architectures

---

# Layers

A neural network consists of layers.

Example

```python
tf.keras.layers.Dense(8)
```

Each layer learns patterns from the previous layer.

Common layer types include:

- Dense
- Conv2D
- MaxPooling2D
- Flatten
- Dropout
- LSTM
- Embedding

We will explore these in later modules.

---

# Activation Functions

Every Dense layer can use an activation function.

Examples

```python
activation="relu"
```

```python
activation="sigmoid"
```

```python
activation="softmax"
```

Activation functions help neural networks learn complex patterns.

---

# Model Compilation

Before training, a model must be compiled.

Example

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Compilation tells TensorFlow:

- Which optimizer to use
- Which loss function to minimize
- Which metrics to monitor

---

# Model Training

Training is performed using:

```python
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16
)
```

During training:

- Forward Propagation
- Loss Calculation
- Backpropagation
- Weight Updates

are executed automatically.

---

# Model Evaluation

Evaluate model performance using:

```python
model.evaluate(
    X_test,
    y_test
)
```

This calculates:

- Loss
- Accuracy

on unseen data.

---

# Prediction

After training, predictions are generated using:

```python
model.predict(data)
```

This is the inference stage.

Our production project uses this method in `predictor.py`.

---

# Save Model

Save a trained model.

```python
model.save("loan_model.keras")
```

The model architecture, weights, and optimizer state are stored.

---

# Load Model

Load an existing model.

```python
tf.keras.models.load_model(
    "loan_model.keras"
)
```

This avoids retraining every time the application starts.

---

# Common Keras APIs

Create Model

```python
tf.keras.Sequential()
```

Input Layer

```python
tf.keras.Input()
```

Dense Layer

```python
tf.keras.layers.Dense()
```

Compile

```python
model.compile()
```

Train

```python
model.fit()
```

Evaluate

```python
model.evaluate()
```

Predict

```python
model.predict()
```

Save

```python
model.save()
```

Load

```python
tf.keras.models.load_model()
```

Summary

```python
model.summary()
```

---

# Best Practices

- Keep models simple initially.
- Normalize input features.
- Save trained models.
- Separate training and prediction.
- Use model.summary() to verify architecture.
- Evaluate before deployment.

---

# Our Production Project

In our Loan Eligibility System:

```
Dataset

↓

Preprocessing

↓

Sequential Model

↓

Compile

↓

Train

↓

Save

↓

Load

↓

Predict
```

Every stage is implemented using Keras.

---

# Summary

Keras is TensorFlow's high-level API for building Deep Learning models.

It simplifies neural network development while leveraging TensorFlow's computational power.

Because of its simplicity and flexibility, Keras has become the standard choice for many Deep Learning applications.

---

# Key Takeaways

- Keras is built on TensorFlow.
- Sequential API is best for simple models.
- Functional API supports advanced architectures.
- Compile before training.
- Save trained models.
- Load models for inference.

---

# Common Mistakes

- Forgetting to compile the model.
- Using the wrong loss function.
- Skipping normalization.
- Retraining instead of loading saved models.
- Ignoring evaluation results.

---

# Interview Questions

1. What is Keras?
2. Why do we use Keras instead of raw TensorFlow?
3. What is the difference between Sequential and Functional APIs?
4. Why must a model be compiled?
5. What does model.fit() do?
6. What does model.predict() return?
7. How do you save a Keras model?
8. How do you load a saved model?
9. What is model.summary() used for?
10. When would you choose the Functional API?

---

# Hands-on Exercise

Modify your project and experiment with:

- Increase hidden neurons from 8 to 16.
- Add one more hidden layer.
- Change activation from ReLU to Tanh.
- Observe changes in accuracy and loss.

Record your observations.

---

# AI Engineering Perspective

Keras is not just a learning library—it is widely used in production AI systems.

Many enterprise applications use Keras models behind REST APIs, containerized services, and cloud deployments.

In this handbook, Keras will remain our primary framework until we introduce PyTorch later in the roadmap.