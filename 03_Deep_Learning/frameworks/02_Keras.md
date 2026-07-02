# Keras

> **Framework:** Keras
>
> **Module:** 03 - Deep Learning
>
> **Author:** Mohan Raju Amuri

---

# 📖 What is Keras?

Keras is a **high-level Deep Learning API** that makes it easy to build, train and deploy Neural Networks.

Today, Keras is fully integrated into TensorFlow and is accessed using:

```python
import tensorflow as tf

model = tf.keras.Sequential()
```

Instead of writing complex mathematical equations, Keras allows us to build Neural Networks using simple Python code.

---

# 🎯 Why was Keras Created?

Building Neural Networks from scratch requires writing:

- Matrix Multiplication
- Weight Initialization
- Forward Propagation
- Backpropagation
- Gradient Descent

This is time-consuming and difficult.

Keras was created to simplify Deep Learning development.

It lets developers focus on solving business problems rather than implementing mathematical details.

---

# 📅 History

| Year | Event |
|------|-------|
| 2015 | Keras created by François Chollet |
| 2017 | Became one of the most popular Deep Learning APIs |
| TensorFlow 2.x | Keras integrated as `tf.keras` |

---

# 👨‍💻 Developed By

**François Chollet**

Later, Google integrated Keras into TensorFlow.

Today, Keras is the official high-level API of TensorFlow.

---

# 🚀 Why Are We Using Keras?

Without Keras,

developers must manually implement Neural Networks.

With Keras,

we simply describe the model.

Example:

```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(64))

model.add(tf.keras.layers.Dense(1))
```

Keras automatically handles much of the underlying complexity.

---

# 🏗 Keras Workflow

```text
Create Model

↓

Add Layers

↓

Compile Model

↓

Train Model

↓

Evaluate Model

↓

Predict
```

This is the standard workflow used in most Deep Learning projects.

---

# 📂 Important Keras Modules

```text
tf.keras

│

├── models

├── layers

├── optimizers

├── losses

├── metrics

├── callbacks

└── datasets
```

---

# ⭐ Most Important Classes

| Class | Purpose |
|--------|---------|
| Sequential | Build models layer by layer |
| Model | Base model class |
| Dense | Fully connected layer |
| Input | Define model input |
| Adam | Optimizer |

---

# ⭐ Most Important Methods

| Method | Purpose |
|---------|---------|
| add() | Add a layer |
| compile() | Configure the model |
| fit() | Train the model |
| evaluate() | Test the model |
| predict() | Make predictions |
| save() | Save the trained model |
| summary() | Display model architecture |

---

# 💻 Example

```python
import tensorflow as tf

model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Dense(8)
)

model.add(
    tf.keras.layers.Dense(1)
)

model.summary()
```

Notice how simple it is to create a Neural Network.

---

# 🏦 Where Are We Using Keras?

In our AI Loan Eligibility System,

Keras will build the Neural Network that replaces the Decision Tree model.

```text
User Input

↓

Keras Model

↓

Prediction

↓

Approved / Rejected
```

---

# 🆚 TensorFlow vs Keras

| TensorFlow | Keras |
|-------------|--------|
| Complete Deep Learning Framework | High-Level API |
| Lower-Level Operations | Simplifies Model Building |
| Includes Keras | Built on TensorFlow |
| Powerful and Flexible | Easy to Learn |

Think of it like this:

```text
TensorFlow

↓

Engine

↓

Keras

↓

Driver
```

TensorFlow provides the power.

Keras provides the simplicity.

---

# 🎤 Interview Questions

### Q1. What is Keras?

Keras is a high-level API for building Deep Learning models.

---

### Q2. Who created Keras?

François Chollet.

---

### Q3. What is the relationship between TensorFlow and Keras?

Keras is integrated into TensorFlow as `tf.keras`.

---

### Q4. Why is Keras popular?

Because it simplifies building and training Neural Networks.

---

### Q5. Which method is used to train a model?

```python
fit()
```

---

# ⚠ Common Mistakes

❌ Keras is a replacement for TensorFlow.

✔ Keras is built on top of TensorFlow.

---

❌ Keras performs calculations without TensorFlow.

✔ TensorFlow performs the computations behind the scenes.

---

❌ Keras is only for beginners.

✔ Keras is widely used in both research and production.

---

# 🧠 AI Engineer Memory Box

```text
Framework

Keras

-------------------------

Created By

François Chollet

-------------------------

Integrated Into

TensorFlow

-------------------------

Main Purpose

Build Neural Networks

-------------------------

Most Important Class

Sequential

-------------------------

Most Important Methods

add()

compile()

fit()

evaluate()

predict()

save()

-------------------------

Remember

Model

↓

Layers

↓

Training

↓

Prediction
```

---

# ⭐ Must Remember

✅ Keras is TensorFlow's high-level API.

✅ Use `tf.keras` in modern projects.

✅ Sequential is the simplest model type.

✅ `fit()` trains the model.

✅ `predict()` makes predictions.

---

# 📖 References

- TensorFlow Keras Documentation



# Keras Framework

> **Module:** 03 - Deep Learning
>
> **Framework:** Keras
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

After completing this document, you will understand:

- What Keras is
- Why Keras was created
- Relationship between TensorFlow and Keras
- Why AI Engineers use Keras
- Important classes and functions
- Where Keras is used in real-world AI systems

---

# 📖 What is Keras?

Keras is a high-level Deep Learning framework used to build, train, evaluate, and deploy Neural Networks with minimal code.

Think of Keras as a user-friendly interface built on top of TensorFlow.

Instead of writing hundreds of lines of mathematical operations, Keras allows you to build a Neural Network using only a few lines of code.

---

# ❓ Why was Keras created?

Before Keras, building a Neural Network required developers to manually:

- Define tensors
- Perform matrix multiplication
- Apply activation functions
- Calculate gradients
- Update weights

This made Deep Learning difficult for beginners and time-consuming for developers.

Keras simplified the entire process.

---

# 🏛 History

- **2015** – Keras was created by François Chollet.
- Initially, it supported multiple backends like TensorFlow and Theano.
- **2019** – Keras became the official high-level API of TensorFlow.

Today, when developers say "Keras", they usually mean:

```python
tf.keras
```

---

# 🌍 Real-World Analogy

Imagine you want to build a house.

### Without Keras

You manufacture:

- Bricks
- Cement
- Steel
- Doors
- Windows

Then you build everything manually.

---

### With Keras

You hire an experienced construction company.

You simply provide:

- Number of floors
- Number of rooms
- Design

The construction company handles the details.

Keras plays the same role for Deep Learning.

---

# Relationship Between TensorFlow and Keras

```
                 TensorFlow

------------------------------------------------

Automatic Differentiation

Tensors

Matrix Operations

Optimizers

GPU Support

Model Execution

------------------------------------------------

                    ▲

                    │

                 Keras

------------------------------------------------

Layers

Models

Training

Evaluation

Prediction

Saving Models

------------------------------------------------
```

TensorFlow provides the engine.

Keras provides the easy-to-use interface.

---

# Why AI Engineers Love Keras

Without Keras

```python
Matrix Multiplication

↓

Activation

↓

Loss

↓

Gradient

↓

Optimizer

↓

Weight Update
```

With Keras

```python
model.fit()
```

Keras performs all those internal operations automatically.

---

# Keras Workflow

```
Create Model

↓

Add Layers

↓

Compile Model

↓

Train Model

↓

Evaluate Model

↓

Predict

↓

Save Model
```

This workflow is used in almost every Deep Learning project.

---

# Important Keras Components

| Component | Purpose |
|-----------|----------|
| Sequential | Build Neural Networks layer by layer |
| Dense | Fully Connected Layer |
| Input | Define input shape |
| compile() | Configure the model |
| fit() | Train the model |
| evaluate() | Evaluate model performance |
| predict() | Make predictions |
| save() | Save trained model |

---

# Understanding Sequential

Sequential means:

```
Layer

↓

Layer

↓

Layer

↓

Output
```

Data moves through the layers in sequence.

This is the simplest Neural Network architecture.

---

# Understanding Dense Layer

A Dense Layer means:

Every neuron is connected to every neuron in the previous layer.

Example:

```
Input Layer

○ ○ ○

↓

Dense Layer

○ ○ ○ ○

↓

Output Layer

○
```

Dense layers are the most common type of layer in Deep Learning.

---

# Most Common Activation Functions

| Activation | Usage |
|------------|-------|
| ReLU | Hidden Layers |
| Sigmoid | Binary Classification |
| Softmax | Multi-Class Classification |

---

# Building a Neural Network in Keras

Example

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

This creates a simple Neural Network with:

- One Hidden Layer
- Four Neurons
- One Output Neuron

---

# Compiling the Model

Before training, we configure the model.

```python
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

Here we specify:

- Optimizer
- Loss Function
- Evaluation Metric

---

# Training the Model

Training is simple.

```python
model.fit(X_train, y_train)
```

Internally, Keras performs:

- Forward Propagation
- Loss Calculation
- Backpropagation
- Gradient Descent
- Weight Updates

automatically.

---

# Making Predictions

```python
predictions = model.predict(X_test)
```

Only Forward Propagation happens during prediction.

Weights are not updated.

---

# Saving the Model

```python
model.save("loan_model.keras")
```

The trained model can later be loaded without retraining.

---

# AI Engineer Insight

Keras is not a replacement for TensorFlow.

Instead,

Keras is built on top of TensorFlow.

Understanding TensorFlow helps you understand what Keras is doing internally.

---

# Enterprise Perspective

Keras is widely used for:

- Image Classification
- Text Classification
- Recommendation Systems
- Fraud Detection
- Medical AI
- Speech Recognition
- Time-Series Forecasting
- NLP Applications

Many production AI systems start as Keras models before deployment.

---

# Best Practices

✅ Start with `Sequential` for simple models.

✅ Use meaningful layer sizes.

✅ Choose the correct activation function.

✅ Select the appropriate loss function.

✅ Save trained models.

✅ Track evaluation metrics during training.

---

# Common Mistakes

❌ Using ReLU in the output layer for binary classification.

✔ Use Sigmoid.

---

❌ Calling `predict()` before training.

✔ Train the model first.

---

❌ Thinking `fit()` is a single operation.

✔ It performs multiple internal steps automatically.

---

# Interview Questions

### Q1. What is Keras?

A high-level Deep Learning API built on top of TensorFlow.

---

### Q2. Why do we use Keras?

To build and train Neural Networks with less code.

---

### Q3. What is a Sequential model?

A model where layers are connected one after another.

---

### Q4. What is a Dense layer?

A fully connected neural network layer.

---

### Q5. Which method trains a model?

```python
model.fit()
```

---

### Q6. Which method makes predictions?

```python
model.predict()
```

---

# Must Remember

✅ TensorFlow is the engine.

✅ Keras is the interface.

✅ Sequential builds simple Neural Networks.

✅ Dense means fully connected.

✅ `fit()` trains the model.

✅ `predict()` makes predictions.

✅ `save()` stores the trained model.

---

# Summary

In this document, you learned:

- What Keras is
- Why it was created
- How it relates to TensorFlow
- The Keras workflow
- Key classes and methods
- Enterprise usage

You are now ready to build your first real Neural Network using TensorFlow and Keras.


| Method         | Purpose                                | Remember                 |
| -------------- | -------------------------------------- | ------------------------ |
| `Sequential()` | Creates a layer-by-layer model         | Most common beginner API |
| `Dense()`      | Adds a fully connected layer           | Core building block      |
| `compile()`    | Configures optimizer, loss and metrics | Required before training |
| `summary()`    | Displays model architecture            | Great for debugging      |


| Method       | Purpose                       | Remember                 |
| ------------ | ----------------------------- | ------------------------ |
| `evaluate()` | Measures model performance    | Returns Loss and Metrics |
| `predict()`  | Makes predictions on new data | Used after training      |
