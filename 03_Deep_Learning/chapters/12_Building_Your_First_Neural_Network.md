# Chapter 12 - Building Your First Neural Network

---

# Learning Objectives

After completing this chapter, you will understand:

- Steps involved in building a Neural Network
- Designing the architecture
- Input Layer
- Hidden Layers
- Output Layer
- Compiling the model
- Training the model
- Evaluating the model
- Saving the model
- Production workflow

---

# Introduction

A Neural Network is not created in a single step.

Building a Deep Learning model is a structured engineering process.

```
Problem

↓

Collect Data

↓

Preprocess Data

↓

Design Neural Network

↓

Compile

↓

Train

↓

Evaluate

↓

Save

↓

Deploy

↓

Prediction
```

This is exactly the workflow we implemented in our AI Loan Eligibility System.

---

# Step 1 - Define the Problem

Every AI project starts with a business problem.

Example

```
Predict

Loan Approval
```

Input

```
Salary

Experience

Credit Score
```

Output

```
Approved

or

Rejected
```

---

# Step 2 - Collect Data

Example Dataset

| Salary | Experience | Credit Score | Approved |
|---------|------------|--------------|----------|
|50000|8|750|1|
|30000|3|600|0|

Our production project generates this dataset automatically.

---

# Step 3 - Preprocess Data

Raw data is rarely suitable for training.

We performed

```
Normalization
```

Example

```
50000

↓

0.50
```

This helps the Neural Network learn efficiently.

---

# Step 4 - Build the Model

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```

This is the model we built in our production project.

---

# Understanding Every Line

---

## Input Layer

```python
tf.keras.Input(shape=(3,))
```

Why 3?

Because our dataset has

- Salary
- Experience
- Credit Score

Three features.

```
3 Inputs

↓

Neural Network
```

---

## Hidden Layer 1

```python
Dense(8, activation="relu")
```

Creates

```
8 Neurons
```

These neurons learn relationships between the input features.

---

## Hidden Layer 2

```python
Dense(4, activation="relu")
```

Further processes the information learned by the first layer.

```
8 Features

↓

4 Higher-Level Features
```

---

## Output Layer

```python
Dense(1, activation="sigmoid")
```

One neuron because we predict only one value.

```
Approved

or

Rejected
```

Sigmoid returns

```
0

↓

1
```

which can be interpreted as probability.

Example

```
0.82

↓

82% Approval Probability
```

---

# Step 5 - Compile

```python
model.compile(

optimizer="adam",

loss="binary_crossentropy",

metrics=["accuracy"]

)
```

Compilation tells TensorFlow

How should the model learn?

Optimizer

```
Adam
```

How should error be calculated?

Loss

```
Binary Crossentropy
```

What should be monitored?

```
Accuracy
```

---

# Step 6 - Train

```python
model.fit(

X_train,

y_train,

epochs=30,

batch_size=16

)
```

During training

```
Forward Propagation

↓

Prediction

↓

Loss

↓

Backpropagation

↓

Weight Update

↓

Repeat
```

---

# Step 7 - Evaluate

```python
model.evaluate(
X_test,
y_test
)
```

Purpose

Measure

```
Loss

Accuracy
```

using unseen data.

---

# Step 8 - Save

```python
model.save(
"loan_eligibility_model.keras"
)
```

This stores

- Architecture
- Weights
- Optimizer State

---

# Step 9 - Load

```python
loaded_model = tf.keras.models.load_model(
"loan_eligibility_model.keras"
)
```

The model is now ready for prediction.

---

# Step 10 - Predict

```python
model.predict(customer)
```

Output

```
0.77
```

Interpretation

```
77%

Probability

↓

Approved
```

---

# Complete Workflow

```
Dataset

↓

Preprocessing

↓

Build Model

↓

Compile

↓

Train

↓

Evaluate

↓

Save

↓

Load

↓

Prediction

↓

Deployment
```

This is the same lifecycle followed by enterprise AI teams.

---

# Mapping to Our Production Project

| Stage | Project File |
|--------|--------------|
| Dataset | dataset_generator.py |
| Load Dataset | dataset.py |
| Preprocessing | preprocessing.py |
| Build Model | train_model.py |
| Train | train.py |
| Save | train_model.py |
| Load | model_loader.py |
| Predict | predictor.py |
| Entry Point | predict.py |

This chapter is no longer theoretical.

Every step exists inside our production project.

---

# Best Practices

- Normalize data
- Keep architecture simple initially
- Save trained models
- Separate training and inference
- Evaluate before deployment
- Use modular code

---

# Summary

Building a Neural Network is a structured engineering process.

A successful AI project combines data preprocessing, model design, training, evaluation, saving, and deployment into one complete workflow.

---

# Key Takeaways

- Build before training.
- Compile before fit().
- Evaluate before deployment.
- Save trained models.
- Load for inference.
- Keep training and prediction separate.

---

# Common Mistakes

- Forgetting Input Layer
- Wrong output neurons
- Wrong activation function
- No preprocessing
- Retraining every prediction
- Skipping evaluation

---

# Interview Questions

1. Describe the complete lifecycle of a Neural Network.
2. Why do we need an Input Layer?
3. Why did we use three input neurons?
4. Why ReLU?
5. Why Sigmoid?
6. Why compile before training?
7. What does model.fit() perform internally?
8. Why save the model?
9. Why separate training and prediction?
10. How does your production project implement this lifecycle?

---

# Hands-on Exercise

Modify your project:

- Change first hidden layer from 8 to 16 neurons.
- Change second hidden layer from 4 to 8 neurons.
- Train again.
- Compare:

- Accuracy
- Loss
- Model Summary

Record your observations.

---

# AI Engineering Perspective

In enterprise AI, building the neural network is only a small part of the system.

A production AI solution also includes:

- Data pipelines
- Configuration
- Logging
- APIs
- Monitoring
- Deployment
- CI/CD
- Model versioning

Our AI Loan Eligibility System is gradually evolving into this architecture, preparing us for deployment and integration into Mohan's AI World.