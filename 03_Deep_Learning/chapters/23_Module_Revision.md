# Module 03 - Deep Learning Revision Guide

> Estimated Revision Time: **20 Minutes**

---

# 1. Deep Learning Workflow

```
Business Problem

↓

Collect Data

↓

Preprocess Data

↓

Build Neural Network

↓

Train Model

↓

Evaluate Model

↓

Save Model

↓

Deploy Model

↓

Prediction

↓

Monitor

↓

Retrain
```

---

# 2. Artificial Neural Network

```
Input Layer

↓

Hidden Layer(s)

↓

Output Layer
```

Every neuron performs:

```
Inputs

↓

Weights

↓

Weighted Sum

↓

Bias

↓

Activation Function

↓

Output
```

---

# 3. Perceptron Formula

```
Output

=

Activation(

Σ(Input × Weight)

+

Bias
)
```

---

# 4. Activation Functions

| Function | Range | Common Usage |
|-----------|-------|--------------|
| ReLU | 0 → ∞ | Hidden Layers |
| Sigmoid | 0 → 1 | Binary Classification |
| Tanh | -1 → 1 | Hidden Layers |
| Softmax | 0 → 1 | Multi-class Classification |

---

# 5. Forward Propagation

```
Input

↓

Hidden Layer

↓

Output Layer

↓

Prediction
```

Purpose

Generate predictions.

---

# 6. Loss Function

Measures prediction error.

Common Loss Functions

```
Binary Crossentropy

Categorical Crossentropy

Mean Squared Error
```

Goal

```
Minimize Loss
```

---

# 7. Gradient Descent

Purpose

```
Reduce Loss

↓

Update Weights
```

Formula

```
New Weight

=

Old Weight

−

Learning Rate × Gradient
```

---

# 8. Backpropagation

```
Prediction

↓

Loss

↓

Gradient

↓

Update Weights
```

Purpose

Improve predictions.

---

# 9. Training Terminology

| Term | Meaning |
|------|---------|
| Epoch | One complete pass through dataset |
| Batch | Small group of samples |
| Iteration | One weight update |

Formula

```
Iterations

=

Dataset Size

/

Batch Size
```

---

# 10. TensorFlow APIs

Create Tensor

```python
tf.constant()
```

Variable

```python
tf.Variable()
```

Random Tensor

```python
tf.random.normal()
```

Shape

```python
tensor.shape
```

Datatype

```python
tensor.dtype
```

NumPy

```python
tensor.numpy()
```

---

# 11. Keras APIs

Sequential Model

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

# 12. Model Evaluation

Important Metrics

| Metric | Purpose |
|----------|----------|
| Accuracy | Overall Performance |
| Precision | Correct Positive Predictions |
| Recall | Find Actual Positives |
| F1 Score | Precision + Recall |
| ROC | Classification Performance |
| AUC | Overall Classifier Quality |

---

# 13. Overfitting vs Underfitting

| Underfitting | Good Fit | Overfitting |
|---------------|----------|-------------|
| Low Train Accuracy | High Accuracy | High Train Accuracy |
| Low Test Accuracy | High Accuracy | Low Test Accuracy |

---

# 14. Regularization

Dropout

```python
Dropout(0.30)
```

L2

```python
kernel_regularizer=tf.keras.regularizers.l2(0.001)
```

Early Stopping

```python
EarlyStopping()
```

Purpose

```
Reduce Overfitting
```

---

# 15. Optimizers

| Optimizer | Usage |
|------------|-------|
| SGD | Simple Models |
| Adam | Default Choice |
| RMSprop | RNN |
| Adagrad | Sparse Data |
| AdamW | Transformers & LLMs |

---

# 16. Deep Learning Architectures

| Architecture | Best For |
|--------------|----------|
| ANN | Tabular Data |
| CNN | Images |
| RNN | Sequential Data |
| LSTM | Long Sequences |
| GRU | Efficient Sequential Learning |
| Transformer | NLP & LLMs |

---

# 17. Production Workflow

Training

```
Dataset

↓

Preprocessing

↓

Train

↓

Evaluate

↓

Save
```

Inference

```
User

↓

Load Model

↓

Prediction
```

Always keep Training and Prediction separate.

---

# 18. AI Engineering Best Practices

✅ Modular Project Structure

✅ Configuration Files

✅ Logging

✅ Testing

✅ Documentation

✅ Model Versioning

✅ Deployment

✅ Monitoring

---

# 19. Our Project

Project

```
AI Loan Eligibility System
```

Features

- Dataset Generation
- Preprocessing
- ANN Model
- TensorFlow
- Keras
- Training
- Evaluation
- Save Model
- Load Model
- Prediction

Future

- FastAPI
- Streamlit
- Docker
- Kubernetes
- CI/CD
- Monitoring

---

# 20. Most Important Interview Questions

1. What is Deep Learning?
2. Explain ANN.
3. Explain Perceptron.
4. What is Forward Propagation?
5. What is Backpropagation?
6. Difference between ANN and CNN?
7. Difference between CNN and RNN?
8. Difference between RNN and LSTM?
9. Why Adam Optimizer?
10. What is Overfitting?
11. What is Dropout?
12. Why save a model?
13. Why separate training and inference?
14. Explain your Deep Learning project.
15. How would you deploy your model?

---

# 21. Memory Tricks

ANN

```
Numbers
```

CNN

```
Images
```

RNN

```
Sequences
```

LSTM

```
Long Memory
```

Transformer

```
Attention
```

---

# 22. Common Mistakes

❌ No preprocessing

❌ Testing on training data

❌ No evaluation

❌ No model saving

❌ Hardcoding configuration

❌ Mixing training and prediction

❌ No documentation

❌ No deployment plan

---

# 23. Module Achievement Checklist

After completing Module 03, you can:

✅ Explain Deep Learning

✅ Build ANN models

✅ Use TensorFlow

✅ Use Keras

✅ Train models

✅ Evaluate models

✅ Save & Load models

✅ Build prediction systems

✅ Explain CNN & RNN basics

✅ Build production-ready project structure

---

# Final Mentor Message

Deep Learning is one of the foundational pillars of modern Artificial Intelligence.

You now understand not only **how to build neural networks**, but also **how to engineer complete AI systems**.

The concepts from this module will be reused throughout the remaining modules on Computer Vision, Natural Language Processing, Transformers, Generative AI, RAG, Agentic AI, and MLOps.

Keep this revision guide handy before interviews or while building future projects.

**Congratulations on completing Module 03!** 🎉