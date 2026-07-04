# Prerequisites — Module 03 Deep Learning

## No API Keys — Runs Locally

All examples use TensorFlow and Keras. No cloud services, no API keys.
Models train on CPU — no GPU required (training will be slower but works fine).

---

## Install

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Verify:
```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

> **Apple Silicon (M1/M2/M3)?** Use the Metal-optimised build:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

---

## What Each Example Does & How to Run

| Example | What it does | Run |
|---------|-------------|-----|
| `01_first_tensor.py` | What a tensor is — the data type of deep learning | `python 01_first_tensor.py` |
| `02_tensor_operations.py` | Math on tensors: add, multiply, reshape, slice | `python 02_tensor_operations.py` |
| `03_first_neuron.py` | A single neuron: weights × input + bias | `python 03_first_neuron.py` |
| `04_perceptron.py` | The simplest neural network — one layer | `python 04_perceptron.py` |
| `05_activation_functions.py` | ReLU, sigmoid, tanh — why non-linearity matters | `python 05_activation_functions.py` |
| `06_forward_propagation.py` | How data flows through layers | `python 06_forward_propagation.py` |
| `07_loss_function.py` | MSE, cross-entropy — how the model measures mistakes | `python 07_loss_function.py` |
| `08_gradient_descent.py` | How the model learns — minimising loss step by step | `python 08_gradient_descent.py` |
| `09_first_neural_network.py` | Build and train your first network with Keras | `python 09_first_neural_network.py` |
| `10_model_training.py` | Training loop, epochs, batch size, callbacks | `python 10_model_training.py` |
| `11_model_evaluation.py` | Evaluate on test set — accuracy, loss, confusion matrix | `python 11_model_evaluation.py` |
| `12_save_load_model.py` | Save trained model to disk, reload and predict | `python 12_save_load_model.py` |

---

## Suggested Run Order

**Foundation (understand building blocks first):**
```bash
python 01_first_tensor.py
python 02_tensor_operations.py
python 03_first_neuron.py
python 04_perceptron.py
python 05_activation_functions.py
```

**Training mechanics:**
```bash
python 06_forward_propagation.py
python 07_loss_function.py
python 08_gradient_descent.py
```

**Full network:**
```bash
python 09_first_neural_network.py
python 10_model_training.py
python 11_model_evaluation.py
python 12_save_load_model.py
```

---

## Training Time (CPU)

| Example | Approx time on CPU |
|---------|-------------------|
| `03_first_neuron.py` | Instant |
| `09_first_neural_network.py` | ~10–30 seconds |
| `10_model_training.py` | ~30–60 seconds |
| `11_model_evaluation.py` | ~30 seconds |

---

## Run All Examples

```bash
cd 03_Deep_Learning/examples/

for f in $(ls *.py | grep -v ".keras" | sort); do
    echo "=== $f ==="; python "$f"; echo
done
```

---

## No API Keys — No Cost

Everything trains locally. The `.keras` file in this folder is a pre-trained model
saved by `12_save_load_model.py` — safe to delete and regenerate.
