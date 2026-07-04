# Prerequisites — Module 02 Machine Learning

## No API Keys — Runs Locally

All examples use scikit-learn, pandas, and numpy. No cloud services, no API keys.

---

## Install

```bash
pip install scikit-learn pandas numpy matplotlib
```

Verify:
```bash
python -c "import sklearn, pandas, numpy; print('All good')"
```

---

## What Each Example Does & How to Run

| Example | What it does | Run |
|---------|-------------|-----|
| `01_learning_from_data.py` | Intuition: how a model learns from examples | `python 01_learning_from_data.py` |
| `02_first_dataset.py` | Load and explore your first dataset with pandas | `python 02_first_dataset.py` |
| `03_simple_prediction.py` | Make a first prediction — no algorithm yet | `python 03_simple_prediction.py` |
| `03_train_test_split.py` | Split data: why training on test data is cheating | `python 03_train_test_split.py` |
| `04_my_first_ml_model.py` | Train your first ML model end-to-end | `python 04_my_first_ml_model.py` |
| `05_first_sklearn_model.py` | scikit-learn fit/predict pattern | `python 05_first_sklearn_model.py` |
| `06_decision_tree_classifier.py` | Decision Tree — most explainable classifier | `python 06_decision_tree_classifier.py` |
| `07_linear_regression.py` | Predict continuous values (price, score) | `python 07_linear_regression.py` |
| `08_logistic_regression.py` | Binary classification despite the name | `python 08_logistic_regression.py` |
| `09_knn_classifier.py` | K-Nearest Neighbors — no training, just distance | `python 09_knn_classifier.py` |
| `10_random_forest.py` | Ensemble of trees — usually beats single tree | `python 10_random_forest.py` |
| `11_support_vector_machine.py` | SVM — finds the widest margin between classes | `python 11_support_vector_machine.py` |
| `12_overfitting_vs_underfitting.py` | The most common ML mistake — visualized | `python 12_overfitting_vs_underfitting.py` |
| `13_confusion_matrix.py` | Beyond accuracy: TP, FP, TN, FN explained | `python 13_confusion_matrix.py` |
| `14_precision_recall_f1.py` | When accuracy lies — imbalanced class problem | `python 14_precision_recall_f1.py` |
| `15_cross_validation.py` | Reliable model evaluation with k-fold CV | `python 15_cross_validation.py` |

---

## Suggested Run Order

**Start here (conceptual foundation):**
```bash
python 01_learning_from_data.py
python 02_first_dataset.py
python 03_train_test_split.py
python 04_my_first_ml_model.py
```

**Core algorithms:**
```bash
python 06_decision_tree_classifier.py
python 07_linear_regression.py
python 08_logistic_regression.py
python 10_random_forest.py
```

**Evaluation (run after algorithms):**
```bash
python 12_overfitting_vs_underfitting.py
python 13_confusion_matrix.py
python 14_precision_recall_f1.py
python 15_cross_validation.py
```

---

## Run All Examples

```bash
cd 02_Machine_Learning/examples/

for f in $(ls *.py | sort); do
    echo "=== $f ==="; python "$f"; echo
done
```

---

## No API Keys — No Cost

Everything runs locally on CPU. No GPU required.
