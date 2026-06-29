# Machine Learning Interview Questions

> **Module:** 02 - Machine Learning
>
> **Reading Time:** 20 Minutes
>
> **Difficulty:** ⭐⭐⭐☆☆
>
> **Author:** Mohan Raju Amuri

---

# 🎯 Goal

This chapter helps you revise the most important Machine Learning interview questions.

These questions are based on the concepts covered throughout Module 02.

---

# Machine Learning Basics

## Q1. What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence that enables computers to learn patterns from historical data without being explicitly programmed.

---

## Q2. Why do we need Machine Learning?

Machine Learning is useful when writing and maintaining manual business rules becomes difficult.

Instead of writing rules, we train a model using historical data.

---

## Q3. What is the difference between AI and Machine Learning?

| Artificial Intelligence | Machine Learning |
|--------------------------|------------------|
| Broad field | Subset of AI |
| Can use rules or learning | Learns from data |
| Includes ML, DL, Agents | One part of AI |

---

# Types of Machine Learning

## Q4. What are the main types of Machine Learning?

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

---

## Q5. What is Supervised Learning?

Learning using input data and correct answers (labels).

Example:

Loan Approval.

---

## Q6. What is Unsupervised Learning?

Learning without labels.

The model discovers hidden patterns.

Example:

Customer Segmentation.

---

# Classification vs Regression

## Q7. What is Classification?

Predicting categories.

Examples:

- Approved / Rejected
- Spam / Not Spam

---

## Q8. What is Regression?

Predicting continuous numerical values.

Examples:

- Salary
- House Price

---

## Q9. Is Loan Approval Classification or Regression?

Classification.

---

## Q10. Is Salary Prediction Classification or Regression?

Regression.

---

# Train-Test Split

## Q11. Why do we split the dataset?

To evaluate the model using unseen data.

---

## Q12. What is Training Data?

Used for learning.

---

## Q13. What is Testing Data?

Used for evaluation.

---

## Q14. What does random_state=42 mean?

It ensures the dataset is split the same way every time.

---

# Decision Tree

## Q15. What is a Decision Tree?

A Machine Learning algorithm that learns decision rules from historical data.

---

## Q16. Why did you choose Decision Tree?

- Easy to understand
- Easy to explain
- Good for Classification
- Suitable for our project

---

## Q17. What is one disadvantage of Decision Tree?

It may overfit.

---

# Linear Regression

## Q18. When do you use Linear Regression?

When predicting numerical values.

---

## Q19. What is R² Score?

It measures how well the regression model fits the data.

---

## Q20. What is a Coefficient?

It shows how much the prediction changes when the input changes.

---

# Logistic Regression

## Q21. Is Logistic Regression a Regression algorithm?

No.

It is a Classification algorithm.

---

## Q22. Why is it called Regression?

It uses a regression equation internally, but predicts categories.

---

# Random Forest

## Q23. What is Random Forest?

A collection of multiple Decision Trees.

---

## Q24. Why is Random Forest better than Decision Tree?

It reduces overfitting and improves stability.

---

# Support Vector Machine

## Q25. What is the goal of SVM?

To find the best decision boundary between classes.

---

# KNN

## Q26. What does K represent?

The number of nearest neighbors considered during prediction.

---

## Q27. Why is KNN called Lazy Learning?

Because most computation happens during prediction instead of training.

---

# Naive Bayes

## Q28. Why is Naive Bayes called "Naive"?

Because it assumes that all features are independent.

---

## Q29. Where is Naive Bayes commonly used?

- Spam Detection
- Email Filtering
- Sentiment Analysis

---

# Model Evaluation

## Q30. What is Accuracy?

The percentage of correct predictions.

---

## Q31. Why is Accuracy alone not enough?

Because it may hide important mistakes, especially with imbalanced datasets.

---

## Q32. What is Precision?

How many predicted positive cases are actually positive.

---

## Q33. What is Recall?

How many actual positive cases were correctly identified.

---

## Q34. What is F1 Score?

The balance between Precision and Recall.

---

## Q35. What is Cross Validation?

Evaluating the model multiple times using different Train-Test splits.

---

# Overfitting vs Underfitting

## Q36. What is Overfitting?

The model memorizes training data and performs poorly on new data.

---

## Q37. What is Underfitting?

The model is too simple to learn the patterns in the data.

---

# Project Questions

## Q38. Explain your AI Loan Eligibility Project.

We built a Machine Learning application that predicts whether a customer is eligible for a loan using a Decision Tree Classifier.

The application:

- Loads historical data.
- Trains the model.
- Saves the model using Joblib.
- Loads the model.
- Accepts user input.
- Predicts loan eligibility.
- Displays a report.

---

## Q39. Why did you use Decision Tree instead of Random Forest?

Decision Tree is:

- Easy to understand.
- Easy to explain.
- Suitable for a beginner Machine Learning project.

Random Forest can be considered as a future improvement.

---

## Q40. Which Python libraries did you use?

- Pandas
- Scikit-Learn
- Joblib

---

# ⭐ Rapid Revision

Remember these points.

```text
AI
↓

Machine Learning
↓

Supervised Learning
↓

Classification
↓

Decision Tree
↓

Train-Test Split
↓

Model Evaluation
↓

Cross Validation
↓

Deployment
```

---

# Interview Tips

✅ Understand the business problem before selecting an algorithm.

✅ Explain concepts using real-world examples.

✅ Do not memorize definitions only.

✅ Relate every algorithm to a practical use case.

✅ Mention your AI Loan Eligibility Project whenever possible.

---

# 📝 Summary

If you can confidently answer these questions, you have a strong foundation in Machine Learning and are well prepared for beginner to intermediate AI Engineer interviews.

---

# Author

**Mohan Raju Amuri**