"""
Text Classification
--------------------
One-liner: Assign a label (spam/ham, topic, sentiment) to a piece of text using ML.

Pipeline: text → clean → TF-IDF vectorize → classifier → label

Remember:
- TF-IDF + Logistic Regression is a strong, fast baseline — try this first
- Naive Bayes is fast and works well on small datasets
- Always evaluate with classification_report (precision, recall, F1)
- Use Pipeline to prevent data leakage (fit vectorizer only on train data)

Don't:
- Don't fit the vectorizer on the full dataset before splitting → data leakage
  → fit_transform on train, transform only on test
- Don't skip class imbalance check — 95% majority class inflates accuracy
- Don't jump to deep learning for text classification — baselines often win
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Dataset: spam detection ---
texts = [
    "Win a free iPhone now! Click here to claim your prize",
    "You have been selected for a cash reward of $1000",
    "URGENT: Your account will be suspended. Verify now",
    "Congratulations! You won a lottery. Call immediately",
    "Free gift cards available for limited time only",
    "Hi, are we still meeting tomorrow for the project review?",
    "Please find the attached report for Q3 analysis",
    "The meeting has been rescheduled to 3pm on Friday",
    "Can you review my pull request when you have time?",
    "Reminder: Team standup at 9am tomorrow",
    "Lunch plans for today? I was thinking Thai food",
    "Thanks for the update. I'll review it by EOD",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# --- Model 1: TF-IDF + Logistic Regression ---
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(random_state=42)),
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=["ham", "spam"]))

# --- Model 2: TF-IDF + Naive Bayes ---
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])
nb_pipeline.fit(X_train, y_train)
y_pred_nb = nb_pipeline.predict(X_test)

print("=== Naive Bayes ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}")

# --- Predict new texts ---
print("=== Predictions on new texts ===")
new_texts = [
    "Click here to win a free vacation trip!",
    "Can we sync up this afternoon about the deployment?",
]
predictions = lr_pipeline.predict(new_texts)
labels_map = {0: "HAM", 1: "SPAM"}
for text, pred in zip(new_texts, predictions):
    print(f"  [{labels_map[pred]}] {text}")
