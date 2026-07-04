# Author: Mohan Raju Amuri
"""
TF-IDF (Term Frequency - Inverse Document Frequency)
------------------------------------------------------
One-liner: Words rare across documents but frequent in one doc are more important.

Formula:
  TF(t,d)  = count of t in d / total words in d
  IDF(t)   = log(N / df(t))   where N=total docs, df=docs containing t
  TF-IDF   = TF × IDF

Remember:
- High TF-IDF = word is IMPORTANT in this doc and RARE elsewhere
- Low TF-IDF  = word is common everywhere (like "the", "is")
- sklearn's TfidfVectorizer does cleaning + normalization automatically
- Use for: text classification, document retrieval, keyword extraction

Don't:
- Don't use BoW when document lengths vary — TF-IDF handles that
- Don't manually compute TF-IDF — use sklearn's TfidfVectorizer
- Don't expect TF-IDF to understand meaning ("car" ≠ "automobile")
- Don't use on very small corpora — IDF needs enough documents to be meaningful
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

corpus = [
    "Python is great for machine learning",
    "Machine learning requires a lot of data",
    "Python is also great for web development",
    "Data science uses Python and machine learning",
]

# --- Basic TF-IDF ---
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

vocab = tfidf.get_feature_names_out()
df = pd.DataFrame(X.toarray().round(3), columns=vocab)
df.index = [f"doc{i+1}" for i in range(len(corpus))]

print("TF-IDF Matrix (rounded):")
print(df.to_string())

# --- Top keywords per document ---
print("\n--- Top 3 keywords per document ---")
for i, row in enumerate(X.toarray()):
    top_idx = np.argsort(row)[::-1][:3]
    top_words = [(vocab[j], round(row[j], 3)) for j in top_idx if row[j] > 0]
    print(f"  doc{i+1}: {top_words}")

# --- TF-IDF for keyword extraction ---
print("\n--- Keyword extraction from new document ---")
new_doc = ["Natural language processing is a subfield of machine learning and AI"]
tfidf2 = TfidfVectorizer(max_features=20, stop_words="english")
tfidf2.fit(corpus + new_doc)
scores = tfidf2.transform(new_doc).toarray()[0]
vocab2 = tfidf2.get_feature_names_out()
keywords = sorted(zip(vocab2, scores), key=lambda x: x[1], reverse=True)
print("Keywords:", [(w, round(s, 3)) for w, s in keywords if s > 0])

# --- TF-IDF vs BoW comparison ---
print("\n--- Why TF-IDF beats BoW ---")
print("BoW: 'Python' appears 3 times across docs → same weight everywhere")
print("TF-IDF: 'Python' penalized for appearing in many docs → lower weight")
print("       'web development' only in doc3 → higher weight in doc3")
