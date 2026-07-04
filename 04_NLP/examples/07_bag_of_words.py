# Author: Mohan Raju Amuri
"""
Bag of Words (BoW)
------------------
One-liner: Convert text to a word-count vector — no order, no grammar, just counts.

Remember:
- Each column = one word from the vocabulary
- Each row = one document as a count vector
- Vocabulary size = number of unique words across ALL documents
- Use min_df / max_df to filter rare / too-common words
- Output is sparse (most values are 0)

Don't:
- Don't use raw counts for long documents — use TF-IDF instead
- Don't forget that BoW loses word order: "dog bites man" == "man bites dog"
- Don't use BoW for semantic similarity — use embeddings instead
- Don't let vocabulary blow up — always set max_features
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

corpus = [
    "I love machine learning",
    "machine learning is amazing",
    "I love deep learning too",
    "deep learning is a subset of machine learning",
]

# --- Basic BoW ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

vocab = vectorizer.get_feature_names_out()
df = pd.DataFrame(X.toarray(), columns=vocab)
df.index = [f"doc{i+1}" for i in range(len(corpus))]

print("Vocabulary:", list(vocab))
print("\nBoW Matrix:")
print(df.to_string())

# --- Vocabulary size control ---
print("\n--- With max_features=5 ---")
vec_limited = CountVectorizer(max_features=5)
X_lim = vec_limited.fit_transform(corpus)
print("Top 5 vocab:", vec_limited.get_feature_names_out())

# --- Bigrams: capture "machine learning" as one unit ---
print("\n--- Bigrams (n-gram range 1-2) ---")
vec_bigram = CountVectorizer(ngram_range=(1, 2), max_features=10)
X_bi = vec_bigram.fit_transform(corpus)
print("Bigram vocab:", vec_bigram.get_feature_names_out())
# Captures: 'machine learning', 'deep learning', 'love machine' etc.

# --- Similarity using BoW (cosine) ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim_matrix = cosine_similarity(X.toarray())
print("\n--- Document similarity (cosine) ---")
df_sim = pd.DataFrame(sim_matrix.round(2),
                      index=df.index, columns=df.index)
print(df_sim.to_string())

print("\nKey insight: doc1 & doc3 are similar because both have 'I love ... learning'")
