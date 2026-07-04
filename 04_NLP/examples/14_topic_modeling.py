# Author: Mohan Raju Amuri
"""
Topic Modeling (LDA)
---------------------
One-liner: Discover hidden themes in a large collection of documents without labels.

Remember:
- LDA = Latent Dirichlet Allocation — each doc is a mix of topics
- You choose num_topics (k) — no automatic selection, try 5-20
- Each topic = probability distribution over words
- Each document = probability distribution over topics
- Lower perplexity = better model (but coherence score is more useful)
- Use coherence score (c_v) to find optimal k

Don't:
- Don't run LDA on raw text — clean, remove stopwords, lemmatize first
- Don't use a single magic k — always try multiple values
- Don't expect perfectly labeled topics — you interpret them manually
- Don't use LDA for short texts (tweets) — use NMF or BERTopic instead
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# --- Corpus ---
documents = [
    "machine learning algorithms predict outcomes from data",
    "deep neural networks excel at image recognition tasks",
    "python pandas numpy are essential data science tools",
    "convolutional neural networks used in computer vision",
    "random forest gradient boosting are ensemble methods",
    "stock market trading finance investment portfolio returns",
    "federal reserve interest rates inflation economic policy",
    "cryptocurrency bitcoin blockchain digital currency trading",
    "GDP unemployment rate fiscal monetary policy recession",
    "bank loan mortgage interest rate financial planning",
    "football soccer player goal match stadium champion",
    "basketball NBA playoffs tournament team score points",
    "olympic games athlete medal competition world record",
    "tennis wimbledon grand slam tournament prize money",
    "cricket world cup team innings wicket bowler batsman",
]

# --- Preprocessing ---
vectorizer = CountVectorizer(
    stop_words="english",
    max_df=0.9,     # ignore words in >90% of docs
    min_df=2,       # ignore words in <2 docs
    max_features=100,
)
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()

# --- LDA ---
n_topics = 3
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=50,
)
lda.fit(X)

# --- Top words per topic ---
print("=== Topics (top 6 words each) ===\n")
n_top_words = 6
topic_labels = ["Tech/ML", "Finance", "Sports"]  # manually named after seeing words

for idx, topic in enumerate(lda.components_):
    top_word_idx = topic.argsort()[::-1][:n_top_words]
    top_words = [vocab[i] for i in top_word_idx]
    print(f"Topic {idx} [{topic_labels[idx]}]: {', '.join(top_words)}")

# --- Document → Topic assignment ---
print("\n=== Document → Dominant Topic ===")
doc_topics = lda.transform(X)
for i, doc in enumerate(documents):
    dominant = np.argmax(doc_topics[i])
    confidence = doc_topics[i][dominant]
    print(f"  [{topic_labels[dominant]:<10}] ({confidence:.2f})  {doc[:55]}")

# --- Predict topic for new document ---
print("\n=== New document topic ===")
new_doc = ["neural network training epochs loss accuracy validation"]
new_vec = vectorizer.transform(new_doc)
new_topics = lda.transform(new_vec)
dominant = np.argmax(new_topics[0])
print(f"  Topic: {topic_labels[dominant]} (confidence: {new_topics[0][dominant]:.2f})")
