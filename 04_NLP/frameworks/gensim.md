# Gensim

**What it is:** Library for unsupervised topic modeling and document similarity. Best known for Word2Vec, Doc2Vec, FastText, and LDA implementations.

**Before Gensim / Word2Vec (pre-2013):**
Text was represented as sparse count vectors (BoW, TF-IDF). No way to capture semantic meaning. "car" and "automobile" looked completely different to a model.

**Why we picked it here:**
Word2Vec was the conceptual breakthrough that led to modern embeddings and eventually transformers. Gensim has the cleanest Word2Vec API for learning the concept. Understanding Word2Vec first makes BERT/GPT embeddings much easier to grasp.

**When to use Gensim:**
- Training Word2Vec on your own domain corpus (medical, legal, code)
- Topic modeling with LDA (though sklearn also works)
- Document similarity on large text corpora
- When you want interpretable word vectors

**When NOT to use Gensim:**
- For sentence/paragraph embeddings → use sentence-transformers
- For production semantic search → use Hugging Face + FAISS
- When corpus is small (< 100k words) → Word2Vec won't be meaningful

**Key API:**
```python
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2)
model.wv.most_similar("king")
model.wv.similarity("cat", "dog")
model.wv["word"]   # get vector
```

**Coming up:** In Module 07 (RAG), we'll use proper embedding models (OpenAI, sentence-transformers) that supersede Word2Vec for retrieval tasks.

**Install:** `pip install gensim`

**Introduced in:** Module 04 — NLP (example 09)
