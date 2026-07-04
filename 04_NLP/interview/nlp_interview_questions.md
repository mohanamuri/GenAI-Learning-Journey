# NLP Interview Questions — Quick Reference

## Fundamentals

**Q: What is tokenization? Why not use split()?**
Tokenization splits text into units. `split()` leaves punctuation glued ("world!" not "world"). NLTK/spaCy handles contractions, punctuation, special cases properly.

**Q: Stemming vs Lemmatization — when to use which?**
- Stemming: search engines, IR systems where speed matters, output doesn't need to be readable
- Lemmatization: text fed to models, human-readable output, accuracy matters

**Q: Why is POS tagging important for lemmatization?**
Same word, different POS → different lemma.
- "meeting" (noun) → "meeting"
- "meeting" (verb) → "meet"
Default is noun → wrong results for verbs without POS.

**Q: Why avoid stopword removal in sentiment analysis?**
"not good", "no problem", "never satisfied" — removing "not", "no", "never" flips the sentiment completely.

---

## Representations

**Q: What is Bag of Words? Limitations?**
BoW = word count vector. Limitations: loses word order, no semantics, high dimensionality, sparse.

**Q: TF-IDF formula?**
`TF(t,d) × IDF(t)` where `IDF = log(N / df(t))`. Rare words across docs get higher weight.

**Q: What is data leakage in NLP pipelines?**
Fitting the vectorizer on full data before splitting. Test vocabulary bleeds into training. Fix: use `sklearn.pipeline.Pipeline`.

**Q: Word2Vec CBOW vs Skip-gram?**
- CBOW: predict center word from context → faster, better for frequent words
- Skip-gram: predict context from center → slower, better for rare words

**Q: Static vs contextual embeddings?**
- Static (Word2Vec, GloVe): one fixed vector per word — "bank" is always the same
- Contextual (BERT, GPT): vector depends on surrounding words — "bank" in "river bank" ≠ "bank account"

---

## Applications

**Q: How does spaCy NER work?**
Pre-trained statistical model on annotated text. Returns entity spans with labels (PERSON, ORG, GPE etc.). Custom entities need fine-tuning.

**Q: VADER vs TextBlob vs Transformers for sentiment?**
- VADER: rule-based, fast, social media (slang, caps, emojis)
- TextBlob: simple, quick experiments
- Transformers: highest accuracy, use in production

**Q: What is LDA? How do you choose k?**
Latent Dirichlet Allocation — unsupervised topic discovery. Each doc = mix of topics. Choose k by computing coherence score (c_v) for multiple values.

**Q: Cosine vs Euclidean for text similarity?**
Always cosine. Euclidean is affected by vector magnitude (document length). Cosine only measures direction/angle.

**Q: How to handle OOV (out of vocabulary) in Word2Vec?**
1. Use subword models (FastText) — handles OOV via character n-grams
2. Use sentence-transformers — handles OOV via subword tokenization
3. Filter or use zero vector (not ideal)

---

## System Design

**Q: Design a text classification pipeline.**
```
Input → Clean → Tokenize → Remove Stopwords
     → TF-IDF (fit on train only, via Pipeline)
     → LogisticRegression / RandomForest
     → Evaluate: F1, precision, recall per class
     → Check class imbalance — use stratified split
```

**Q: How would you find the most similar FAQ to a user query?**
Encode FAQs + query with sentence-transformers → compute cosine similarity → return top-k matches.

**Q: What NLP features would you use for a resume screening system?**
TF-IDF for keyword matching, NER for skills/companies/education, sentence embeddings for semantic match to job description, skill extraction via regex + NER.
