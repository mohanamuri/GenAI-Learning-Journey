# Module 04 — Natural Language Processing (NLP)

> One-liner: Teaching machines to read, understand, and work with human language.

---

## What's in This Module

| File | Concept |
|------|---------|
| [01_text_cleaning.py](examples/01_text_cleaning.py) | Lowercase, remove HTML/URLs/punctuation |
| [02_tokenization.py](examples/02_tokenization.py) | Word & sentence splitting |
| [03_stopwords.py](examples/03_stopwords.py) | Remove noise words (carefully) |
| [04_stemming.py](examples/04_stemming.py) | Crude root reduction (fast) |
| [05_lemmatization.py](examples/05_lemmatization.py) | Real-word root reduction (accurate) |
| [06_pos_tagging.py](examples/06_pos_tagging.py) | Grammar labels per word |
| [07_bag_of_words.py](examples/07_bag_of_words.py) | Word count vectors |
| [08_tfidf.py](examples/08_tfidf.py) | Weighted word importance |
| [09_word2vec.py](examples/09_word2vec.py) | Word as vector in semantic space |
| [10_spacy_basics.py](examples/10_spacy_basics.py) | Full NLP pipeline, fast |
| [11_text_classification.py](examples/11_text_classification.py) | Label text (spam/ham, topic) |
| [12_sentiment_analysis.py](examples/12_sentiment_analysis.py) | Positive / negative / neutral |
| [13_ner.py](examples/13_ner.py) | Find people, orgs, dates in text |
| [14_topic_modeling.py](examples/14_topic_modeling.py) | Discover themes without labels |
| [15_text_similarity.py](examples/15_text_similarity.py) | Measure how alike two texts are |

---

## NLP Pipeline Order

```
Raw Text
  → Clean (lowercase, remove noise)
  → Tokenize
  → Remove Stopwords  ← skip for sentiment/NER
  → Stem OR Lemmatize
  → Vectorize (BoW / TF-IDF / Embeddings)
  → Model (classify / cluster / search)
```

---

## One-Liners to Remember

- **Tokenization**: `split()` is not tokenization. Use NLTK or spaCy.
- **Stopwords**: Never remove for sentiment analysis — "not good" → "good" is wrong.
- **Stemming**: Fast, crude, produces non-words (`studies → studi`). Use for search indexes.
- **Lemmatization**: Slow, accurate, real words. Use when output must be readable.
- **BoW**: Loses word order. "dog bites man" == "man bites dog".
- **TF-IDF**: Rare words in a doc matter more. Use over BoW when doc lengths vary.
- **Word2Vec**: Static embedding — "bank" (river) = "bank" (money). Use transformers for context.
- **spaCy**: Load model once. Disable unused pipes for speed.
- **Sentiment**: VADER for social media. Transformers for production.
- **NER**: Works out of the box for common entities. Train custom for domain-specific.
- **Topic Modeling**: You pick k (number of topics). You interpret what each topic means.
- **Similarity**: Use cosine, not Euclidean. Use sentence-transformers for semantic search.

---

## What NOT to Do

| Mistake | Why |
|---------|-----|
| `text.split()` instead of `word_tokenize()` | Punctuation stays glued to words |
| Remove stopwords before sentiment analysis | "not" and "no" flip meaning |
| Fit TF-IDF on full dataset before split | Data leakage — inflated metrics |
| Use stemming when output is human-readable | `studi`, `happi` are not real words |
| Skip POS when lemmatizing | `running` (noun) stays `running`, not `run` |
| Load spaCy model inside a loop | Massive performance hit |
| Use Word2Vec on tiny corpus | Needs millions of words to be meaningful |
| Use `accuracy` on imbalanced classes | 95% "ham" classifier looks great, is useless |

---

## Interview Quick-Fire

**Q: Stemming vs Lemmatization?**
Stemming = rule-based chopping, fast, non-words. Lemmatization = dictionary lookup, slower, real words.

**Q: When NOT to remove stopwords?**
Sentiment analysis, NER, question answering — word order and negation words matter.

**Q: TF-IDF vs BoW?**
BoW = raw counts. TF-IDF = weighted by rarity across docs. TF-IDF is almost always better.

**Q: Word2Vec vs Sentence Transformers?**
Word2Vec = static, word-level. Sentence Transformers = contextual, sentence-level. Use transformers for semantic search.

**Q: How to find optimal k in LDA?**
Try multiple k values, compute coherence score (c_v), pick the k with highest score.

**Q: What is data leakage in NLP?**
Fitting vectorizer on full dataset before train/test split — test data influences vocabulary, inflating scores. Always use Pipeline.

---

## Project

[Sentiment Analyzer](project/sentiment_analyzer/) — Multi-source sentiment analysis on product reviews.

---

## Frameworks Introduced

New libraries introduced in this module — what they are, why we picked them, what came before.
See [frameworks/README.md](frameworks/README.md).

| Framework | Purpose |
|-----------|---------|
| NLTK | Foundational NLP — tokenize, stem, POS, corpora |
| spaCy | Production NLP pipeline — fast, one-call |
| Gensim | Word2Vec — origin of modern embeddings |
| TextBlob | Quick sentiment scoring |
| sentence-transformers | Semantic similarity — preview of RAG (Module 07) |

---

## Setup

```bash
pip install nltk spacy textblob gensim scikit-learn pandas sentence-transformers
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('all')"
```
