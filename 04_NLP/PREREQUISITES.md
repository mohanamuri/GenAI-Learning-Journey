# Prerequisites — Module 04 NLP

## No API Keys — Runs Locally

All examples use NLTK, spaCy, scikit-learn, and sentence-transformers.
No cloud services, no API keys. Models download once and cache locally.

---

## Install

```bash
pip install nltk spacy textblob gensim scikit-learn pandas sentence-transformers
```

Verify:
```bash
python -c "import nltk, spacy, textblob, gensim, sklearn, sentence_transformers; print('All good')"
```

---

## One-Time Downloads (Run Once)

```bash
# NLTK data
python -c "
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('vader_lexicon')
"

# spaCy English model (~12 MB)
python -m spacy download en_core_web_sm
```

Downloads are cached at `~/nltk_data` and `~/.local/lib/python3.x/site-packages/spacy/`.
Never need to re-download unless you wipe those directories.

---

## What Each Example Does & How to Run

| Example | What it does | Needs | Run |
|---------|-------------|-------|-----|
| `01_text_cleaning.py` | Lowercase, strip HTML/URLs/punctuation | stdlib only | `python 01_text_cleaning.py` |
| `02_tokenization.py` | Split text into words and sentences | nltk punkt | `python 02_tokenization.py` |
| `03_stopwords.py` | Remove noise words — with sentiment trap warning | nltk stopwords | `python 03_stopwords.py` |
| `04_stemming.py` | Crude root reduction: running → run | nltk | `python 04_stemming.py` |
| `05_lemmatization.py` | Real-word root reduction with POS | nltk wordnet + tagger | `python 05_lemmatization.py` |
| `06_pos_tagging.py` | Label each word as noun/verb/adjective | nltk tagger | `python 06_pos_tagging.py` |
| `07_bag_of_words.py` | Word count vectors with CountVectorizer | scikit-learn, pandas | `python 07_bag_of_words.py` |
| `08_tfidf.py` | Weighted word importance, keyword extraction | scikit-learn, pandas | `python 08_tfidf.py` |
| `09_word2vec.py` | Train word vectors, king−man+woman≈queen | gensim | `python 09_word2vec.py` |
| `10_spacy_basics.py` | Full NLP pipeline: tokens, POS, NER, deps | spacy en_core_web_sm | `python 10_spacy_basics.py` |
| `11_text_classification.py` | Spam detection with TF-IDF + LogisticRegression | scikit-learn | `python 11_text_classification.py` |
| `12_sentiment_analysis.py` | VADER + TextBlob sentiment scoring | nltk vader, textblob | `python 12_sentiment_analysis.py` |
| `13_ner.py` | Named Entity Recognition — people, orgs, dates | spacy en_core_web_sm | `python 13_ner.py` |
| `14_topic_modeling.py` | Discover hidden themes with LDA | scikit-learn | `python 14_topic_modeling.py` |
| `15_text_similarity.py` | Jaccard, TF-IDF cosine, sentence embeddings | scikit-learn, sentence-transformers | `python 15_text_similarity.py` |

---

## Suggested Run Order

**Start here — no downloads needed:**
```bash
python 01_text_cleaning.py
python 07_bag_of_words.py
python 08_tfidf.py
python 11_text_classification.py
python 14_topic_modeling.py
```

**After NLTK downloads:**
```bash
python 02_tokenization.py
python 03_stopwords.py
python 04_stemming.py
python 05_lemmatization.py
python 06_pos_tagging.py
python 12_sentiment_analysis.py
```

**After spaCy download:**
```bash
python 10_spacy_basics.py
python 13_ner.py
```

**Downloads model on first run (~80 MB):**
```bash
python 09_word2vec.py       # trains on toy data locally — no download
python 15_text_similarity.py  # downloads all-MiniLM-L6-v2 on first run
```

---

## Model Downloads (Automatic on First Run)

| Example | Model | Size | Cached at |
|---------|-------|------|-----------|
| `15_text_similarity.py` | `all-MiniLM-L6-v2` | ~80 MB | `~/.cache/huggingface` |

---

## No API Keys — No Cost

Everything runs locally. `sentence-transformers` model downloads once and is reused.
