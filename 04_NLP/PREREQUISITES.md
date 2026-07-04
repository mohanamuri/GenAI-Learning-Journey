# Prerequisites — Module 04 NLP

## Install

```bash
pip install nltk spacy textblob gensim scikit-learn pandas sentence-transformers
```

## One-time Downloads

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

## What Each Example Needs

| Example | Packages | Notes |
|---------|----------|-------|
| `01_text_cleaning.py` | `re`, `string` (stdlib) | No install needed |
| `02_tokenization.py` | `nltk` | punkt_tab download |
| `03_stopwords.py` | `nltk` | stopwords download |
| `04_stemming.py` | `nltk` | No extra download |
| `05_lemmatization.py` | `nltk` | wordnet + tagger download |
| `06_pos_tagging.py` | `nltk` | tagger download |
| `07_bag_of_words.py` | `scikit-learn`, `pandas` | — |
| `08_tfidf.py` | `scikit-learn`, `pandas` | — |
| `09_word2vec.py` | `gensim` | Trains on toy data locally |
| `10_spacy_basics.py` | `spacy` | en_core_web_sm download |
| `11_text_classification.py` | `scikit-learn` | — |
| `12_sentiment_analysis.py` | `nltk`, `textblob` | vader_lexicon download |
| `13_ner.py` | `spacy` | en_core_web_sm |
| `14_topic_modeling.py` | `scikit-learn` | — |
| `15_text_similarity.py` | `scikit-learn`, `sentence-transformers` | Downloads `all-MiniLM-L6-v2` (~80 MB) on first run |

## No API Keys Needed

Everything runs locally. `sentence-transformers` downloads `all-MiniLM-L6-v2` (~80 MB)
on first run and caches it at `~/.cache/huggingface`.

## Sanity Check

```bash
python -c "import nltk, spacy, textblob, gensim, sklearn, sentence_transformers; print('All good')"
```

## Suggested Run Order

Start here (no downloads):
```
01 → 02 → 03 → 04 → 07 → 08 → 11
```
Run after downloads complete:
```
05 → 06 → 09 → 10 → 12 → 13 → 14 → 15
```
