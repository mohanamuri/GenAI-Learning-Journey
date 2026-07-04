# spaCy

**What it is:** Industrial-strength NLP library. One `nlp(text)` call runs a full pipeline: tokenize → POS tag → dependency parse → NER. Designed for production, not research.

**Before spaCy existed (2015):**
NLTK was the standard — but it was slow, verbose, and required chaining many separate calls. Processing millions of documents was painful. There was no clean pipeline abstraction.

**Why we picked it here:**
spaCy does in one line what takes 10 lines in NLTK, and it's faster. Once you understand NLTK fundamentals (Module 04 examples 01–06), spaCy is how you'd actually build things.

**When to use spaCy:**
- Production NLP pipelines
- NER, dependency parsing, sentence segmentation
- Processing large volumes of text
- When you need a battle-tested, fast pipeline

**When NOT to use spaCy:**
- When you need model training flexibility (use Hugging Face Transformers)
- For sentence embeddings (use sentence-transformers)
- For sentiment analysis (VADER or transformers are better)

**vs NLTK:**
spaCy = production pipeline, fast, opinionated. NLTK = research toolkit, flexible, educational.

**Key concepts:**
```
nlp = spacy.load("en_core_web_sm")   # load once
doc = nlp(text)                       # runs full pipeline
doc.ents        → named entities
token.pos_      → part of speech
token.lemma_    → base form
doc.sents       → sentences
```

**Models:**
- `en_core_web_sm` — fast, no word vectors
- `en_core_web_md` — medium, has vectors
- `en_core_web_lg` — best accuracy, large

**Install:** `pip install spacy && python -m spacy download en_core_web_sm`

**Introduced in:** Module 04 — NLP (example 10)
