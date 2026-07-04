# NLTK (Natural Language Toolkit)

**What it is:** The original Python NLP library. Comes with corpora, tokenizers, stemmers, POS taggers, parsers, and 50+ datasets.

**Before NLTK existed:**
Custom regex + string manipulation for every NLP task. No standard tokenizers, no shared corpora. Researchers reinvented the same tools repeatedly.

**Why we picked it here:**
It's the foundational layer. Every NLP concept (tokenization, stemming, stopwords, POS tagging) is easiest to understand through NLTK because the code maps directly to the concept. No abstraction hiding what's happening.

**When to use NLTK:**
- Learning NLP fundamentals
- Quick experiments and prototyping
- Access to linguistic resources (WordNet, Brown corpus, treebanks)
- When you need fine-grained control over classic NLP steps

**When NOT to use NLTK:**
- Production systems (use spaCy — it's 10x faster)
- Deep learning pipelines (use Hugging Face tokenizers)
- When you need pre-trained models

**vs spaCy:**
NLTK = research/learning, modular, slower. spaCy = production, opinionated, faster pipeline.

**Install:** `pip install nltk` then `python -c "import nltk; nltk.download('all')"`

**Introduced in:** Module 04 — NLP (examples 02, 03, 04, 05, 06)
