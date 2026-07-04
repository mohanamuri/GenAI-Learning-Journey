# Frameworks — Module 04 NLP

New frameworks introduced in this module. Each file answers: what is it, what existed before, why we picked it, when to use vs skip.

| Framework | File | Why Introduced |
|-----------|------|----------------|
| NLTK | [nltk.md](nltk.md) | Foundational NLP toolkit — tokenization, stemming, POS, corpora |
| spaCy | [spacy.md](spacy.md) | Production NLP pipeline — fast, one-call tokenize+POS+NER |
| Gensim | [gensim.md](gensim.md) | Word2Vec — the origin of modern embeddings |
| TextBlob | [textblob.md](textblob.md) | Quickest sentiment scoring, used in project ensemble |
| sentence-transformers | [sentence_transformers.md](sentence_transformers.md) | Semantic similarity — preview of RAG embeddings (Module 07) |

---

## Framework Evolution in This Repo

```
Module 01  No external ML frameworks (pure Python, rule-based)
Module 02  scikit-learn          ← train/test, algorithms, metrics
Module 03  TensorFlow / Keras    ← neural networks, GPU training
Module 04  NLTK, spaCy, Gensim   ← NLP: text → vectors → meaning
           TextBlob, sentence-transformers
Module 05  Hugging Face Transformers  ← BERT, GPT, tokenizers  [coming]
Module 06  OpenAI SDK, Anthropic SDK  ← calling LLMs via API   [coming]
Module 07  FAISS, ChromaDB            ← vector databases, RAG  [coming]
Module 08  LangGraph, CrewAI          ← agentic workflows       [coming]
Module 09  MLflow, FastAPI, Docker    ← deployment, monitoring  [coming]
```

The evolution is intentional — each module's frameworks build on the previous.
