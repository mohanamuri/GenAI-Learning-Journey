# Frameworks — Module 05 Generative AI

New frameworks introduced in this module.

| Framework | File | Why Introduced |
|-----------|------|----------------|
| Hugging Face Transformers | [huggingface_transformers.md](huggingface_transformers.md) | Run BERT, GPT-2, T5 locally — the standard model hub |
| tiktoken | [tiktoken.md](tiktoken.md) | Count tokens accurately for GPT models |
| sentence-transformers | *(introduced in Module 04)* | Semantic embeddings — expanded use here |

---

## Framework Evolution

```
Module 01  Pure Python (rule-based)
Module 02  scikit-learn              ← ML algorithms, pipelines, metrics
Module 03  TensorFlow / Keras        ← neural networks, training loops
Module 04  NLTK, spaCy, Gensim      ← NLP: tokenize, stem, POS, Word2Vec
           TextBlob, sentence-transformers
Module 05  Hugging Face Transformers ← BERT, GPT-2, T5 — pretrained models  ← HERE
           tiktoken                  ← token counting for LLMs
Module 06  OpenAI SDK, Anthropic SDK ← calling cloud LLMs via API           [coming]
Module 07  FAISS, ChromaDB           ← vector databases for RAG              [coming]
Module 08  LangGraph, CrewAI         ← multi-agent orchestration             [coming]
Module 09  MLflow, FastAPI, Docker   ← deployment, monitoring                [coming]
```
