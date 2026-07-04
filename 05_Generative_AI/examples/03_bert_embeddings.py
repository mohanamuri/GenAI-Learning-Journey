"""
BERT Embeddings
----------------
One-liner: BERT reads the whole sentence bidirectionally — "bank" means different things in different contexts.

Architecture: Encoder-only transformer. Input → Attention over all tokens → Contextual embeddings.

Remember:
- BERT = Bidirectional Encoder Representations from Transformers (2018, Google)
- Pre-trained on masked language modeling + next sentence prediction
- [CLS] token embedding = sentence-level representation (use for classification)
- Best for: classification, NER, embeddings, sentence similarity
- NOT designed for text generation (encoder-only, no decoder)
- Fine-tuning > feature extraction for most downstream tasks

Don't:
- Don't use BERT for text generation — use GPT-style (decoder) models
- Don't average ALL token embeddings blindly — [CLS] is the sentence vector
- Don't forget BERT has a 512-token limit
- Don't use bert-base for production similarity — use sentence-transformers (optimized for it)

Key variants:
  bert-base-uncased    → lowercase input, general purpose
  bert-base-cased      → case-sensitive, better for NER
  distilbert-*         → 40% smaller, 60% faster, 97% of BERT's performance
  roberta-*            → BERT trained better (more data, no NSP task)
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

model_name = "distilbert-base-uncased"  # smaller, faster than bert-base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # inference mode — disables dropout


def get_embedding(text: str) -> np.ndarray:
    """Extract [CLS] token embedding as sentence representation."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.last_hidden_state shape: [batch, seq_len, hidden_size]
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Contextual embeddings: same word, different meaning ───────────────────
print("=== Contextual Embeddings ===")
sentences_bank = [
    "I deposited money at the bank",           # financial bank
    "She sat by the river bank and read",      # river bank
    "The bank approved my loan application",   # financial bank
]
embeddings = [get_embedding(s) for s in sentences_bank]
print(f"Embedding shape: {embeddings[0].shape}")  # (768,)

print("\nSimilarity (same word 'bank', different contexts):")
for i in range(len(sentences_bank)):
    for j in range(i + 1, len(sentences_bank)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  [{sim:.3f}] '{sentences_bank[i][:35]}...' vs '{sentences_bank[j][:35]}...'")
# Financial bank sentences are more similar to each other than to the river bank sentence

# ── Sentence similarity ───────────────────────────────────────────────────
print("\n=== Semantic Similarity ===")
query = "What is machine learning?"
candidates = [
    "Machine learning is a type of artificial intelligence",   # high match
    "Deep learning uses neural networks with many layers",      # medium match
    "Python is a popular programming language",                 # low match
    "I enjoy cooking pasta on weekends",                        # no match
]
query_emb = get_embedding(query)
print(f"Query: {query}\n")
for candidate in candidates:
    emb = get_embedding(candidate)
    sim = cosine_similarity(query_emb, emb)
    bar = "█" * int(sim * 20)
    print(f"  {sim:.3f}  {bar:<20}  {candidate[:50]}")

# ── Token-level embeddings (useful for NER, tagging) ─────────────────────
print("\n=== Token-level embeddings ===")
text = "Apple CEO Tim Cook announced a new product"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
with torch.no_grad():
    outputs = model(**inputs)
hidden = outputs.last_hidden_state[0]  # [seq_len, 768]
print(f"Tokens: {tokens}")
print(f"Each token has a {hidden.shape[1]}-dim vector → use for token classification (NER)")
