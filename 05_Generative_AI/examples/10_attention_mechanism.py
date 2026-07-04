# Author: Mohan Raju Amuri
"""
Attention Mechanism — The Core of Transformers
-----------------------------------------------
One-liner: Attention lets each token look at every other token and decide how relevant it is — that's why "bank" means differently in different sentences.

The formula: Attention(Q, K, V) = softmax(QK^T / √d_k) × V

  Q = Query  — "what am I looking for?"
  K = Key    — "what do I contain?"
  V = Value  — "what do I give if selected?"
  √d_k       — scaling to prevent vanishing gradients in softmax

Remember:
- Self-attention = Q, K, V all from same sequence
- Multi-head attention = run attention H times in parallel, concat → richer representation
- Positional encoding added to embeddings because attention has no order by default
- Encoder uses bidirectional attention (BERT)
- Decoder uses causal/masked attention — can only see past tokens (GPT)
- Attention is O(n²) in sequence length — why long contexts are expensive

Don't:
- Don't memorise the formula — understand the intuition:
  "each token votes on which other tokens to pay attention to"
- Don't confuse attention with memory — it recomputes every forward pass
- Don't forget positional encoding — without it, "dog bites man" = "man bites dog"
"""

import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: np.ndarray = None):
    """
    Q: [seq_len, d_k]
    K: [seq_len, d_k]
    V: [seq_len, d_v]
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)    # [seq_len, seq_len]
    if mask is not None:
        scores = np.where(mask, scores, -1e9)  # mask future tokens
    weights = softmax(scores)           # attention weights
    output = weights @ V                # [seq_len, d_v]
    return output, weights


np.random.seed(42)

# ── 1. Minimal attention: 4 tokens, dim=8 ─────────────────────────────────
print("=== 1. Self-Attention (4 tokens) ===")
seq_len, d_k = 4, 8
tokens = ["The", "bank", "is", "steep"]

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Attention weights (each row = how much token attends to others):")
print(f"{'':8}", end="")
for t in tokens:
    print(f" {t:>8}", end="")
print()
for i, token in enumerate(tokens):
    print(f"{token:>8}", end="")
    for w in weights[i]:
        print(f" {w:8.3f}", end="")
    print()

# ── 2. Causal mask (GPT-style — can't look ahead) ─────────────────────────
print("\n=== 2. Causal Mask (decoder / GPT-style) ===")
mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
print("Mask (True=visible, False=blocked):")
print(mask.astype(int))

_, causal_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
print("\nCausal attention weights (upper triangle = 0):")
for i, token in enumerate(tokens):
    print(f"  {token:>8}: {[round(w, 3) for w in causal_weights[i]]}")

# ── 3. Attention intuition: "bank" disambiguation ─────────────────────────
print("\n=== 3. Intuition: Context Changes Meaning ===")
print("""
Sentence A: "I went to the bank to deposit money"
  'bank' attends strongly to → 'deposit', 'money'
  Result: financial meaning

Sentence B: "I sat by the river bank and fished"
  'bank' attends strongly to → 'river', 'fished'
  Result: geographical meaning

This context-dependence is WHY transformers outperform Word2Vec:
  Word2Vec: 'bank' always has the same vector
  BERT:     'bank' has a different vector in each context
""")

# ── 4. Multi-head attention (conceptual) ──────────────────────────────────
print("=== 4. Multi-Head Attention ===")
num_heads = 4
d_model = 32
d_k = d_model // num_heads  # 8 per head

print(f"d_model={d_model}, num_heads={num_heads}, d_k per head={d_k}")
print(f"Each head learns different attention patterns:")
print(f"  Head 1 might focus on syntactic relationships (subject → verb)")
print(f"  Head 2 might focus on semantic relationships (noun → adjective)")
print(f"  Head 3 might focus on long-range dependencies")
print(f"  Head 4 might focus on local context")
print(f"Output = concatenate all heads → project back to d_model")
