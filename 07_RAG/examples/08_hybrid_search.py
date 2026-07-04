# Author: Mohan Raju Amuri
"""
08_hybrid_search.py — Combine semantic (vector) + keyword (BM25) search for better recall

What to remember:
- Pure semantic search misses exact keyword matches (product codes, names, IDs)
- Pure keyword search (BM25) misses synonyms and paraphrasing
- Hybrid = semantic + keyword scores fused with Reciprocal Rank Fusion (RRF)
- RRF formula: score = sum(1 / (k + rank_i)) where k=60 is a tuning constant

When hybrid beats pure semantic:
  "GPT-4o pricing"     — exact product name (keyword wins)
  "cost of the plan"   — semantic paraphrase (semantic wins)
  "llama3.2 inference" — model name + concept (hybrid wins)

What NOT to do:
- Don't use only BM25 for RAG — it fails on paraphrasing entirely
- Don't tune alpha (fusion weight) without evaluation data — A/B test it
- Don't skip stop-word handling in BM25 — it inflates scores for common words

Interview one-liner:
  "Hybrid search = semantic + BM25 via RRF — catches both concept matches and exact keywords."
"""

import math
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import numpy as np

# ── Corpus ────────────────────────────────────────────────────────────────────
corpus = [
    "Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",
    "Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",
    "Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",
    "SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",
    "99.9% uptime SLA for Professional and Enterprise plans. Starter SLA is 99.5%.",
    "14-day free trial with full Professional features, no credit card required.",
    "REST API endpoint (auto-scaling, pay-per-request) deployment.",
    "Edge deployment via ONNX export for offline inference on edge devices.",
    "Batch inference pipeline: scheduled or triggered processing.",
    "Streaming inference with Kafka integration for real-time data.",
    "Enterprise support: 24/7 phone + Slack, dedicated support engineer.",
    "Starter support: community forum + email, 48h response time.",
    "Supported NLP: BERT, GPT-2, T5, custom transformer architectures.",
    "Supported CV models: ResNet, EfficientNet, YOLO for image tasks.",
    "Python SDK: import techcorp; client.models.load('model-v1').predict(data)",
    "Data formats: CSV, Parquet, JSON, JPEG/PNG images, SDK connectors.",
    "RBAC access control with SSO support. Audit logs retained 90 days.",
    "Bring-your-own-compute supported on Enterprise plan (custom GPUs).",
]

# ── BM25 Implementation ───────────────────────────────────────────────────────
# BM25 is the gold-standard for keyword search (used in Elasticsearch, Solr)
class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # term frequency saturation
        self.b = b    # length normalization
        self.corpus = corpus
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in self.tokenized) / self.N

        # Build inverted index: term → {doc_id: count}
        self.df: dict[str, int] = {}
        self.tf: list[Counter] = []
        for tokens in self.tokenized:
            tf = Counter(tokens)
            self.tf.append(tf)
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> list[float]:
        """Return BM25 score for each document in the corpus."""
        query_terms = self._tokenize(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = len(self.tokenized[i])
            for term in query_terms:
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                # BM25 scoring formula
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * numerator / denominator
            scores.append(s)
        return scores


# ── Semantic Search ───────────────────────────────────────────────────────────
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_vecs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)

def semantic_scores(query: str) -> list[float]:
    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    return [float(np.dot(q_vec, c_vec)) for c_vec in corpus_vecs]


# ── Reciprocal Rank Fusion (RRF) ──────────────────────────────────────────────
def rrf(rankings: list[list[int]], k: int = 60) -> list[float]:
    """
    Merge multiple ranked lists into a single score.
    rankings: list of lists, each is a ranking of doc indices (best first)
    Returns fused scores for each document.
    """
    n = max(max(r) for r in rankings) + 1
    fused = [0.0] * n
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] += 1.0 / (k + rank + 1)
    return fused


# ── Hybrid Search ─────────────────────────────────────────────────────────────
bm25 = BM25(corpus)

def hybrid_search(query: str, top_k: int = 4, alpha: float = 0.5) -> list[dict]:
    """
    Hybrid search via RRF.
    alpha: not used in pure RRF — both rankings are merged with equal weight.
           For weighted fusion: final = alpha * sem_score + (1-alpha) * bm25_score
    """
    sem = semantic_scores(query)
    kw  = bm25.score(query)

    # Build rankings (sort by score, return doc indices best-first)
    sem_ranking = sorted(range(len(sem)), key=lambda i: sem[i], reverse=True)
    kw_ranking  = sorted(range(len(kw)),  key=lambda i: kw[i],  reverse=True)

    fused = rrf([sem_ranking, kw_ranking])
    top = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:top_k]

    return [{
        "rank": rank + 1,
        "doc": corpus[i],
        "rrf_score": fused[i],
        "sem_score": sem[i],
        "bm25_score": kw[i],
        "sem_rank": sem_ranking.index(i) + 1,
        "bm25_rank": kw_ranking.index(i) + 1,
    } for rank, i in enumerate(top)]


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Step 8: Hybrid Search (Semantic + BM25)")
    print("=" * 60)

    queries = [
        ("Exact keyword query",  "SOC 2 Type II AES-256"),
        ("Paraphrase query",     "how much does the cheapest tier cost"),
        ("Mixed query",          "BERT GPT-2 NLP model support pricing"),
        ("Vague semantic query", "how safe and reliable is the platform"),
    ]

    for label, query in queries:
        print(f"\n{'─'*60}")
        print(f"[{label}]  Query: '{query}'")
        print(f"{'─'*60}")

        results = hybrid_search(query, top_k=3)
        print(f"  {'Rank':<5} {'RRF':>6} {'Sem':>6} {'BM25':>6}  {'SemR':>5} {'KwR':>5}  Document")
        for r in results:
            print(f"  #{r['rank']:<4} {r['rrf_score']:>6.4f} {r['sem_score']:>6.3f} {r['bm25_score']:>6.2f}  "
                  f"  {r['sem_rank']:>3}   {r['bm25_rank']:>3}  {r['doc'][:55]}")

    # Show a case where each method fails the other
    print("\n" + "=" * 60)
    print("Head-to-head: keyword vs semantic vs hybrid")
    print("=" * 60)

    query = "AES encryption and uptime guarantee"
    sem = semantic_scores(query)
    kw  = bm25.score(query)
    hybrid = hybrid_search(query, top_k=5)

    sem_top = sorted(range(len(sem)), key=lambda i: sem[i], reverse=True)[:3]
    kw_top  = sorted(range(len(kw)),  key=lambda i: kw[i],  reverse=True)[:3]

    print(f"\nQuery: '{query}'")
    print("\n  Semantic top-3:")
    for i in sem_top:
        print(f"    {sem[i]:.3f}  {corpus[i][:70]}")

    print("\n  BM25 top-3:")
    for i in kw_top:
        print(f"    {kw[i]:.2f}   {corpus[i][:70]}")

    print("\n  Hybrid (RRF) top-3:")
    for r in hybrid[:3]:
        print(f"    {r['rrf_score']:.4f}  {r['doc'][:70]}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - BM25: great for exact terms (product codes, model names, IDs)")
    print("  - Semantic: great for paraphrasing and meaning-based retrieval")
    print("  - RRF fusion is simple and robust — no score normalization needed")
    print("  - Hybrid typically outperforms either alone by 5-15% on NDCG")
    print("  - k=60 in RRF is a standard default — rarely needs tuning")
    print("=" * 60)
