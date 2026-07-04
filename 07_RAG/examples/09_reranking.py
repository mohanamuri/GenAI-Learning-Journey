# Author: Mohan Raju Amuri
"""
09_reranking.py — Re-score retrieved chunks for better precision before sending to LLM

What to remember:
- First-stage retrieval (vector search) optimizes for RECALL — finds many candidate chunks
- Reranking optimizes for PRECISION — picks the truly relevant ones from candidates
- Cross-encoder rerankers read BOTH query and document together → much more accurate
- Typical pipeline: retrieve top-20 with embedding → rerank → send top-3 to LLM

Why reranking matters:
  Bi-encoder (embedding model): encodes query and doc SEPARATELY — fast, O(N) build-once
  Cross-encoder (reranker):     reads query+doc TOGETHER — slow, but O(k) at search time
  Use bi-encoder for retrieval (top-20), cross-encoder for reranking (top-3)

What NOT to do:
- Don't rerank the entire corpus — rerank only the top-k retrieved candidates
- Don't skip retrieval and only rerank — cross-encoders are too slow for full-corpus scan
- Don't use a reranker that wasn't trained on passage-level similarity (use ms-marco models)

Interview one-liner:
  "Reranking uses a cross-encoder to re-score retrieval candidates — higher precision before LLM."
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# ── Models ────────────────────────────────────────────────────────────────────
# Stage 1: Bi-encoder for fast retrieval (already cached from earlier examples)
print("Loading bi-encoder (retrieval stage) ...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Stage 2: Cross-encoder for reranking — ms-marco-MiniLM-L-6-v2 is standard
# Downloads ~80 MB on first run
print("Loading cross-encoder (reranking stage) ...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
# Note: ms-marco models output raw logits — higher = more relevant (no fixed range)


# ── Knowledge base ────────────────────────────────────────────────────────────
chunks = [
    "Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",
    "Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",
    "Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",
    "SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",
    "99.9% uptime SLA for Professional and Enterprise plans. Starter SLA is 99.5%.",
    "14-day free trial with full Professional features, no credit card required.",
    "REST API endpoints with auto-scaling and pay-per-request pricing.",
    "Edge deployment via ONNX export for offline inference on edge devices.",
    "Batch inference pipeline for scheduled or triggered processing of large datasets.",
    "Streaming inference with Apache Kafka integration for real-time event streams.",
    "Enterprise support: 24/7 phone + Slack, dedicated support engineer assigned.",
    "Starter support: community forum + email with 48-hour response time.",
    "NLP models: BERT, GPT-2, T5 and custom transformer architectures supported.",
    "CV models: ResNet, EfficientNet, YOLO for image classification and detection.",
    "Python SDK quick start: client = techcorp.Client(api_key='...'); model.predict(data)",
    "RBAC with SSO support. Audit logs retained for 90 days.",
    "Free trial: 14-day access with no credit card, full Professional features included.",
    "Bring-your-own-compute supported on Enterprise plan for custom GPU configurations.",
    "Data formats: CSV, Parquet, JSON, JPEG/PNG images, and SDK connectors.",
    "The platform guarantees 99.9% availability measured monthly over rolling 30-day window.",
]

# Pre-compute bi-encoder embeddings (done once at index time in production)
print("Pre-computing bi-encoder embeddings ...")
chunk_vecs = bi_encoder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)


# ── Stage 1: Bi-encoder Retrieval ─────────────────────────────────────────────
def retrieve_candidates(query: str, top_k: int = 10) -> list[dict]:
    """Fast first-stage retrieval using bi-encoder cosine similarity."""
    q_vec = bi_encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = [float(np.dot(q_vec, c_vec)) for c_vec in chunk_vecs]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"text": chunks[i], "bi_score": scores[i], "original_rank": rank + 1}
            for rank, i in enumerate(top_indices)]


# ── Stage 2: Cross-encoder Reranking ─────────────────────────────────────────
def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Rerank candidates using cross-encoder — slower but more precise."""
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False).tolist()

    for candidate, score in zip(candidates, ce_scores):
        candidate["ce_score"] = score

    # Sort by cross-encoder score (higher = more relevant)
    reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)

    for rank, r in enumerate(reranked):
        r["final_rank"] = rank + 1

    return reranked[:top_k]


# ── Full Two-Stage Pipeline ───────────────────────────────────────────────────
def two_stage_retrieve(query: str, retrieve_k: int = 10, final_k: int = 3) -> dict:
    """Retrieve many candidates, rerank to top-k."""
    candidates = retrieve_candidates(query, top_k=retrieve_k)
    final = rerank(query, candidates, top_k=final_k)
    return {"query": query, "candidates": candidates, "final": final}


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG Step 9: Reranking (Two-Stage Retrieval)")
    print("=" * 60)

    queries = [
        "What is the uptime guarantee for enterprise customers?",
        "How much does it cost to get started?",
        "Can I use my own GPU hardware?",
    ]

    for query in queries:
        result = two_stage_retrieve(query, retrieve_k=8, final_k=3)

        print(f"\n{'─'*60}")
        print(f"Query: '{query}'")

        print(f"\n  Stage 1 — Bi-encoder top-8 (retrieval):")
        print(f"  {'Rank':<5} {'BiScore':>8}  Document")
        for c in result["candidates"]:
            print(f"  #{c['original_rank']:<4} {c['bi_score']:>8.4f}  {c['text'][:60]}")

        print(f"\n  Stage 2 — Cross-encoder reranked top-3 (final):")
        print(f"  {'Rank':<5} {'CeScore':>8} {'BiRank':>7}  Document")
        for r in result["final"]:
            moved = r['original_rank'] - r['final_rank']
            arrow = f"↑{moved}" if moved > 0 else (f"↓{abs(moved)}" if moved < 0 else "=")
            print(f"  #{r['final_rank']:<4} {r['ce_score']:>8.3f}  {arrow:>5}    {r['text'][:55]}")

    # Show rank movement — the most interesting part
    print("\n" + "=" * 60)
    print("Rank movement analysis — where reranking made a difference")
    print("=" * 60)
    query = "Is there a free option available?"
    result = two_stage_retrieve(query, retrieve_k=10, final_k=5)
    print(f"\nQuery: '{query}'")
    print(f"\n  Bi-encoder rank  →  Reranker rank  |  Document")
    all_docs = {c["text"]: c["original_rank"] for c in result["candidates"]}
    for r in result["final"]:
        moved = r["original_rank"] - r["final_rank"]
        direction = f"↑ moved up {moved}" if moved > 0 else ("↓ moved down" if moved < 0 else "  same")
        print(f"  BI#{r['original_rank']:<3} → CE#{r['final_rank']:<3}  {direction:<20} {r['text'][:55]}")

    # Performance note
    print("\n" + "=" * 60)
    print("Latency tradeoff:")
    print("  Bi-encoder:   ~5-20ms  for top-k over 1M vectors (batch pre-encoded)")
    print("  Cross-encoder: ~50-200ms for top-20 candidates (runs once at query time)")
    print("  Total (two-stage): ~100-250ms — acceptable for most applications")
    print()
    print("  Rule of thumb: retrieve 10x-20x more than you need, rerank to final top-k")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Two-stage = retrieve wide (bi-encoder), then rerank precise (cross-encoder)")
    print("  - Cross-encoder reads query+doc together — much more nuanced than cosine sim")
    print("  - ms-marco-MiniLM-L-6-v2: ~80 MB, excellent quality-latency balance")
    print("  - Reranking moves relevant docs up even if their embedding score was lower")
    print("  - Always rerank before sending to LLM — reduces hallucination from bad context")
    print("=" * 60)
