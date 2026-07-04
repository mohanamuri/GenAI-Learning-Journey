# Author: Mohan Raju Amuri
"""
10_rag_evaluation.py — Measure how good your RAG pipeline actually is

What to remember:
- RAG has two failure modes: retrieval fails (wrong chunks) OR generation fails (ignores context)
- Evaluate retrieval and generation separately — fix the right component
- Key retrieval metrics: Precision@k, Recall@k, MRR, NDCG
- Key generation metrics: Faithfulness (grounded in context?), Answer Relevance, Context Recall

The RAGAS framework metrics (explained here from scratch):
  Faithfulness      — does the answer only use information from the retrieved context?
  Answer Relevance  — does the answer actually address the question?
  Context Precision — are the retrieved chunks relevant to the question?
  Context Recall    — does the retrieved context contain the ground truth answer?

What NOT to do:
- Don't evaluate only end-to-end accuracy — you won't know where to fix
- Don't use LLM-as-judge without a reference dataset — biased toward confident wrong answers
- Don't skip evaluation before going to production — RAG quality varies widely by chunk size/overlap

Interview one-liner:
  "RAG evaluation needs two separate loops: retrieval quality (precision/recall) and generation faithfulness."
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import re

print("Loading embedding model for evaluation ...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Ground Truth Dataset ──────────────────────────────────────────────────────
# In production: annotate 50-200 Q&A pairs with expected source chunks
# Format: {question, ground_truth_answer, relevant_chunk_ids}
eval_dataset = [
    {
        "question": "How much does the Starter plan cost?",
        "ground_truth": "The Starter Plan costs $99 per month.",
        "relevant_ids": {0},  # which corpus indices contain the answer
    },
    {
        "question": "What is the uptime SLA for Professional plan?",
        "ground_truth": "The Professional plan has 99.9% uptime SLA.",
        "relevant_ids": {4},
    },
    {
        "question": "Does TechCorp offer a free trial?",
        "ground_truth": "Yes, there is a 14-day free trial with full Professional features.",
        "relevant_ids": {5},
    },
    {
        "question": "What security certifications does TechCorp have?",
        "ground_truth": "TechCorp is SOC 2 Type II certified with AES-256 encryption.",
        "relevant_ids": {3},
    },
    {
        "question": "What support does the Enterprise plan include?",
        "ground_truth": "Enterprise support includes 24/7 phone and Slack with a dedicated engineer.",
        "relevant_ids": {10},
    },
]

corpus = [
    "Starter Plan: $99/month — up to 5 users, 100 GB storage, CPU-only training",            # 0
    "Professional Plan: $499/month — up to 25 users, 1 TB storage, GPU training",            # 1
    "Enterprise Plan: custom pricing — unlimited users, dedicated infrastructure",             # 2
    "SOC 2 Type II certified. AES-256 encryption at rest, TLS 1.3 in transit.",              # 3
    "99.9% uptime SLA for Professional and Enterprise plans. Starter SLA is 99.5%.",         # 4
    "14-day free trial with full Professional features, no credit card required.",            # 5
    "REST API endpoints with auto-scaling and pay-per-request pricing.",                      # 6
    "Edge deployment via ONNX export for offline inference on edge devices.",                 # 7
    "Batch inference pipeline for scheduled or triggered processing.",                        # 8
    "Streaming inference with Kafka integration for real-time data.",                         # 9
    "Enterprise support: 24/7 phone + Slack, dedicated support engineer assigned.",          # 10
    "Starter support: community forum + email with 48-hour response time.",                  # 11
    "NLP models: BERT, GPT-2, T5 and custom transformer architectures.",                     # 12
    "CV models: ResNet, EfficientNet, YOLO for image classification.",                       # 13
    "Python SDK: client = techcorp.Client(api_key='...'); model.predict(data)",              # 14
]

corpus_vecs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 3) -> list[int]:
    """Return top-k corpus indices by cosine similarity."""
    q_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    scores = [float(np.dot(q_vec, c_vec)) for c_vec in corpus_vecs]
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


# ── Metric 1: Precision@k ─────────────────────────────────────────────────────
def precision_at_k(retrieved: list[int], relevant: set[int]) -> float:
    """What fraction of retrieved chunks are actually relevant?"""
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved) if retrieved else 0.0


# ── Metric 2: Recall@k ────────────────────────────────────────────────────────
def recall_at_k(retrieved: list[int], relevant: set[int]) -> float:
    """What fraction of relevant chunks were retrieved?"""
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant) if relevant else 0.0


# ── Metric 3: Mean Reciprocal Rank (MRR) ─────────────────────────────────────
def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """1/rank of the first relevant result. 0 if none found."""
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


# ── Metric 4: Context Faithfulness (Semantic) ─────────────────────────────────
def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
    """
    Approximate faithfulness: does the answer's meaning align with the context?
    Production: use an LLM judge. Here: sentence-level cosine similarity.
    """
    if not context_chunks:
        return 0.0
    answer_vec = model.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0]
    context_text = " ".join(context_chunks)
    ctx_vec = model.encode([context_text], normalize_embeddings=True, show_progress_bar=False)[0]
    return float(np.dot(answer_vec, ctx_vec))


# ── Metric 5: Answer Relevance ────────────────────────────────────────────────
def answer_relevance(question: str, answer: str) -> float:
    """Does the answer address the question? (semantic similarity)"""
    vecs = model.encode([question, answer], normalize_embeddings=True, show_progress_bar=False)
    return float(np.dot(vecs[0], vecs[1]))


# ── Simulated RAG answers (replace with real LLM output in production) ────────
simulated_answers = {
    "How much does the Starter plan cost?":
        "The Starter plan costs $99 per month and supports up to 5 users.",
    "What is the uptime SLA for Professional plan?":
        "The Professional plan guarantees 99.9% uptime per the SLA.",
    "Does TechCorp offer a free trial?":
        "Yes, TechCorp offers a 14-day free trial with full Professional features and no credit card needed.",
    "What security certifications does TechCorp have?":
        "TechCorp is SOC 2 Type II certified and uses AES-256 encryption.",
    "What support does the Enterprise plan include?":
        "Enterprise support includes 24/7 phone and Slack with a dedicated support engineer.",
}

# A bad answer to demonstrate low faithfulness
bad_answer_example = {
    "What is the uptime SLA for Professional plan?":
        "TechCorp provides excellent support for all customers regardless of plan tier."
}


# ── Full Evaluation Loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Step 10: Evaluation Metrics")
    print("=" * 60)

    k = 3
    results = []

    print(f"\nEvaluating {len(eval_dataset)} questions, top-{k} retrieval\n")
    print(f"  {'Question':<55} P@{k}   R@{k}   MRR   Faith  Relev")
    print(f"  {'─'*55}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}")

    for item in eval_dataset:
        q = item["question"]
        retrieved = retrieve(q, top_k=k)
        rel = item["relevant_ids"]

        p = precision_at_k(retrieved, rel)
        r = recall_at_k(retrieved, rel)
        mrr = reciprocal_rank(retrieved, rel)

        answer = simulated_answers.get(q, "")
        ctx = [corpus[i] for i in retrieved]
        faith = faithfulness_score(answer, ctx)
        relev = answer_relevance(q, answer)

        results.append({"p": p, "r": r, "mrr": mrr, "faith": faith, "relev": relev})
        print(f"  {q[:55]:<55} {p:.2f}   {r:.2f}   {mrr:.2f}   {faith:.2f}   {relev:.2f}")

    # Aggregate
    avg = {k: sum(r[k] for r in results) / len(results) for k in ["p", "r", "mrr", "faith", "relev"]}
    print(f"\n  {'AVERAGE':<55} {avg['p']:.2f}   {avg['r']:.2f}   {avg['mrr']:.2f}   {avg['faith']:.2f}   {avg['relev']:.2f}")

    # Faithfulness of a bad answer
    print("\n" + "=" * 60)
    print("Faithfulness demo — good vs bad answer")
    print("=" * 60)
    q = "What is the uptime SLA for Professional plan?"
    ctx = [corpus[i] for i in retrieve(q)]
    good_faith = faithfulness_score(simulated_answers[q], ctx)
    bad_faith  = faithfulness_score(bad_answer_example[q], ctx)
    print(f"\n  Question: '{q}'")
    print(f"  Context:  '{corpus[4][:70]}'")
    print(f"\n  Good answer: '{simulated_answers[q]}'")
    print(f"    Faithfulness: {good_faith:.3f}")
    print(f"\n  Bad answer:  '{bad_answer_example[q]}'")
    print(f"    Faithfulness: {bad_faith:.3f}")
    print(f"  → Bad answer has lower faithfulness — not grounded in context")

    # Metric interpretation guide
    print("\n" + "=" * 60)
    print("Metric interpretation guide:")
    print("=" * 60)
    guide = [
        ("Precision@k", "Retrieved chunks that are relevant", "> 0.7 = good"),
        ("Recall@k",    "Relevant chunks that were retrieved",  "> 0.8 = good (use k=5-10)"),
        ("MRR",         "Was the best chunk ranked #1?",         "> 0.8 = good"),
        ("Faithfulness","Answer grounded in context",           "> 0.7 = good"),
        ("Relevance",   "Answer addresses the question",         "> 0.7 = good"),
    ]
    for metric, desc, target in guide:
        print(f"  {metric:<15} {desc:<40} {target}")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Measure retrieval and generation separately")
    print("  - Low Recall → increase chunk overlap or chunk_size")
    print("  - Low Precision → add metadata filters or increase MIN_SCORE threshold")
    print("  - Low Faithfulness → LLM is ignoring context (lower temperature, shorter context)")
    print("  - Build an eval set of 50+ Q&A pairs before tuning any RAG parameter")
    print("=" * 60)
