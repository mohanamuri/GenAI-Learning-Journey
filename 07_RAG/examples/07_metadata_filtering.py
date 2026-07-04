# Author: Mohan Raju Amuri
"""
07_metadata_filtering.py — Narrow retrieval scope with metadata before vector search

What to remember:
- Metadata filtering = pre-filter the corpus BEFORE vector similarity
- Result: faster search over a smaller, more relevant subset
- Prevents cross-contamination in multi-tenant or multi-document RAG systems
- Example: user asks about "pricing" → filter section=pricing, then search

What NOT to do:
- Don't over-filter — too narrow a filter means no results even when the answer exists
- Don't rely only on metadata — always combine with semantic search
- Don't store everything as metadata — keep metadata searchable/filterable, not full text

Key use cases:
  Multi-tenant RAG    — filter by tenant_id so users only see their data
  Date-range filtering — filter by doc_date for time-sensitive queries
  Section filtering   — filter by section to prevent off-topic retrievals
  Language filtering  — filter by language for multilingual KB

Interview one-liner:
  "Metadata filtering pre-scopes vector search — faster retrieval and tenant isolation."
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Build a richer dataset with metadata ─────────────────────────────────────
# Simulate a multi-product knowledge base with documents from different companies
documents = [
    # TechCorp — various sections
    "TechCorp Starter Plan: $99/month — 5 users, 100 GB storage, CPU training.",
    "TechCorp Professional Plan: $499/month — 25 users, 1 TB storage, GPU training.",
    "TechCorp Enterprise Plan: custom pricing, unlimited users, dedicated infra.",
    "TechCorp uptime SLA: 99.9% for Pro/Enterprise, 99.5% for Starter.",
    "TechCorp security: SOC 2 Type II, AES-256 at rest, TLS 1.3 in transit.",
    "TechCorp free trial: 14 days, full Professional features, no credit card.",
    "TechCorp NLP models: BERT, GPT-2, T5, custom transformer architectures.",
    "TechCorp vision models: ResNet, EfficientNet, YOLO.",
    "TechCorp deployment: REST API, batch pipeline, edge (ONNX), streaming (Kafka).",
    "TechCorp support (Enterprise): 24/7 phone + Slack, dedicated engineer.",
    "TechCorp support (Starter): community forum + email, 48h response.",
    # DataCo — competitor docs (different company)
    "DataCo Basic Plan: $79/month — 3 users, 50 GB storage.",
    "DataCo Enterprise Plan: $999/month — unlimited users, dedicated cluster.",
    "DataCo security: ISO 27001 certified, GDPR compliant.",
    "DataCo uptime SLA: 99.5% across all plans.",
]

metadatas = [
    {"company": "techcorp", "section": "pricing", "plan": "starter", "year": 2024},
    {"company": "techcorp", "section": "pricing", "plan": "professional", "year": 2024},
    {"company": "techcorp", "section": "pricing", "plan": "enterprise", "year": 2024},
    {"company": "techcorp", "section": "sla", "year": 2024},
    {"company": "techcorp", "section": "security", "year": 2024},
    {"company": "techcorp", "section": "faq", "year": 2024},
    {"company": "techcorp", "section": "models", "type": "nlp", "year": 2024},
    {"company": "techcorp", "section": "models", "type": "cv", "year": 2024},
    {"company": "techcorp", "section": "deployment", "year": 2024},
    {"company": "techcorp", "section": "support", "plan": "enterprise", "year": 2024},
    {"company": "techcorp", "section": "support", "plan": "starter", "year": 2024},
    {"company": "dataco",   "section": "pricing", "plan": "basic", "year": 2024},
    {"company": "dataco",   "section": "pricing", "plan": "enterprise", "year": 2024},
    {"company": "dataco",   "section": "security", "year": 2024},
    {"company": "dataco",   "section": "sla", "year": 2024},
]

ids = [f"doc_{i}" for i in range(len(documents))]

# ── Setup ChromaDB ────────────────────────────────────────────────────────────
client = chromadb.EphemeralClient()
ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
col = client.create_collection("kb_with_metadata",
                                embedding_function=ef,
                                metadata={"hnsw:space": "cosine"})
col.add(documents=documents, metadatas=metadatas, ids=ids)


def search(query: str, where: dict = None, n: int = 3, label: str = "") -> None:
    """Run a query and print results."""
    results = col.query(query_texts=[query], n_results=n, where=where)
    print(f"\n  Query: '{query}'")
    if label:
        print(f"  Filter: {label}")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sim = 1.0 - dist / 2.0
        tag = f"[{meta['company']}|{meta['section']}]"
        print(f"    {sim:.3f}  {tag:<30}  {doc[:65]}")


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Step 7: Metadata Filtering")
    print("=" * 60)

    # 1. No filter — returns mixed results from both companies
    print("\n[1] No filter — mixed results from all companies")
    search("What is the enterprise pricing?", n=4, label="none")

    # 2. Filter by company — tenant isolation
    print("\n[2] Company filter — only TechCorp results")
    search("What is the enterprise pricing?",
           where={"company": "techcorp"},
           n=3, label="company=techcorp")

    print("\n[3] Company filter — only DataCo results")
    search("What is the enterprise pricing?",
           where={"company": "dataco"},
           n=3, label="company=dataco")

    # 3. Section filter — only look in pricing
    print("\n[4] Section filter — only pricing chunks")
    search("How much does it cost?",
           where={"section": "pricing"},
           n=4, label="section=pricing")

    # 4. Multi-condition filter with $and
    print("\n[5] Combined filter: TechCorp + pricing section")
    search("What are the plan options?",
           where={"$and": [{"company": "techcorp"}, {"section": "pricing"}]},
           n=3, label="company=techcorp AND section=pricing")

    # 5. $in operator — multiple valid values
    print("\n[6] $in operator: section in [security, sla]")
    search("Is the platform reliable and secure?",
           where={"section": {"$in": ["security", "sla"]}},
           n=4, label="section in [security, sla]")

    # 6. $ne operator — exclude a section
    print("\n[7] $ne (not-equal): exclude pricing section")
    search("Tell me about TechCorp",
           where={"$and": [{"company": "techcorp"}, {"section": {"$ne": "pricing"}}]},
           n=3, label="company=techcorp AND section != pricing")

    # 7. Real-world pattern: intent-based routing
    print("\n" + "=" * 60)
    print("[8] Real-world pattern: Intent-based metadata routing")
    print("=" * 60)

    def intent_based_rag(query: str) -> None:
        """Detect intent and apply matching metadata filter."""
        q_lower = query.lower()
        if any(word in q_lower for word in ["cost", "price", "plan", "pricing", "$"]):
            where = {"section": "pricing"}
            intent = "pricing"
        elif any(word in q_lower for word in ["secure", "security", "encrypt", "certif"]):
            where = {"section": "security"}
            intent = "security"
        elif any(word in q_lower for word in ["support", "help", "contact"]):
            where = {"section": "support"}
            intent = "support"
        else:
            where = None  # no filter — full search
            intent = "general"

        print(f"\n  Query: '{query}'  →  Detected intent: '{intent}'")
        results = col.query(query_texts=[query], n_results=2, where=where)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"    [{meta['company']}|{meta['section']}]  {doc[:70]}")

    intent_based_rag("How much does the starter plan cost?")
    intent_based_rag("Is my data encrypted and GDPR compliant?")
    intent_based_rag("How do I contact support?")
    intent_based_rag("What models does TechCorp support?")

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  - Metadata filtering happens before vector search — faster + more accurate")
    print("  - Use company/tenant ID for multi-tenant isolation (critical for SaaS RAG)")
    print("  - $and, $or, $in, $ne operators cover most filtering patterns")
    print("  - Intent detection + metadata routing = powerful hybrid approach")
    print("  - Don't over-filter: always have a fallback to broad search")
    print("=" * 60)
