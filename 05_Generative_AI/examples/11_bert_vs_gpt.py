# Author: Mohan Raju Amuri
"""
BERT vs GPT — Encoder vs Decoder
----------------------------------
One-liner: BERT reads everything (encoder). GPT writes one token at a time (decoder). Different architectures, different jobs.

                 BERT                    GPT
Architecture:    Encoder-only            Decoder-only
Attention:       Bidirectional           Causal (left-to-right)
Training:        Masked LM + NSP         Next token prediction
Good at:         Understanding           Generating
Tasks:           Classification, NER,    Text generation, chat,
                 Q&A, embeddings         summarization, code
Input:           [CLS] sentence [SEP]    Prompt text...
Output:          Token embeddings         Next token probabilities

Remember:
- BERT sees the full sentence at once → better context understanding
- GPT predicts next token only → better at open-ended generation
- Modern LLMs (Claude, GPT-4, LLaMA) are decoder-only + instruction-tuned
- T5, BART = encoder-decoder → good for translation, summarization
- DistilBERT = BERT distilled to 40% smaller, 97% accuracy (use for production)

Don't:
- Don't use BERT for text generation — it's not designed for it
- Don't use GPT for classification without fine-tuning — use BERT
- Don't confuse GPT-2 (base model) with ChatGPT (instruction-tuned + RLHF)
"""

from transformers import pipeline

# ── BERT: Understanding task (classification) ─────────────────────────────
print("=== BERT: Fill-Mask (understanding) ===")
bert_fill = pipeline("fill-mask", model="distilbert-base-uncased")
# BERT predicts the masked token using BOTH left and right context
masked = "The [MASK] sat on the mat."
results = bert_fill(masked)
print(f"Input: {masked}")
for r in results[:3]:
    print(f"  [{r['score']:.3f}] {r['sequence']}")
# BERT uses "sat on the mat" (right context) + "The" (left) to pick the word

print()
masked2 = "I [MASK] to the bank to deposit my money."
results2 = bert_fill(masked2)
print(f"Input: {masked2}")
for r in results2[:3]:
    print(f"  [{r['score']:.3f}] {r['sequence']}")

# ── GPT: Generation task (causal / decoder) ───────────────────────────────
print("\n=== GPT-2: Text Generation (generating) ===")
gpt_gen = pipeline("text-generation", model="gpt2", pad_token_id=50256)
prompt = "The future of AI in healthcare is"
results = gpt_gen(prompt, max_new_tokens=30, do_sample=True, temperature=0.7)
print(f"Prompt: {prompt}")
print(f"Output: {results[0]['generated_text']}")

# ── Encoder-Decoder: T5 for summarization ────────────────────────────────
print("\n=== T5: Summarization (encoder-decoder) ===")
t5 = pipeline("summarization", model="t5-small")
text = """
Artificial intelligence is rapidly transforming industries worldwide.
From healthcare diagnostics to autonomous vehicles, AI applications are
growing in scope and sophistication. Companies are investing billions
in AI research and development, creating both opportunities and challenges
for workers and society at large.
"""
summary = t5(text, max_length=40, min_length=15, do_sample=False)
print(f"Input ({len(text.split())} words): {text.strip()[:80]}...")
print(f"Summary: {summary[0]['summary_text']}")

# ── Architecture comparison table ─────────────────────────────────────────
print("\n=== Architecture Comparison ===")
print(f"{'Model':<15} {'Arch':<18} {'Best For':<30} {'Example Models'}")
print("-" * 80)
rows = [
    ("BERT",      "Encoder-only",     "Classification, NER, QA",    "BERT, RoBERTa, DistilBERT"),
    ("GPT",       "Decoder-only",     "Text generation, chat",      "GPT-2, GPT-4, LLaMA, Mistral"),
    ("T5/BART",   "Encoder-Decoder",  "Translation, summarization", "T5, BART, mT5"),
    ("ChatGPT",   "Decoder + RLHF",   "Instruction following",      "GPT-3.5, GPT-4, Claude"),
]
for model, arch, use, examples in rows:
    print(f"{model:<15} {arch:<18} {use:<30} {examples}")
