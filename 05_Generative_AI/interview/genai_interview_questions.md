# Generative AI — Interview Questions

## Transformers & Architecture

**Q: What is the transformer architecture in one sentence?**
Multi-head self-attention + feed-forward layers, stacked N times, with positional encoding added to embeddings.

**Q: What does attention do?**
Each token computes a weighted sum of all other tokens' values — the weights (from Q×K^T) determine how much to "attend to" each token. Result: context-aware representations.

**Q: Why is positional encoding needed?**
Attention has no notion of order — "dog bites man" = "man bites dog" without it. Positional encoding injects position information into the embeddings.

**Q: BERT vs GPT — key differences?**
BERT: encoder-only, bidirectional, trained with masked LM → good at understanding.
GPT: decoder-only, left-to-right, trained with next-token prediction → good at generating.

**Q: What is attention complexity? Why does it matter?**
O(n²) in sequence length. Doubling sequence length = 4× compute. This is why long context is expensive and slow.

---

## Tokenization

**Q: What is a token?**
A subword unit — not always a whole word. "unbelievable" → ["un", "##bel", "##iev", "##able"]. Roughly 1 token ≈ 4 characters ≈ 0.75 words.

**Q: Why does non-English text use more tokens?**
Tokenizer vocabulary is trained mostly on English. Non-Latin scripts (Chinese, Arabic, Korean) often have 1 character = 1–3 tokens.

**Q: What is BPE?**
Byte Pair Encoding — merges most frequent character pairs iteratively to build a vocabulary of subword units. Used by GPT-2, GPT-4, LLaMA.

---

## Generation & Sampling

**Q: What is temperature in LLM generation?**
Scales the logits before softmax. Low temp → confident, focused. High temp → diverse, random. 0 = deterministic greedy.

**Q: Top-P vs Top-K?**
Top-K: always keep exactly K tokens. Top-P: keep smallest set covering P probability mass — adapts to context. Top-P is preferred.

**Q: What is greedy decoding? Problem?**
Always pick the highest probability token. Deterministic but leads to repetitive, low-diversity output.

**Q: What is beam search?**
Keep top-B candidate sequences at each step. Better than greedy, but still can be repetitive. Good for translation/summarization.

---

## Practical / Production

**Q: What is a hallucination?**
LLM generates confident but factually incorrect output. Cause: next-token prediction doesn't have a "truth" signal.

**Q: How do you reduce hallucinations?**
RAG (ground with retrieved context), temperature=0 for factual tasks, tell model to say "I don't know", validate output.

**Q: What is a context window?**
Max tokens the model can process at once (input + output combined). Beyond this, content is truncated or errors.

**Q: "Lost in the middle" effect?**
LLMs perform worse on information buried in the middle of a long context. Put key info at the start or end.

**Q: Why use structured output / JSON mode?**
Free text requires fragile parsing. JSON mode guarantees parseable structure. Use Pydantic for schema validation.

**Q: What is the difference between base model and instruction-tuned model?**
Base: trained on raw text, continues prompts. Instruction-tuned (e.g. ChatGPT): fine-tuned with RLHF to follow instructions, refuse harmful requests, be helpful.

---

## Embeddings

**Q: What are embeddings?**
Dense vector representation of text where semantic similarity = geometric proximity (cosine distance).

**Q: When to use embeddings vs TF-IDF?**
TF-IDF: keyword matching, fast, no model. Embeddings: semantic matching, understands synonyms, slower, needs model.

**Q: Can you mix embeddings from different models?**
No — embedding spaces are model-specific. Never compare vectors from different models.

**Q: What is the best embedding model for most tasks?**
`all-MiniLM-L6-v2` (sentence-transformers) for local/fast. `text-embedding-3-small` (OpenAI) for production quality.
