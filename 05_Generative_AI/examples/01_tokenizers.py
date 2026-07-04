# Author: Mohan Raju Amuri
"""
Tokenizers — How Text Becomes Numbers
---------------------------------------
One-liner: LLMs don't see words — they see token IDs. Tokenization is the first and last step.

Remember:
- Tokens are NOT words. "unbelievable" → ["un", "believ", "able"] (3 tokens)
- 1 token ≈ 4 characters ≈ 0.75 words (rough rule of thumb)
- Different models use DIFFERENT tokenizers — token counts vary
- BPE (Byte Pair Encoding) is the most common: GPT-2, GPT-4, LLaMA all use it
- tiktoken is OpenAI's tokenizer (fastest, most accurate for GPT models)
- Tokenizer is trained alongside the model — never swap them

Don't:
- Don't estimate cost/context using word count — use token count
- Don't assume tokenization is language-agnostic: Chinese/Arabic use more tokens per word
- Don't split text by words before feeding to a transformer — it does its own tokenization
- Don't ignore the special tokens: [CLS], [SEP], <|endoftext|>, <s>, </s>

Setup:
  pip install transformers tiktoken
"""

# ── Part 1: tiktoken (OpenAI's tokenizer) ──────────────────────────────────
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 / GPT-3.5 tokenizer

texts = [
    "Hello world",
    "Machine learning is fascinating",
    "Tokenization is not trivial",
    "ChatGPT uses Byte Pair Encoding",
    "Привет мир",          # Russian — more tokens per word
    "こんにちは世界",          # Japanese — even more tokens
]

print("=== tiktoken (GPT-4 tokenizer) ===")
print(f"{'Text':<35} {'Tokens':>6}  {'IDs'}")
print("-" * 70)
for text in texts:
    ids = enc.encode(text)
    print(f"{text:<35} {len(ids):>6}  {ids[:8]}")

# Decode back
encoded = enc.encode("Hello, I'm a language model")
print(f"\nEncoded : {encoded}")
print(f"Decoded : {enc.decode(encoded)}")
print(f"By token: {[enc.decode([t]) for t in encoded]}")

# ── Part 2: Hugging Face tokenizer (BERT) ─────────────────────────────────
from transformers import AutoTokenizer

print("\n=== BERT Tokenizer (WordPiece) ===")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

sentence = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer.tokenize(sentence)
ids = tokenizer.encode(sentence)

print(f"Text    : {sentence}")
print(f"Tokens  : {tokens}")
print(f"IDs     : {ids}")
print(f"Special : [CLS]={tokenizer.cls_token_id}, [SEP]={tokenizer.sep_token_id}")

# See how subwords work
words = ["unbelievable", "tokenization", "GPT", "Anthropic", "supercalifragilistic"]
print("\n--- Subword tokenization ---")
for word in words:
    toks = tokenizer.tokenize(word)
    print(f"  {word:<25} → {toks}")
# "unbelievable" → ["un", "##bel", "##iev", "##able"]
# ## prefix means "continuation of previous token"

# ── Part 3: Token budget awareness ────────────────────────────────────────
print("\n=== Token budget (practical) ===")
context_limits = {
    "GPT-3.5-turbo": 16_385,
    "GPT-4o":        128_000,
    "Claude 3.5":    200_000,
    "LLaMA 3 8B":    8_192,
}
my_text = "This is a sample document. " * 500  # ~3500 words
my_tokens = len(enc.encode(my_text))

print(f"Document: ~{len(my_text.split())} words → {my_tokens} tokens")
for model, limit in context_limits.items():
    fits = "✓ fits" if my_tokens < limit else "✗ too long"
    print(f"  {model:<20} limit={limit:>7,}  {fits}")
