"""
Text Generation with GPT-2
---------------------------
One-liner: GPT = decoder-only transformer that predicts the next token, one at a time.

Architecture: Decoder-only. Left-to-right (autoregressive). No bidirectional attention.

Remember:
- GPT-2 is small (117M–1.5B params) and open — good for learning generation
- model.generate() is the main API for all text generation
- greedy decoding (default) = always pick highest prob token → repetitive
- Use sampling (do_sample=True) for creative/diverse output
- max_new_tokens limits generation length (not input+output total)
- GPT-2 has no instruction following — it just continues your text

Don't:
- Don't use max_length — it counts input + output. Use max_new_tokens.
- Don't use greedy for open-ended generation — outputs get repetitive
- Don't expect GPT-2 to follow instructions (that's an instruction-tuned model like GPT-3.5+)
- Don't confuse GPT-2 with ChatGPT — GPT-2 is a base model, no RLHF

Key differences: BERT vs GPT
  BERT  → encoder, bidirectional, good at understanding (classification, NER)
  GPT   → decoder, left-to-right, good at generating (completion, summarization)
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# ── 1. Simple generation via pipeline ─────────────────────────────────────
print("=== 1. Basic text generation ===")
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
result = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1,
    do_sample=False,  # greedy
    pad_token_id=50256,
)
print(f"Prompt : {prompt}")
print(f"Output : {result[0]['generated_text']}\n")

# ── 2. Greedy vs Sampling ─────────────────────────────────────────────────
print("=== 2. Greedy vs Sampling ===")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

prompt = "Once upon a time, a robot learned to"
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy — deterministic, often repetitive
with torch.no_grad():
    greedy_out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
print("Greedy :", tokenizer.decode(greedy_out[0], skip_special_tokens=True))

# Sampling — diverse, creative
with torch.no_grad():
    sample_out = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
print("Sampled:", tokenizer.decode(sample_out[0], skip_special_tokens=True))

# ── 3. Multiple completions (see diversity) ───────────────────────────────
print("\n=== 3. Multiple completions ===")
gen = pipeline("text-generation", model="gpt2", pad_token_id=50256)
completions = gen(
    "Machine learning helps us",
    max_new_tokens=20,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.9,
)
for i, c in enumerate(completions, 1):
    text = c["generated_text"].replace("\n", " ")
    print(f"  [{i}] {text}")

# ── 4. Perplexity — how "surprised" the model is ─────────────────────────
print("\n=== 4. Perplexity (model confidence) ===")
# Lower perplexity = model finds text more probable / natural
def compute_perplexity(text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

sentences = [
    "The dog sat on the mat",           # natural English
    "Mat the on sat dog the",           # scrambled — unnatural
    "Python is used for machine learning",  # coherent
]
for s in sentences:
    ppl = compute_perplexity(s)
    print(f"  PPL={ppl:6.1f}  {s}")
# Natural sentences → lower perplexity
