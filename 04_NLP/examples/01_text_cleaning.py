# Author: Mohan Raju Amuri
"""
Text Cleaning
-------------
One-liner: Raw text is dirty. Clean it before ANY NLP task.

Remember:
- Always lowercase first
- Remove HTML tags if scraping web data
- Keep apostrophes sometimes (don't → do n't is intentional in tokenization)
- Order matters: lowercase → remove noise → tokenize

Don't:
- Don't skip cleaning and go straight to tokenization
- Don't remove ALL punctuation blindly (periods matter for sentence splitting)
- Don't clean after tokenizing
"""

import re
import string

# --- Sample dirty text ---
raw = "  Hello, World!! Visit https://example.com 😊 <b>Today</b> is the 1st DAY...  "


def clean_text(text: str) -> str:
    text = text.lower()                          # lowercase
    text = re.sub(r"<[^>]+>", "", text)          # remove HTML tags
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^\x00-\x7F]+", "", text)    # remove non-ASCII (emojis etc.)
    text = re.sub(r"\d+", "", text)              # remove numbers
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )                                            # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()     # collapse whitespace
    return text


print("Raw :", raw)
print("Clean:", clean_text(raw))

# Expected output:
# Raw :   Hello, World!! Visit https://example.com 😊 <b>Today</b> is the 1st DAY...
# Clean: hello world  today is the  st day

# --- Minimal pipeline you'll reuse everywhere ---
def basic_clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters + spaces
    return text

samples = [
    "Machine Learning is AMAZING!!!",
    "Contact us @ support@email.com",
    "Price: $299.99 (limited time)",
]
for s in samples:
    print(f"  {s!r:45} → {basic_clean(s)!r}")
