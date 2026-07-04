# Author: Mohan Raju Amuri
"""
Stop Words
----------
One-liner: Remove high-frequency words that carry no meaning ("the", "is", "at").

Remember:
- Stopwords help reduce noise for BoW / TF-IDF tasks
- NEVER remove stopwords for sentiment analysis
  → "not good" → removes "not" → becomes "good" ← wrong sentiment!
- NLTK has 179 English stopwords; spaCy ~326
- You can add/remove words from the list for your domain

Don't:
- Don't remove stopwords for tasks that rely on sentence structure
  (sentiment, NER, question answering, chatbots)
- Don't apply stopword removal before tokenization
- Don't assume the default list fits your domain
  (e.g., "no" and "not" are stopwords but flip sentiment)
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))

# --- Basic removal ---
text = "This is a simple example showing how to remove stop words from text"
tokens = word_tokenize(text.lower())
filtered = [w for w in tokens if w not in stop_words and w.isalpha()]

print("Original :", tokens)
print("Filtered :", filtered)
# Original : ['this', 'is', 'a', 'simple', 'example', 'showing', 'how', ...]
# Filtered : ['simple', 'example', 'showing', 'remove', 'stop', 'words', 'text']

# --- The sentiment trap ---
print("\n--- Sentiment trap ---")
bad_text = "This movie is not good at all"
tokens_bad = word_tokenize(bad_text.lower())
removed = [w for w in tokens_bad if w not in stop_words]
print("After stopword removal:", removed)
# ['movie', 'good']  ← 'not' was removed → model sees POSITIVE!

# --- Custom domain stopwords ---
domain_stops = stop_words | {"please", "thank", "dear", "regards", "sincerely"}
email = "Dear John, please review the attached report. Thank you, regards"
tokens_email = word_tokenize(email.lower())
filtered_email = [w for w in tokens_email if w not in domain_stops and w.isalpha()]
print("\nEmail filtered:", filtered_email)

print(f"\nDefault stop count : {len(stop_words)}")
print("Sample stops       :", list(stop_words)[:10])
