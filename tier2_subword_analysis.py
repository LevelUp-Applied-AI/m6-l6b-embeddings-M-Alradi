"""
Tier 2 — Subword Tokenization Analysis

Compare whitespace tokenization (GloVe-style) vs BERT WordPiece.
"""

import re
import pandas as pd
from collections import Counter, defaultdict

from transformers import AutoTokenizer

# Tokenization helpers
def whitespace_tokenize(text):
    """Simple whitespace tokenization (GloVe-style)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)  # keep words + hyphens
    return text.split()


def bert_tokenize_words(text, tokenizer):
    """
    Tokenize each whitespace word using BERT tokenizer.

    Returns:
        dict: {word: [subword_tokens]}
    """
    words = whitespace_tokenize(text)
    token_map = {}

    for word in words:
        sub_tokens = tokenizer.tokenize(word)
        token_map[word] = sub_tokens

    return token_map


# Analysis functions
def compute_subword_ratio(texts, tokenizer):
    """
    Compute avg subwords per word for each text.

    Returns:
        list of floats
    """
    ratios = []

    for text in texts:
        token_map = bert_tokenize_words(text, tokenizer)

        total_words = len(token_map)
        total_subwords = sum(len(v) for v in token_map.values())

        ratio = total_subwords / total_words if total_words > 0 else 0
        ratios.append(ratio)

    return ratios


def most_split_words(texts, tokenizer, top_k=10):
    """
    Find most frequently split words.

    Returns:
        list of (word, count)
    """
    split_counter = Counter()

    for text in texts:
        token_map = bert_tokenize_words(text, tokenizer)

        for word, sub_tokens in token_map.items():
            if is_real_subword_split(sub_tokens):
                split_counter[word] += 1

    return split_counter.most_common(top_k)


def is_real_subword_split(sub_tokens):
    """Check if BERT actually split into subword pieces."""
    return any(token.startswith("##") for token in sub_tokens)


def collect_split_examples(texts, tokenizer):
    """
    Collect examples of how words are split.

    Returns:
        dict: {word: [example_subtokens]}
    """
    examples = {}

    for text in texts:
        token_map = bert_tokenize_words(text, tokenizer)

        for word, sub_tokens in token_map.items():
            if len(sub_tokens) > 1 and word not in examples:
                examples[word] = sub_tokens

    return examples


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts")

    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # --- 1. Subword ratio ---
    ratios = compute_subword_ratio(texts, tokenizer)
    print(f"\nAverage subwords per word (dataset): {sum(ratios)/len(ratios):.3f}")

    # --- 2. Most split words ---
    top_split = most_split_words(texts, tokenizer)

    print("\nTop 10 most frequently split words:")
    for word, count in top_split:
        print(f"- {word}: {count} times")

    # --- 3. Examples ---
    examples = collect_split_examples(texts, tokenizer)

    print("\nSample split examples:")
    for word, sub in list(examples.items())[:10]:
        print(f"- {word} -> {sub}")