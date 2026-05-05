import numpy as np
import torch


# Normalization
def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


def l2_normalize_vector(v):
    return v / np.clip(np.linalg.norm(v), 1e-9, None)

# Ranking helper
def top_k(texts, scores, query, k=3):
    pairs = [(t, s) for t, s in zip(texts, scores) if t != query]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]

# GloVe utils
def load_glove(filepath):
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector

    return embeddings


def text_to_glove(text, embeddings, dim=50):
    words = text.lower().split()
    vectors = [embeddings[w] for w in words if w in embeddings]

    if not vectors:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


# BERT utils
def extract_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)

    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return (summed / counts).squeeze().cpu().numpy()