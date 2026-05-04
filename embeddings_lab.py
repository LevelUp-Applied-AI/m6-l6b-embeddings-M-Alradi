"""
Module 6 Week B — Lab: Embeddings Comparison

Compare three text representation methods — TF-IDF, GloVe, and
DistilBERT — on the BBC News corpus (5 categories).
"""

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from transformers import AutoTokenizer, AutoModel

def build_tfidf(texts):
    """Build TF-IDF representations for a list of texts.

    Returns (tfidf_matrix, vectorizer).
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compute_tfidf_similarity(tfidf_matrix):
    """Compute pairwise cosine similarity from a TF-IDF matrix.

    Returns a numpy array of shape (n, n).
    """
    similarity_matrix = sklearn_cosine(tfidf_matrix)
    return similarity_matrix


def load_glove(filepath):
    """Load pre-trained GloVe vectors from a text file.

    Returns a dict mapping each word to a numpy array.
    """
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            embeddings[word] = vector

    return embeddings


def text_to_glove(text, embeddings):
    """Compute the average GloVe embedding for a text.

    Skip out-of-vocabulary words. If every word is OOV, return a zero
    vector of shape (50,).
    """
    words = text.lower().split()
    vectors = []

    for word in words:
        if word in embeddings:
            vectors.append(embeddings[word])

    if len(vectors) == 0:
        return np.zeros(50)

    return np.mean(vectors, axis=0)


def extract_bert_embedding(text, tokenizer, model):
    """Extract a sentence embedding from DistilBERT.

    Returns a numpy array of shape (768,).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Forward pass (no gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Last hidden states: (batch_size=1, seq_len, hidden_size=768)
    last_hidden_state = outputs.last_hidden_state

    # Attention mask: (1, seq_len)
    attention_mask = inputs["attention_mask"]

    # Expand mask to match hidden state shape
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Apply mask and compute mean pooling
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = torch.sum(masked_embeddings, dim=1)
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

    mean_pooled = sum_embeddings / sum_mask

    # Convert to numpy (shape: (768,))
    return mean_pooled.squeeze().cpu().numpy()


def compare_similarities(texts, queries, tfidf_sim, glove_embeddings,
                         bert_model, bert_tokenizer):
    """Compare similarity rankings across TF-IDF, GloVe, and BERT.

    For each query, find the top-3 most similar texts under each method,
    excluding the query itself. Return:

        {query_text: {"tfidf": [(text, score), ...],
                      "glove": [(text, score), ...],
                      "bert":  [(text, score), ...]}}
    """
    results = {}

    # --- Precompute embeddings once ---
    text_glove_vecs = np.array([
        text_to_glove(t, glove_embeddings) for t in texts
    ])

    text_bert_vecs = np.array([
        extract_bert_embedding(t, bert_tokenizer, bert_model) for t in texts
    ])

    for query in queries:
        query_results = {}

        # -------- TF-IDF --------
        try:
            q_idx = texts.index(query)
            tfidf_scores = tfidf_sim[q_idx]
        except ValueError:
            tfidf_scores = np.zeros(len(texts))

        query_results["tfidf"] = _top_k(texts, tfidf_scores, query)

        # -------- GloVe --------
        query_glove = text_to_glove(query, glove_embeddings).reshape(1, -1)
        glove_scores = sklearn_cosine(query_glove, text_glove_vecs)[0]

        query_results["glove"] = _top_k(texts, glove_scores, query)

        # -------- BERT --------
        query_bert = extract_bert_embedding(query, bert_tokenizer, bert_model).reshape(1, -1)
        bert_scores = sklearn_cosine(query_bert, text_bert_vecs)[0]

        query_results["bert"] = _top_k(texts, bert_scores, query)

        results[query] = query_results

    return results


def _top_k(texts, scores, query, k=3):
    """Helper to get top-k (text, score) excluding the query."""
    pairs = [(t, s) for t, s in zip(texts, scores) if t != query]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} texts")

    # Task 1: TF-IDF
    result = build_tfidf(texts)
    if result:
        tfidf_matrix, vectorizer = result
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        tfidf_sim = compute_tfidf_similarity(tfidf_matrix)
        if tfidf_sim is not None:
            print(f"TF-IDF similarity matrix shape: {tfidf_sim.shape}")

    # Task 2: GloVe
    glove = load_glove("data/glove_50k_50d.txt")
    if glove:
        print(f"Loaded {len(glove)} GloVe vectors")
        sample_emb = text_to_glove(texts[0], glove)
        if sample_emb is not None:
            print(f"Sample GloVe text embedding shape: {sample_emb.shape}")

    # Task 3: DistilBERT
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    sample_bert = extract_bert_embedding(texts[0], tokenizer, model)
    if sample_bert is not None:
        print(f"Sample BERT embedding shape: {sample_bert.shape}")

    # Task 4: Compare — pick one query per category so the cross-method
    # ranking comparison is not degenerate (the CSV is sorted by category,
    # so texts[:5] would all be from the same one).
    if result and glove and tfidf_sim is not None:
        queries = [df[df["category"] == cat]["text"].iloc[0]
                   for cat in df["category"].unique()]
        comparison = compare_similarities(
            texts, queries, tfidf_sim, glove, model, tokenizer
        )
        if comparison:
            for q in list(comparison.keys())[:2]:
                print(f"\nQuery: {q[:80]}...")
                for method in ["tfidf", "glove", "bert"]:
                    top = comparison[q].get(method, [])
                    print(f"  {method}: {[t[:40] for t, _ in top[:3]]}")