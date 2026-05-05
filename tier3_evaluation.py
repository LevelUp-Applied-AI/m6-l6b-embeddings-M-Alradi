"""
Tier 3 — Retrieval Evaluation Harness

Evaluate TF-IDF, GloVe, and BERT using:
- Mean Reciprocal Rank (MRR)
- Precision@K (P@3, P@5)
"""

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from embeddings_lab import (
    build_tfidf,
    compute_tfidf_similarity,
    text_to_glove,
    load_glove,
    extract_bert_embedding
)


# Retrieval helpers
def rank_documents(query_vec, doc_vecs):
    """Return ranked indices based on cosine similarity."""
    scores = cosine_similarity(query_vec, doc_vecs)[0]
    ranked_indices = np.argsort(scores)[::-1]
    return ranked_indices, scores


# Metrics
def precision_at_k(ranked_indices, relevant_set, k):
    """Compute Precision@K."""
    top_k = ranked_indices[:k]
    hits = sum(1 for idx in top_k if idx in relevant_set)
    return hits / k


def reciprocal_rank(ranked_indices, relevant_set):
    """Compute Reciprocal Rank."""
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx in relevant_set:
            return 1 / rank
    return 0.0


def evaluate_query(ranked_indices, relevant_set):
    """Compute all metrics for a single query."""
    return {
        "MRR": reciprocal_rank(ranked_indices, relevant_set),
        "P@3": precision_at_k(ranked_indices, relevant_set, 3),
        "P@5": precision_at_k(ranked_indices, relevant_set, 5),
    }


# Evaluation pipeline
def evaluate_all(texts, queries, relevance, glove, tokenizer, model):
    """
    Evaluate all methods.

    queries: list of query strings
    relevance: dict {query: set(doc_indices)}
    """

    results = []

    # --- TF-IDF ---
    tfidf_matrix, _ = build_tfidf(texts)

    # --- Precompute embeddings ---
    glove_vecs = np.array([text_to_glove(t, glove) for t in texts])
    bert_vecs = np.array([
        extract_bert_embedding(t, tokenizer, model) for t in texts
    ])

    for query in queries:
        rel_set = relevance[query]

        # ---------- TF-IDF ----------
        q_vec = tfidf_matrix[texts.index(query)]
        ranked, _ = rank_documents(q_vec, tfidf_matrix)
        tfidf_metrics = evaluate_query(ranked, rel_set)

        # ---------- GloVe ----------
        q_glove = text_to_glove(query, glove).reshape(1, -1)
        ranked, _ = rank_documents(q_glove, glove_vecs)
        glove_metrics = evaluate_query(ranked, rel_set)

        # ---------- BERT ----------
        q_bert = extract_bert_embedding(query, tokenizer, model).reshape(1, -1)
        ranked, _ = rank_documents(q_bert, bert_vecs)
        bert_metrics = evaluate_query(ranked, rel_set)

        results.append({
            "query": query[:50],
            "tfidf_MRR": tfidf_metrics["MRR"],
            "tfidf_P@3": tfidf_metrics["P@3"],
            "tfidf_P@5": tfidf_metrics["P@5"],
            "glove_MRR": glove_metrics["MRR"],
            "glove_P@3": glove_metrics["P@3"],
            "glove_P@5": glove_metrics["P@5"],
            "bert_MRR": bert_metrics["MRR"],
            "bert_P@3": bert_metrics["P@3"],
            "bert_P@5": bert_metrics["P@5"],
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} texts")

    # Load models
    glove = load_glove("data/glove_50k_50d.txt")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    # Define queries + relevance
    queries = [
        texts[0],
        texts[100],
        texts[200],
        texts[300],
        texts[400],
        texts[500],
        texts[600],
        texts[700],
        texts[800],
        texts[900],
    ]

    # Example relevance (same category = relevant)
    relevance = {}
    for q in queries:
        q_cat = df[df["text"] == q]["category"].values[0]
        relevant_docs = set(df[df["category"] == q_cat].index.tolist())
        relevance[q] = relevant_docs

    # Run evaluation
    results_df = evaluate_all(texts, queries, relevance, glove, tokenizer, model)

    print("\nEvaluation Results:")
    print(results_df.mean(numeric_only=True))