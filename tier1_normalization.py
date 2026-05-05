import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from transformers import AutoTokenizer, AutoModel
from utils import (
    l2_normalize,
    l2_normalize_vector,
    top_k,
    load_glove,
    text_to_glove,
    extract_bert_embedding
)

# TF-IDF
def build_tfidf(texts):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(texts).toarray()


# CORE COMPARISON FUNCTION
def compare_method(texts, query, embeddings, method_name):
    query_idx = texts.index(query)
    X = np.array(embeddings)

    # ---------------- RAW SPACE ----------------
    q_raw = X[query_idx].reshape(1, -1)

    cos_raw = cosine_similarity(q_raw, X)[0]
    euc_raw = -euclidean_distances(q_raw, X)[0]

    cos_raw_top = top_k(texts, cos_raw, query)
    euc_raw_top = top_k(texts, euc_raw, query)

    # ---------------- NORMALIZED SPACE ----------------
    X_norm = l2_normalize(X)
    q_norm = l2_normalize_vector(X[query_idx]).reshape(1, -1)

    cos_norm = cosine_similarity(q_norm, X_norm)[0]
    euc_norm = -euclidean_distances(q_norm, X_norm)[0]

    cos_norm_top = top_k(texts, cos_norm, query)
    euc_norm_top = top_k(texts, euc_norm, query)

    # ---------------- PRINT RESULTS ----------------
    print(f"\n\n================ {method_name} ================")
    print("Query:", query[:80], "...")

    print("\n--- RAW SPACE ---")
    print("\nCosine Top-3:")
    for t, _ in cos_raw_top:
        print("-", t[:60])

    print("\nEuclidean Top-3:")
    for t, _ in euc_raw_top:
        print("-", t[:60])

    print("\n--- NORMALIZED SPACE ---")
    print("\nCosine Top-3:")
    for t, _ in cos_norm_top:
        print("-", t[:60])

    print("\nEuclidean Top-3:")
    for t, _ in euc_norm_top:
        print("-", t[:60])

    # ---------------- DIFFERENCE DETECTION ----------------
    def same(a, b):
        return [x for x, _ in a] == [x for x, _ in b]

    if not same(cos_raw_top, euc_raw_top):
        print("\n RAW: cosine vs euclidean ranking differs")

    if not same(cos_norm_top, euc_norm_top):
        print("\n NORMALIZED: cosine vs euclidean ranking differs")

    if not same(cos_raw_top, cos_norm_top):
        print("\n NORMALIZATION CHANGED COSINE RANKING")

    if not same(euc_raw_top, euc_norm_top):
        print("\n NORMALIZATION CHANGED EUCLIDEAN RANKING")


if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("data/bbc_news.csv")
    texts = df["text"].tolist()

    print(f"Loaded {len(texts)} documents")

    # Pick 5 diverse queries (one per category)
    queries = [
        df[df["category"] == cat]["text"].iloc[0]
        for cat in df["category"].unique()
    ][:5]


    # TF-IDF EXPERIMENT
    print("\n\n################ TF-IDF ################")
    tfidf = build_tfidf(texts)

    for q in queries:
        compare_method(texts, q, tfidf, "TF-IDF")


    # GloVe EXPERIMENT
    print("\n\n################ GloVe ################")
    glove = load_glove("data/glove_50k_50d.txt")
    glove_vecs = np.array([text_to_glove(t, glove) for t in texts])

    for q in queries:
        compare_method(texts, q, glove_vecs, "GloVe")


    # BERT EXPERIMENT
    print("\n\n################ BERT ################")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    bert_vecs = np.array([
        extract_bert_embedding(t, tokenizer, model)
        for t in texts
    ])

    for q in queries:
        compare_method(texts, q, bert_vecs, "BERT")