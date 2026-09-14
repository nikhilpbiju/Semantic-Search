import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# load models
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# load data
df = pd.read_csv("data/products.csv")

# load FAISS index
index = faiss.read_index("data/faiss.index")

# BM25 setup
tokenized_corpus = [doc.split() for doc in df["text"]]
bm25 = BM25Okapi(tokenized_corpus)


def search(query, k=5):
    # ---- Step 1: semantic retrieval ----
    q_emb = bi_encoder.encode([query])
    faiss.normalize_L2(q_emb)

    scores, indices = index.search(q_emb, k * 4)  # get more candidates

    candidates = df.iloc[indices[0]].copy()
    candidates["semantic_score"] = scores[0]

    # ---- Step 2: BM25 scoring ----
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    candidates["bm25_score"] = [bm25_scores[i] for i in indices[0]]

    # ---- Step 3: combine (initial ranking) ----
    candidates["hybrid_score"] = (
        0.6 * candidates["semantic_score"] +
        0.4 * candidates["bm25_score"]
    )

    candidates = candidates.sort_values(
        by="hybrid_score", ascending=False
    ).head(k * 2)

    # ---- Step 4: Re-ranking using cross-encoder ----
    pairs = [
        (query, text) for text in candidates["text"]
    ]

    rerank_scores = cross_encoder.predict(pairs)

    candidates["rerank_score"] = rerank_scores

    # ---- Final ranking ----
    final_results = candidates.sort_values(
        by="rerank_score", ascending=False
    ).head(k)

    return final_results[["title", "description", "rerank_score"]]