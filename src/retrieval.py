from pathlib import Path
import pickle
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PRODUCT_FILE = Path("data/processed/product_documents.parquet")
BM25_DIR = Path("data/indexes/bm25")
FAISS_DIR = Path("data/indexes/faiss")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BM25_TOP_K = 15
FAISS_TOP_K = 15
FINAL_TOP_K = 5

RRF_K = 60


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am",
    "an", "and", "any", "are", "aren", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "couldn", "did", "didn", "do", "does", "doesn",
    "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "has", "hasn", "have", "haven", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn", "it", "its", "itself",
    "just", "me", "more", "most", "mustn", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "shouldn", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn", "we", "were", "weren", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with",
    "won", "would", "wouldn", "you", "your", "yours", "yourself",
    "yourselves"
}


def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS]


# ---------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------

class HybridRetriever:

    def __init__(self):
        print("Loading product data...")

        self.products = pd.read_parquet(
            PRODUCT_FILE
        )

        self.products["parent_asin"] = (
            self.products["parent_asin"].astype(str)
        )

        # -------------------------------------------------------------
        # BM25
        # -------------------------------------------------------------

        print("Loading BM25 index...")

        with open(BM25_DIR / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        self.bm25_ids = np.load(
            BM25_DIR / "doc_ids.npy",
            allow_pickle=True
        ).astype(str)

        # -------------------------------------------------------------
        # FAISS
        # -------------------------------------------------------------

        print("Loading FAISS index...")

        self.faiss_index = faiss.read_index(
            str(FAISS_DIR / "faiss_index.bin")
        )

        self.faiss_ids = np.load(
            FAISS_DIR / "doc_ids.npy",
            allow_pickle=True
        ).astype(str)

        # Use the same search-time setting used during index creation.
        self.faiss_index.nprobe = 32

        # -------------------------------------------------------------
        # Embedding model
        # -------------------------------------------------------------

        print("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_NAME)

        # -------------------------------------------------------------
        # Product lookup
        # -------------------------------------------------------------

        self.product_lookup = (
            self.products
            .set_index("parent_asin")
            .to_dict(orient="index")
        )

        print("Hybrid retriever ready.")

    # -----------------------------------------------------------------
    # BM25 retrieval
    # -----------------------------------------------------------------

    def bm25_search(self, query, top_k=BM25_TOP_K):

        tokens = simple_tokenize(query)

        scores = self.bm25.get_scores(tokens)

        top_positions = np.argsort(scores)[::-1][:top_k]

        results = []

        for position in top_positions:
            parent_asin = self.bm25_ids[position]

            results.append({
                "parent_asin": parent_asin,
                "score": float(scores[position])
            })

        return results

    # -----------------------------------------------------------------
    # FAISS semantic retrieval
    # -----------------------------------------------------------------

    def semantic_search(self, query, top_k=FAISS_TOP_K):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        scores, positions = self.faiss_index.search(
            query_embedding,
            top_k
        )

        results = []

        for position, score in zip(
            positions[0],
            scores[0]
        ):

            if position < 0:
                continue

            parent_asin = self.faiss_ids[position]

            results.append({
                "parent_asin": parent_asin,
                "score": float(score)
            })

        return results

    # -----------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -----------------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        bm25_results,
        semantic_results,
        top_k=FINAL_TOP_K
    ):

        rrf_scores = {}

        # BM25 ranks
        for rank, result in enumerate(bm25_results, start=1):

            product_id = result["parent_asin"]

            rrf_scores.setdefault(product_id, 0.0)

            rrf_scores[product_id] += (
                1.0 / (RRF_K + rank)
            )

        # Semantic ranks
        for rank, result in enumerate(
            semantic_results,
            start=1
        ):

            product_id = result["parent_asin"]

            rrf_scores.setdefault(product_id, 0.0)

            rrf_scores[product_id] += (
                1.0 / (RRF_K + rank)
            )

        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for product_id, rrf_score in ranked[:top_k]:

            product = self.product_lookup.get(product_id)

            if product is None:
                continue

            results.append({
                "parent_asin": product_id,
                "rrf_score": rrf_score,
                "title": product["title"],
                "average_rating": product["average_rating"],
                "rating_number": product["rating_number"],
                "review_count": product["review_count"],
                "price": product["price"],
                "store": product["store"],
                "features": product["features"],
                "categories": product["categories"],
                "description": product["description"],
                "subtitle": product["subtitle"],
                "document": product["document"]
            })

        return results

    # -----------------------------------------------------------------
    # Complete hybrid retrieval
    # -----------------------------------------------------------------

    def search(self, query, top_k=FINAL_TOP_K):

        if not query or not query.strip():
            return []

        bm25_results = self.bm25_search(
            query,
            top_k=BM25_TOP_K
        )

        semantic_results = self.semantic_search(
            query,
            top_k=FAISS_TOP_K
        )

        final_results = self.reciprocal_rank_fusion(
            bm25_results,
            semantic_results,
            top_k=top_k
        )

        return final_results