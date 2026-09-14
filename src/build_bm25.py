from pathlib import Path
import json
import pickle
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("data/processed/product_documents.parquet")
OUTPUT_DIR = Path("data/indexes/bm25")

ID_COLUMN = "parent_asin"
TEXT_COLUMN = "document"


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

# Same basic tokenization strategy used by the reference project.
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
    """
    Lowercase, keep alphanumeric characters/spaces/hyphens,
    split into tokens, and remove English stopwords.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"Loading: {INPUT_FILE}")

    df = pd.read_parquet(
        INPUT_FILE,
        columns=[ID_COLUMN, TEXT_COLUMN]
    )

    print(f"Products loaded: {len(df):,}")

    # Basic validation
    if df[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains null values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} contains duplicates.")

    if df[TEXT_COLUMN].isna().any():
        raise ValueError(f"{TEXT_COLUMN} contains null values.")

    # Tokenize full product documents
    print("Tokenizing documents...")

    tokenized_documents = [
        simple_tokenize(text)
        for text in df[TEXT_COLUMN]
    ]

    # Check for empty tokenized documents
    empty_count = sum(len(tokens) == 0 for tokens in tokenized_documents)

    if empty_count > 0:
        raise ValueError(
            f"{empty_count} documents became empty after tokenization."
        )

    print(f"Empty tokenized documents: {empty_count}")

    # Build BM25
    print("Building BM25 index...")

    bm25 = BM25Okapi(tokenized_documents)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save BM25 object
    bm25_path = OUTPUT_DIR / "bm25.pkl"

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    # Save document IDs in exactly the same order as BM25 documents
    doc_ids = df[ID_COLUMN].astype(str).to_numpy()

    np.save(
        OUTPUT_DIR / "doc_ids.npy",
        doc_ids
    )

    # Save configuration
    config = {
        "input_file": str(INPUT_FILE),
        "id_column": ID_COLUMN,
        "text_column": TEXT_COLUMN,
        "num_documents": len(df),
        "tokenizer": "simple_tokenize",
        "stopwords": "english",
        "bm25": "BM25Okapi"
    }

    with open(OUTPUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("BM25 index created successfully.")
    print(f"BM25 index : {bm25_path}")
    print(f"Document IDs: {OUTPUT_DIR / 'doc_ids.npy'}")
    print(f"Config      : {OUTPUT_DIR / 'config.json'}")


if __name__ == "__main__":
    main()