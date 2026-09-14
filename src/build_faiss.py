from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path("data/processed/product_documents_final.parquet")
OUTPUT_DIR = Path("data/indexes/faiss")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

ID_COLUMN = "parent_asin"
TEXT_COLUMN = "embedding_text"

BATCH_SIZE = 256

# FAISS IVF configuration
NLIST = 256
NPROBE = 32


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

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    if df[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains null values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} contains duplicates.")

    if df[TEXT_COLUMN].isna().any():
        raise ValueError(f"{TEXT_COLUMN} contains null values.")

    if df[TEXT_COLUMN].str.strip().eq("").any():
        raise ValueError(f"{TEXT_COLUMN} contains empty documents.")

    # -------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------

    print()
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    dimension = model.get_embedding_dimension()

    print(f"Embedding dimension: {dimension}")
    print(f"Model max sequence length: {model.max_seq_length}")

    # -------------------------------------------------------------
    # Generate embeddings
    # -------------------------------------------------------------

    texts = df[TEXT_COLUMN].tolist()

    print()
    print("Generating embeddings...")
    print(f"Batch size: {BATCH_SIZE}")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"Embedding shape: {embeddings.shape}")

    # -------------------------------------------------------------
    # Validate embeddings
    # -------------------------------------------------------------

    if embeddings.shape != (len(df), dimension):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    # Since normalize_embeddings=True was used, vectors should
    # have approximately unit L2 norm.
    norms = np.linalg.norm(embeddings, axis=1)

    print(
        "Embedding norm range: "
        f"{norms.min():.6f} - {norms.max():.6f}"
    )

    # -------------------------------------------------------------
    # Build FAISS index
    # -------------------------------------------------------------

    print()
    print("Building FAISS IVF index...")

    # Inner product on normalized vectors is equivalent to cosine
    # similarity.
    quantizer = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        NLIST,
        faiss.METRIC_INNER_PRODUCT
    )

    # IVF requires a training step before adding vectors.
    print("Training FAISS index...")

    index.train(embeddings)

    print("Adding embeddings to FAISS index...")

    index.add(embeddings)

    # Set number of clusters searched at query time.
    index.nprobe = NPROBE

    # -------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_path = OUTPUT_DIR / "faiss_index.bin"
    ids_path = OUTPUT_DIR / "doc_ids.npy"
    config_path = OUTPUT_DIR / "config.json"

    faiss.write_index(index, str(index_path))

    # IMPORTANT:
    # The IDs must remain in exactly the same order as the
    # embeddings added to FAISS.
    doc_ids = df[ID_COLUMN].astype(str).to_numpy()

    np.save(ids_path, doc_ids)

    config = {
        "input_file": str(INPUT_FILE),
        "id_column": ID_COLUMN,
        "text_column": TEXT_COLUMN,
        "model_name": MODEL_NAME,
        "embedding_dimension": dimension,
        "num_documents": len(df),
        "batch_size": BATCH_SIZE,
        "index_type": "IndexIVFFlat",
        "metric": "inner_product",
        "nlist": NLIST,
        "nprobe": NPROBE,
        "normalized_embeddings": True,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print()
    print("FAISS index created successfully.")
    print(f"FAISS index : {index_path}")
    print(f"Document IDs: {ids_path}")
    print(f"Config      : {config_path}")
    print(f"Vectors     : {index.ntotal:,}")


if __name__ == "__main__":
    main()