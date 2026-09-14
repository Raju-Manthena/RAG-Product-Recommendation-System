# RAG-Based Product Recommendation System

A product recommendation system that combines lexical and semantic retrieval with Retrieval-Augmented Generation (RAG) to recommend products from Amazon Cell Phones & Accessories data.

## Overview

This project implements a RAG-based product recommendation system using product metadata and customer reviews from the Amazon Reviews 2023 dataset.

The system uses a hybrid retrieval pipeline that combines BM25 keyword retrieval and FAISS-based semantic retrieval. The retrieved results are combined using Reciprocal Rank Fusion (RRF), after which the top products and their supporting information are provided to Gemini to generate recommendations.

The application is implemented using Streamlit.

## Live Demo

[Open the deployed application](https://rag-appuct-recommendation-system-lmtsahyyd9yyvnwgthfbzd.streamlit.app/)

## Features

- Product retrieval using BM25 keyword matching
- Semantic retrieval using SentenceTransformers and FAISS
- Hybrid retrieval using Reciprocal Rank Fusion
- Product-level documents containing metadata and representative customer reviews
- RAG-based recommendation generation using Gemini
- Constraint-aware recommendation prompting
- Streamlit user interface
- Manual evaluation of retrieval results
- Manual evaluation of generated recommendations

## System Architecture

```text
                         User Query
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
            BM25 Retrieval       FAISS Retrieval
              Top 15                 Top 15
                 |                       |
                 +-----------+-----------+
                             |
                             v
                    Reciprocal Rank
                       Fusion (RRF)
                             |
                         Top 5 Products
                             |
                             v
                    Context Construction
                             |
                             v
                       Gemini LLM
                             |
                             v
                    Product Recommendations
```

## Dataset

The project uses the Amazon Reviews 2023 — Cell Phones & Accessories dataset.

Dataset source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The dataset contains product metadata and customer reviews.

### Product Filtering

Products were retained if they had at least 50 reviews.

This resulted in:

- Products with reviews: 1,288,441
- Products retained: 60,889
- Minimum reviews per retained product: 50

The filtering criterion was used to provide sufficient review evidence for product-level retrieval and recommendation.

### Representative Review Selection

Five representative reviews were selected for each retained product.

Reviews were selected using:

1. Helpful vote count in descending order
2. Review timestamp in descending order

This resulted in:

- Products: 60,889
- Representative reviews: 304,445
- Reviews per product: 5

## Data Processing

A product-level document was constructed for each retained product using:

- Product title
- Main category
- Subtitle
- Features
- Description
- Categories
- Price
- Average rating
- Rating count
- Five representative customer reviews

The resulting product documents are used for lexical retrieval and for constructing the context supplied to the language model.

### Embedding Text

A shorter embedding-specific representation was created for semantic retrieval.

It contains:

- Product title
- Categories
- Product features
- Selected customer reviews

The embedding text is capped at 900 characters before being encoded.

The retrieval representations are therefore separated:

```text
Product Document
       |
       +----> BM25
       |
       +----> RAG Context

Embedding Text
       |
       +----> SentenceTransformer
                    |
                    v
                  FAISS
```

## Retrieval Pipeline

### BM25 Retrieval

BM25 is used for lexical retrieval and is useful for queries containing specific product terms and keywords.

The system retrieves the top 15 products using BM25.

### Semantic Retrieval

Semantic retrieval uses:

- Model: `sentence-transformers/all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings.

FAISS is used to perform approximate nearest-neighbor search using an `IndexIVFFlat` index.

Configuration:

- Embedding dimension: 384
- Index: IndexIVFFlat
- Number of clusters (nlist): 256
- Search probes (nprobe): 32
- Similarity metric: Inner Product
- Normalized embeddings: Yes
- Semantic retrieval: Top 15

### Reciprocal Rank Fusion

The BM25 and FAISS results are combined using Reciprocal Rank Fusion.

The pipeline combines:

- BM25 top 15 results
- FAISS top 15 results

and produces a final ranked list of 5 products.

The RRF constant is: `k = 60`

## RAG Generation

The retrieved products are converted into a context containing product metadata and customer reviews.

The context is provided to Gemini together with the user's query.

The generation prompt instructs the model to:

- Use only information present in the retrieved context
- Identify requirements from the user query
- Treat explicit requirements as constraints
- Avoid assuming unsupported product characteristics
- Distinguish product specifications from customer opinions
- Consider negative customer feedback
- Identify conflicting information
- Avoid recommending products that strongly conflict with the user's requirements
- State when the available information is insufficient to make a reliable recommendation

Customer reviews are treated as evidence of customer experiences rather than guaranteed product specifications.

## Evaluation

### Retrieval Evaluation

A manual evaluation was performed using six representative queries.

The following retrieval methods were evaluated:

- BM25
- FAISS
- Hybrid RRF

The top 5 results from each method were manually classified as:

- Relevant
- Partially relevant
- Not relevant

Results:

| Method | Relevant | Partially Relevant | Not Relevant |
|---|---|---|---|
| BM25 | 33.3% | 40.0% | 26.7% |
| FAISS | 56.7% | 36.7% | 6.7% |
| Hybrid RRF | 53.3% | 30.0% | 16.7% |

The small manual assessment showed that BM25 was useful for keyword-specific queries, while FAISS produced stronger semantic matches. Hybrid RRF combined lexical and semantic retrieval and provided a balanced retrieval strategy.

### RAG Evaluation

Generated recommendations were evaluated using four representative product queries covering requirements such as:

- Battery life
- Comfort and running
- Sports and waterproofing
- Phone-case grip

The evaluation focused on:

- Whether recommendations were grounded in retrieved information
- Whether user requirements were considered
- Whether unsupported requirements were identified
- Whether negative or conflicting evidence was considered

The evaluation showed that generation quality depends on the quality of the retrieved products and the information available in the retrieved context.

## Project Structure

```text
RAG-Product-Recommendation-System/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── data/
│   ├── indexes/
│   │   ├── bm25/
│   │   │   ├── bm25.pkl
│   │   │   ├── config.json
│   │   │   └── doc_ids.npy
│   │   │
│   │   └── faiss/
│   │       ├── faiss_index.bin
│   │       ├── config.json
│   │       └── doc_ids.npy
│   │
│   └── processed/
│       └── product_documents.parquet
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_retrieval_evaluation.ipynb
│   └── 03_rag_evaluation.ipynb
│
└── src/
    ├── build_bm25.py
    ├── build_embedding_text.py
    ├── build_faiss.py
    ├── build_product_documents.py
    ├── context_builder.py
    ├── create_filtered_dataset.py
    ├── gemini_generator.py
    ├── retrieval.py
    ├── retrieval_backup.py
    └── select_representative_reviews.py
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Raju-Manthena/RAG-Product-Recommendation-System.git
cd RAG-Product-Recommendation-System
```

Create a Python environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The application uses the Gemini API for recommendation generation.

Set the Gemini API key as an environment variable:

```bash
GEMINI_API_KEY=your_api_key
```

For Streamlit Community Cloud, add the API key through the application's Secrets settings instead of committing it to the repository.

## Running the Application

Run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The application provides a text input where users can describe the type of product they are looking for.

Example query:

```text
wireless earbuds under $50 with good battery life
```

## Deployment

The application can be deployed using Streamlit Community Cloud.

The repository contains:

- `app.py` as the Streamlit entry point
- `requirements.txt` for Python dependencies
- Git LFS for large retrieval index and dataset files

The Gemini API key should be configured using Streamlit's secrets management and should not be committed to the repository.

## Technologies Used

- Python
- Pandas
- PyArrow
- SentenceTransformers
- FAISS
- BM25
- Google Gemini
- Streamlit
- Jupyter Notebook
- Git
- Git LFS

## Limitations

- Retrieval and RAG evaluation are based on a small manually evaluated query set.
- Recommendations are limited to products present in the processed Amazon dataset.
- Product information and customer reviews may contain incomplete or conflicting information.
- Complex constraints may be difficult to verify when the retrieved information does not contain sufficient evidence.
- Recommendation quality depends on retrieval quality.
- The system does not perform personalized recommendations based on an individual user's purchase history or preferences.
- The dataset represents a fixed snapshot rather than real-time Amazon product information.
- The manual evaluation does not provide a statistically rigorous estimate of system-wide recommendation quality.

## Future Improvements

- Larger-scale automated retrieval evaluation
- Automated relevance and grounding metrics
- Improved handling of product categories and constraints
- Query-aware reranking
- More efficient retrieval for larger product collections
- Evaluation using a larger and more diverse query set