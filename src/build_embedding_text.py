from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "products_ge50.parquet"
)

REVIEW_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "representative_reviews.parquet"
)

INPUT_DOCUMENT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_documents.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_documents_final.parquet"
)

con = duckdb.connect()

print("=" * 70)
print("BUILDING EMBEDDING TEXT")
print("=" * 70)

query = f"""
COPY (
    WITH eligible_reviews AS (
        SELECT
            parent_asin,
            text,
            helpful_vote,
            timestamp,

            ROW_NUMBER() OVER (
                PARTITION BY parent_asin
                ORDER BY
                    helpful_vote DESC,
                    timestamp DESC
            ) AS helpful_rank

        FROM read_parquet('{REVIEW_FILE.as_posix()}')

        WHERE text IS NOT NULL
          AND LENGTH(TRIM(text)) BETWEEN 100 AND 400
    ),

    selected_reviews AS (
        SELECT
            parent_asin,

            STRING_AGG(
                TRIM(text),
                ' '
                ORDER BY helpful_rank
            ) AS review_text

        FROM eligible_reviews

        WHERE helpful_rank <= 2

        GROUP BY parent_asin
    ),

    fallback_reviews AS (
        SELECT
            parent_asin,

            STRING_AGG(
                LEFT(TRIM(text), 350),
                ' '
                ORDER BY
                    helpful_vote DESC,
                    timestamp DESC
            ) AS review_text

        FROM read_parquet('{REVIEW_FILE.as_posix()}')

        WHERE text IS NOT NULL
          AND LENGTH(TRIM(text)) > 0

        GROUP BY parent_asin
    ),

    base AS (
        SELECT
            p.*,

            CASE
                WHEN p.features IS NOT NULL
                THEN LEFT(
                    ARRAY_TO_STRING(p.features, '; '),
                    300
                )
                ELSE ''
            END AS features_short,

            COALESCE(
                sr.review_text,
                fr.review_text,
                ''
            ) AS embedding_reviews

        FROM read_parquet('{PRODUCT_FILE.as_posix()}') p

        LEFT JOIN selected_reviews sr
            ON p.parent_asin = sr.parent_asin

        LEFT JOIN fallback_reviews fr
            ON p.parent_asin = fr.parent_asin
    )

    SELECT
        p.*,

        LEFT(
            CONCAT_WS(
                '\\n',

                CASE
                    WHEN title IS NOT NULL
                         AND LENGTH(TRIM(title)) > 0
                    THEN 'Title: ' || TRIM(title)
                END,

                CASE
                    WHEN categories IS NOT NULL
                         AND LENGTH(
                             ARRAY_TO_STRING(categories, ' ')
                         ) > 0
                    THEN 'Categories: '
                         || ARRAY_TO_STRING(categories, ' > ')
                END,

                CASE
                    WHEN features_short IS NOT NULL
                         AND LENGTH(TRIM(features_short)) > 0
                    THEN 'Features: ' || features_short
                END,

                CASE
                    WHEN embedding_reviews IS NOT NULL
                         AND LENGTH(TRIM(embedding_reviews)) > 0
                    THEN 'Reviews: ' || embedding_reviews
                END
            ),
            900
        ) AS embedding_text

    FROM base p

    ORDER BY parent_asin

) TO '{OUTPUT_FILE.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Creating controlled embedding representation...")
print()
con.execute(query)

# ---------------------------------------------------------------
# Validation
# ---------------------------------------------------------------

stats = con.execute(
    f"""
    SELECT
        COUNT(*) AS products,
        COUNT(DISTINCT parent_asin) AS unique_products,

        COUNT(*) FILTER (
            WHERE embedding_text IS NULL
               OR LENGTH(TRIM(embedding_text)) = 0
        ) AS empty_embedding_text,

        AVG(LENGTH(embedding_text)) AS avg_chars,
        MIN(LENGTH(embedding_text)) AS min_chars,
        quantile_cont(LENGTH(embedding_text), 0.10) AS p10_chars,
        quantile_cont(LENGTH(embedding_text), 0.25) AS p25_chars,
        quantile_cont(LENGTH(embedding_text), 0.50) AS median_chars,
        quantile_cont(LENGTH(embedding_text), 0.75) AS p75_chars,
        quantile_cont(LENGTH(embedding_text), 0.90) AS p90_chars,
        quantile_cont(LENGTH(embedding_text), 0.95) AS p95_chars,
        MAX(LENGTH(embedding_text)) AS max_chars,

        COUNT(*) FILTER (
            WHERE LENGTH(embedding_text) = 900
        ) AS exactly_900

    FROM read_parquet('{OUTPUT_FILE.as_posix()}')
    """
).fetchone()

print("Embedding-text validation")
print("-" * 70)

labels = [
    "Products",
    "Unique products",
    "Empty embedding text",
    "Average characters",
    "Minimum characters",
    "p10 characters",
    "p25 characters",
    "Median characters",
    "p75 characters",
    "p90 characters",
    "p95 characters",
    "Maximum characters",
    "Exactly 900 chars",
]

for label, value in zip(labels, stats):
    if isinstance(value, float):
        print(f"{label:<24}: {value:,.2f}")
    else:
        print(f"{label:<24}: {value:,}")

print()
print(f"Output file: {OUTPUT_FILE}")

print("=" * 70)
print("DONE")
print("=" * 70)

con.close()