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

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "product_documents.parquet"
)

con = duckdb.connect()

print("=" * 70)
print("BUILDING PRODUCT-LEVEL DOCUMENTS")
print("=" * 70)


query = f"""
COPY (
    WITH reviews AS (
        SELECT
            parent_asin,

            STRING_AGG(
                CASE
                    WHEN text IS NOT NULL
                         AND LENGTH(TRIM(text)) > 0
                    THEN
                        'Review '
                        || CAST(review_rank AS VARCHAR)
                        || ' ('
                        || CAST(rating AS VARCHAR)
                        || '/5): '
                        || TRIM(text)
                    ELSE NULL
                END,
                '\\n'
                ORDER BY review_rank
            ) AS review_text

        FROM read_parquet('{REVIEW_FILE.as_posix()}')
        GROUP BY parent_asin
    )

    SELECT
        p.parent_asin,
        p.title,
        p.main_category,
        p.average_rating,
        p.rating_number,
        p.review_count,
        p.price,
        p.store,
        p.categories,
        p.features,
        p.description,
        p.details,
        p.subtitle,

        CONCAT_WS(
            '\\n',

            CASE
                WHEN p.title IS NOT NULL
                     AND LENGTH(TRIM(p.title)) > 0
                THEN 'Product Title: ' || TRIM(p.title)
            END,

            CASE
                WHEN p.main_category IS NOT NULL
                     AND LENGTH(TRIM(p.main_category)) > 0
                THEN 'Category: ' || TRIM(p.main_category)
            END,

            CASE
                WHEN p.subtitle IS NOT NULL
                     AND LENGTH(TRIM(p.subtitle)) > 0
                THEN 'Subtitle: ' || TRIM(p.subtitle)
            END,

            CASE
                WHEN p.features IS NOT NULL
                     AND LENGTH(ARRAY_TO_STRING(p.features, ' ')) > 0
                THEN 'Features: '
                     || ARRAY_TO_STRING(p.features, '; ')
            END,

            CASE
                WHEN p.description IS NOT NULL
                     AND LENGTH(ARRAY_TO_STRING(p.description, ' ')) > 0
                THEN 'Description: '
                     || ARRAY_TO_STRING(p.description, ' ')
            END,

            CASE
                WHEN p.categories IS NOT NULL
                     AND LENGTH(ARRAY_TO_STRING(p.categories, ' ')) > 0
                THEN 'Categories: '
                     || ARRAY_TO_STRING(p.categories, ' > ')
            END,

            CASE
                WHEN p.price IS NOT NULL
                     AND LENGTH(TRIM(p.price)) > 0
                THEN 'Price: ' || TRIM(p.price)
            END,

            CASE
                WHEN p.average_rating IS NOT NULL
                THEN 'Average Rating: '
                     || CAST(ROUND(p.average_rating, 2) AS VARCHAR)
                     || '/5'
            END,

            CASE
                WHEN p.rating_number IS NOT NULL
                THEN 'Rating Count: '
                     || CAST(p.rating_number AS VARCHAR)
            END,

            CASE
                WHEN r.review_text IS NOT NULL
                     AND LENGTH(TRIM(r.review_text)) > 0
                THEN 'Customer Reviews:\\n' || r.review_text
            END

        ) AS document

    FROM read_parquet('{PRODUCT_FILE.as_posix()}') AS p

    LEFT JOIN reviews AS r
        ON p.parent_asin = r.parent_asin

    ORDER BY p.parent_asin

) TO '{OUTPUT_FILE.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Combining metadata with representative reviews...")
con.execute(query)

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

stats = con.execute(
    f"""
    SELECT
        COUNT(*) AS products,
        COUNT(DISTINCT parent_asin) AS unique_products,
        COUNT(*) FILTER (
            WHERE document IS NULL OR LENGTH(TRIM(document)) = 0
        ) AS empty_documents,
        AVG(LENGTH(document)) AS avg_chars,
        MIN(LENGTH(document)) AS min_chars,
        quantile_cont(LENGTH(document), 0.10) AS p10_chars,
        quantile_cont(LENGTH(document), 0.25) AS p25_chars,
        quantile_cont(LENGTH(document), 0.50) AS median_chars,
        quantile_cont(LENGTH(document), 0.75) AS p75_chars,
        quantile_cont(LENGTH(document), 0.90) AS p90_chars,
        quantile_cont(LENGTH(document), 0.95) AS p95_chars,
        MAX(LENGTH(document)) AS max_chars
    FROM read_parquet('{OUTPUT_FILE.as_posix()}')
    """
).fetchone()

print()
print("Document validation")
print("-" * 70)

labels = [
    "Products",
    "Unique products",
    "Empty documents",
    "Average characters",
    "Minimum characters",
    "p10 characters",
    "p25 characters",
    "Median characters",
    "p75 characters",
    "p90 characters",
    "p95 characters",
    "Maximum characters",
]

for label, value in zip(labels, stats):
    if isinstance(value, float):
        print(f"{label:<22}: {value:,.2f}")
    else:
        print(f"{label:<22}: {value:,}")

# ------------------------------------------------------------------
# Review coverage
# ------------------------------------------------------------------

coverage = con.execute(
    f"""
    SELECT
        COUNT(*) AS products,
        COUNT(*) FILTER (
            WHERE r.parent_asin IS NOT NULL
        ) AS products_with_reviews
    FROM read_parquet('{PRODUCT_FILE.as_posix()}') p
    LEFT JOIN (
        SELECT DISTINCT parent_asin
        FROM read_parquet('{REVIEW_FILE.as_posix()}')
    ) r
    ON p.parent_asin = r.parent_asin
    """
).fetchone()

print()
print("Review coverage")
print("-" * 70)
print(f"Products                 : {coverage[0]:,}")
print(f"Products with reviews    : {coverage[1]:,}")

print()
print(f"Output file: {OUTPUT_FILE}")

print("=" * 70)
print("DONE")
print("=" * 70)

con.close()