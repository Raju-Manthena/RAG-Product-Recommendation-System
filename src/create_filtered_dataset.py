from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent

META_GLOB = (
    BASE_DIR
    / "data"
    / "raw"
    / "raw_meta_Cell_Phones_and_Accessories"
    / "*.parquet"
)

REVIEW_GLOB = (
    BASE_DIR
    / "data"
    / "raw"
    / "raw_review_Cell_Phones_and_Accessories"
    / "*.parquet"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "products_ge50.parquet"

con = duckdb.connect()

print("=" * 70)
print("CREATING FILTERED PRODUCT DATASET")
print("=" * 70)

query = f"""
COPY (
    WITH review_counts AS (
        SELECT
            parent_asin,
            COUNT(*) AS review_count
        FROM read_parquet('{REVIEW_GLOB.as_posix()}')
        GROUP BY parent_asin
    )

    SELECT
        m.parent_asin,
        m.main_category,
        m.title,
        m.average_rating,
        m.rating_number,
        m.features,
        m.description,
        m.price,
        m.store,
        m.categories,
        m.details,
        m.subtitle,
        r.review_count

    FROM read_parquet('{META_GLOB.as_posix()}') AS m

    INNER JOIN review_counts AS r
        ON m.parent_asin = r.parent_asin

    WHERE r.review_count >= 50
    ORDER BY m.parent_asin

) TO '{OUTPUT_FILE.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Processing raw metadata and reviews...")
print("Filter: review_count >= 50")
print()

con.execute(query)

count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{OUTPUT_FILE.as_posix()}')"
).fetchone()[0]

print(f"Products written : {count:,}")
print(f"Output file      : {OUTPUT_FILE}")
print()

# Basic validation
stats = con.execute(
    f"""
    SELECT
        MIN(review_count),
        MAX(review_count),
        AVG(review_count),
        COUNT(DISTINCT parent_asin)
    FROM read_parquet('{OUTPUT_FILE.as_posix()}')
    """
).fetchone()

print("Validation")
print("-" * 70)
print(f"Minimum reviews  : {stats[0]:,}")
print(f"Maximum reviews  : {stats[1]:,}")
print(f"Mean reviews     : {stats[2]:,.2f}")
print(f"Unique products  : {stats[3]:,}")

print()
print("=" * 70)
print("DONE")
print("=" * 70)

con.close()