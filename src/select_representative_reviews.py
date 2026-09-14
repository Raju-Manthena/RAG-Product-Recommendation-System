from pathlib import Path
import duckdb

BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "products_ge50.parquet"
)

REVIEW_GLOB = (
    BASE_DIR
    / "data"
    / "raw"
    / "raw_review_Cell_Phones_and_Accessories"
    / "*.parquet"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "representative_reviews.parquet"
)

con = duckdb.connect()

print("=" * 70)
print("SELECTING REPRESENTATIVE REVIEWS")
print("=" * 70)

query = f"""
COPY (
    WITH ranked_reviews AS (
        SELECT
            r.parent_asin,
            r.asin,
            r.rating,
            r.title,
            r.text,
            r.helpful_vote,
            r.verified_purchase,
            r.timestamp,

            ROW_NUMBER() OVER (
                PARTITION BY r.parent_asin
                ORDER BY
                    r.helpful_vote DESC,
                    r.timestamp DESC
            ) AS review_rank

        FROM read_parquet('{REVIEW_GLOB.as_posix()}') AS r

        INNER JOIN read_parquet('{PRODUCT_FILE.as_posix()}') AS p
            ON r.parent_asin = p.parent_asin
    )

    SELECT
        parent_asin,
        asin,
        rating,
        title,
        text,
        helpful_vote,
        verified_purchase,
        timestamp,
        review_rank

    FROM ranked_reviews
    WHERE review_rank <= 5

    ORDER BY parent_asin, review_rank

) TO '{OUTPUT_FILE.as_posix()}'
(FORMAT PARQUET, COMPRESSION ZSTD);
"""

print("Selecting top 5 reviews per product...")
print("Ranking: helpful_vote DESC, timestamp DESC")
print()

con.execute(query)

# Validation
stats = con.execute(
    f"""
    SELECT
        COUNT(*) AS total_reviews,
        COUNT(DISTINCT parent_asin) AS products,
        AVG(review_rank) AS avg_rank,
        MIN(review_rank) AS min_rank,
        MAX(review_rank) AS max_rank
    FROM read_parquet('{OUTPUT_FILE.as_posix()}')
    """
).fetchone()

print("Validation")
print("-" * 70)
print(f"Selected reviews : {stats[0]:,}")
print(f"Products covered : {stats[1]:,}")
print(f"Average rank     : {stats[2]:.2f}")
print(f"Minimum rank     : {stats[3]:,}")
print(f"Maximum rank     : {stats[4]:,}")

print()
print("Review count distribution")
print("-" * 70)

distribution = con.execute(
    f"""
    SELECT
        review_rank,
        COUNT(*) AS count
    FROM read_parquet('{OUTPUT_FILE.as_posix()}')
    GROUP BY review_rank
    ORDER BY review_rank
    """
).fetchall()

for rank, count in distribution:
    print(f"Rank {rank}: {count:,}")

print()
print(f"Output file: {OUTPUT_FILE}")

print("=" * 70)
print("DONE")
print("=" * 70)

con.close()