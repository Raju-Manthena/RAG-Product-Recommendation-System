import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Clean simple HTML entities/tags and excessive whitespace."""
    if text is None:
        return ""

    text = str(text)

    # Common HTML entities
    text = text.replace("&#34;", '"')
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    # Convert HTML line breaks
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_context(
    results: List[Dict],
    max_features_chars: int = 1500,
    max_description_chars: int = 2000,
    max_review_chars: int = 1200
) -> str:
    """
    Build structured context for the LLM from retrieved products.

    Retrieval documents and indexes are not modified.
    This function only controls what information is sent to Gemini.
    """

    context_parts = []

    for rank, result in enumerate(results, start=1):

        product_id = result["parent_asin"]

        title = clean_text(result.get("title", ""))
        category = clean_text(result.get("main_category", ""))
        price = clean_text(result.get("price", ""))
        store = clean_text(result.get("store", ""))
        subtitle = clean_text(result.get("subtitle", ""))

        average_rating = result.get("average_rating", "")
        rating_number = result.get("rating_number", "")
        review_count = result.get("review_count", "")

        features = clean_text(result.get("features", ""))
        description = clean_text(result.get("description", ""))

        # Limit very large fields for the LLM only
        features = features[:max_features_chars]
        description = description[:max_description_chars]

        # Reviews are stored in the product document, so extract them
        # from the retrieved document.
        document = clean_text(result.get("document", ""))

        review_section = ""

        if "Customer Reviews:" in document:
            review_section = document.split(
                "Customer Reviews:", 1
            )[1].strip()

        review_section = review_section[:max_review_chars * 5]

        product_context = (
            f"PRODUCT {rank}\n"
            f"Product ID: {product_id}\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Subtitle: {subtitle}\n"
            f"Price: {price}\n"
            f"Average Rating: {average_rating}\n"
            f"Rating Count: {rating_number}\n"
            f"Review Count: {review_count}\n"
            f"Store: {store}\n"
            f"Features: {features}\n"
            f"Description: {description}\n"
            f"Customer Reviews:\n{review_section}"
        )

        context_parts.append(product_context)

    return "\n\n" + (
        "\n\n" + "=" * 80 + "\n\n"
    ).join(context_parts)