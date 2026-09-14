import streamlit as st

from src.retrieval import HybridRetriever
from src.context_builder import build_context
from src.gemini_generator import GeminiGenerator


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Product Recommendation System",
    page_icon="🛒",
    layout="wide"
)


# --------------------------------------------------
# Load models / indexes once
# --------------------------------------------------

@st.cache_resource
def load_system():

    print("DEBUG: Starting system load", flush=True)

    print("DEBUG: Loading retriever...", flush=True)
    retriever = HybridRetriever()
    print("DEBUG: Retriever loaded", flush=True)

    print("DEBUG: Loading Gemini...", flush=True)
    generator = GeminiGenerator()
    print("DEBUG: Gemini loaded", flush=True)

    return retriever, generator


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("RAG-Based Product Recommendation System")

st.write(
    "Find products using Amazon Cell Phones & Accessories "
    "product information and customer reviews."
)

query = st.text_input(
    "What product are you looking for?",
    placeholder="e.g. wireless earbuds under $50 with good battery life"
)

recommend_button = st.button(
    "Recommend",
    type="primary"
)


# --------------------------------------------------
# Recommendation pipeline
# --------------------------------------------------

if recommend_button:

    if not query.strip():
        st.warning("Please enter a product query.")
        st.stop()

    try:

        with st.spinner("Loading recommendation system..."):
            retriever, generator = load_system()

        with st.spinner("Finding relevant products..."):
            results = retriever.search(query)

        if not results:
            st.warning("No relevant products were found.")
            st.stop()

        context = build_context(results)

        with st.spinner("Generating recommendations..."):
            answer = generator.generate(
                query=query,
                context=context
            )

        # ------------------------------------------
        # Recommendation output
        # ------------------------------------------

        st.markdown(answer)

        # ------------------------------------------
        # Retrieved products
        # ------------------------------------------

        with st.expander("Retrieved Products"):

            for i, result in enumerate(results, start=1):

                st.markdown(
                    f"### {i}. {result['title']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"**Product ID:** "
                        f"{result['parent_asin']}"
                    )

                with col2:
                    st.write(
                        f"**Rating:** "
                        f"{result['average_rating']}"
                    )

                with col3:
                    st.write(
                        f"**Reviews:** "
                        f"{result['review_count']}"
                    )

                st.markdown("---")

    except Exception as e:

        st.error(
            "An error occurred while generating the recommendation."
        )

        st.exception(e)