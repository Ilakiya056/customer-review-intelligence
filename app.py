
import streamlit as st
from src.search_reviews import search_reviews
from src.rag_answer import generate_answer

st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="🔎",
    layout="wide"
)

st.title("Customer Review Intelligence")
st.write(
    "Explore movie reviews using Transformer sentiment analysis "
    "and semantic search."
)

question = st.text_input(
    "What would you like to know about the reviews?",
    placeholder="Example: What are the common complaints?"
)

if st.button("Analyze Reviews"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching relevant reviews..."):
            answer = generate_answer(question)

            st.subheader("Review Insights")
            st.write(answer)

            st.subheader("Most Relevant Reviews")
            results = search_reviews(question)

            for _, row in results.iterrows():
                sentiment = (
                    "Positive" if row["label"] == 1 else "Negative"
                )

                with st.expander(
                    f"{sentiment} review | Similarity: "
                    f"{row['similarity']:.3f}"
                ):
                    st.write(row["text"])

st.caption(
    "Demo project using a Rotten Tomatoes movie-review dataset. "
    "Retrieved review themes are preliminary keyword-based insights."
)
