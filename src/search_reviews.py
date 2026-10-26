
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load reviews and saved embeddings
reviews = pd.read_csv("data/processed/reviews_for_retrieval.csv")
embeddings = np.load("data/processed/review_embeddings.npy")

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def search_reviews(query, top_k=5):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Compare the query with all review embeddings
    similarities = embeddings @ query_embedding

    # Select the five most similar reviews
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = reviews.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results

if __name__ == "__main__":
    question = input("Enter a review search query: ")

    results = search_reviews(question)

    print("\nMOST RELEVANT REVIEWS")
    print("-" * 60)

    for _, row in results.iterrows():
        sentiment = (
            "Positive" if row["label"] == 1 else "Negative"
        )

        print(f"\nSimilarity: {row['similarity']:.3f}")
        print(f"Sentiment: {sentiment}")
        print(f"Review: {row['text']}")
