
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np

# Load customer reviews
df = pd.read_csv("data/raw/customer_reviews.csv")
df["text"] = df["text"].fillna("").astype(str)

# Load a compact sentence embedding model
print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert review text into embeddings
print(f"Creating embeddings for {len(df)} reviews...")

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Save embeddings and corresponding review data
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "review_embeddings.npy", embeddings)
df.to_csv(output_dir / "reviews_for_retrieval.csv", index=False)

print("\nEmbedding creation completed!")
print("Embedding shape:", embeddings.shape)
print("Saved embeddings to:", output_dir / "review_embeddings.npy")
print("Saved reviews to:", output_dir / "reviews_for_retrieval.csv")
