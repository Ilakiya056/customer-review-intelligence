
from datasets import load_dataset
import pandas as pd
from pathlib import Path

print("Downloading customer reviews...")

dataset = load_dataset(
    "cornell-movie-review-data/rotten_tomatoes"
)

reviews = pd.DataFrame(dataset["train"])

# Keep a manageable sample for our first version
reviews = reviews.sample(
    n=min(2000, len(reviews)),
    random_state=42
)

output_path = Path("data/raw/customer_reviews.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

reviews.to_csv(output_path, index=False)

print("\nDataset downloaded successfully!")
print("Rows:", len(reviews))
print("Columns:", list(reviews.columns))
print("Saved to:", output_path)
print("\nFirst 3 reviews:")
print(reviews.head(3).to_string(index=False))
