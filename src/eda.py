
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
df = pd.read_csv("data/raw/customer_reviews.csv")

# 2. Prepare output folder
output_dir = Path("reports/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# 3. Calculate review length
df["review_length"] = df["text"].str.split().str.len()

# 4. Display basic information
print("DATASET OVERVIEW")
print("-" * 40)
print("Total reviews:", len(df))
print("\nSentiment distribution:")
print(df["label"].value_counts().rename(index={
    0: "Negative",
    1: "Positive"
}))
print("\nReview length statistics:")
print(df["review_length"].describe().round(2))

# 5. Plot sentiment distribution
sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.xticks([0, 1], ["Negative", "Positive"])
plt.title("Customer Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(output_dir / "sentiment_distribution.png", dpi=150)
plt.close()

# 6. Plot review length distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="review_length", bins=30, kde=True)
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")
plt.tight_layout()
plt.savefig(output_dir / "review_length_distribution.png", dpi=150)
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in:", output_dir)
