
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

df = pd.read_csv("data/raw/customer_reviews.csv")

print("Loading Transformer model...")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

texts = df["text"].fillna("").astype(str).tolist()
predicted_labels = []
confidence_scores = []

print(f"Predicting sentiment for {len(texts)} reviews...")

for start in range(0, len(texts), 32):
    batch = texts[start:start + 32]

    results = sentiment_pipeline(
        batch,
        truncation=True,
        max_length=256
    )

    for result in results:
        predicted_labels.append(
            1 if result["label"] == "POSITIVE" else 0
        )
        confidence_scores.append(result["score"])

df["predicted_label"] = predicted_labels
df["confidence"] = confidence_scores

output_path = Path("data/processed/sentiment_predictions.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

accuracy = accuracy_score(df["label"], df["predicted_label"])

print("\nMODEL EVALUATION")
print("-" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:")

print(classification_report(
    df["label"],
    df["predicted_label"],
    labels=[0, 1],
    target_names=["Negative", "Positive"],
    zero_division=0
))

print("Predictions saved to:", output_path)
print("Total reviews processed:", len(df))
