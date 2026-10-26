from src.search_reviews import search_reviews


def generate_answer(question, top_k=5):
    results = search_reviews(question, top_k=top_k)

    if results.empty:
        return "No relevant reviews were found."

    positive_count = int((results["label"] == 1).sum())
    negative_count = int((results["label"] == 0).sum())
    review_texts = results["text"].fillna("").astype(str).tolist()
    combined_text = " ".join(review_texts).lower()
    question_lower = question.lower()

    complaint_keywords = {
        "Weak acting": [
            "bad acting", "poor acting", "wooden acting",
            "bad performance", "terrible acting"
        ],
        "Weak story or script": [
            "bad story", "weak story", "bad script",
            "poorly written", "cliche", "cliched"
        ],
        "Slow pacing": [
            "too slow", "slow pacing", "tedious", "drags"
        ],
        "Lack of creativity": [
            "unoriginal", "lack of creativity",
            "uncreative", "predictable"
        ],
        "Poor characters": [
            "poorly developed characters",
            "flat characters", "uninteresting characters"
        ]
    }

    positive_question_words = [
        "like", "love", "enjoy", "praise", "good",
        "best", "positive", "favorite", "favourite"
    ]
    negative_question_words = [
        "complaint", "complaints", "dislike", "dislikes",
        "hate", "problem", "problems", "bad", "negative",
        "weakness", "weaknesses", "poor"
    ]

    asking_positive = any(
        word in question_lower for word in positive_question_words
    )
    asking_negative = any(
        word in question_lower for word in negative_question_words
    )

    themes = []
    for theme, keywords in complaint_keywords.items():
        if any(keyword in combined_text for keyword in keywords):
            if asking_positive and not asking_negative:
                continue
            themes.append(theme)

    answer = (
        f"Based on the {len(results)} retrieved reviews, "
        f"{positive_count} are positive and "
        f"{negative_count} are negative.\n\n"
    )

    if themes:
        answer += "Possible common complaint themes:\n"
        for theme in themes:
            answer += f"- {theme}\n"
    elif asking_positive and not asking_negative:
        answer += (
            "Positive themes are not yet detected automatically. "
            "Please review the positive review evidence below.\n"
        )
    else:
        answer += (
            "No common complaint themes were identified "
            "by keyword matching.\n"
        )

    answer += "\nRetrieved review evidence:\n"
    for number, review in enumerate(review_texts, start=1):
        answer += f"\n{number}. {review}\n"

    return answer


if __name__ == "__main__":
    question = input("Ask a question about movie reviews: ")
    print("\nRAG ANSWER")
    print("-" * 60)
    print(generate_answer(question))
