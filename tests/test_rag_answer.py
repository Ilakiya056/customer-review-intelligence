
import pandas as pd
from unittest.mock import patch

from src.rag_answer import generate_answer


def test_empty_results():
    empty_results = pd.DataFrame(
        columns=["text", "label", "similarity"]
    )

    with patch(
        "src.rag_answer.search_reviews",
        return_value=empty_results
    ):
        answer = generate_answer("What do viewers like?")

    assert answer == "No relevant reviews were found."


def test_positive_question_hides_complaint_themes():
    fake_results = pd.DataFrame({
        "text": [
            "The film has predictable writing.",
            "The performances are thrilling."
        ],
        "label": [1, 1],
        "similarity": [0.9, 0.8]
    })

    with patch(
        "src.rag_answer.search_reviews",
        return_value=fake_results
    ):
        answer = generate_answer(
            "What do viewers like about these movies?"
        )

    assert "Positive themes are not yet detected automatically." in answer
    assert "Possible common complaint themes:" not in answer
