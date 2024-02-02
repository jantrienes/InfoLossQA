from textwrap import dedent

from info_loss.prompts import e2e_llama


def test_parse_response():
    response = dedent(
        """
        [
            {
                "category": "imprecise/fuzzy concepts",
                "original": "Test",
                "simplification": null,
                "rationale": "Rationale",
                "question": "Question",
                "answer": "Answer"
            }
        ]
        """
    ).strip()
    assert e2e_llama.parse_response(response) == [
        {
            "category": "concept",
            "original": "Test",
            "simplification": None,
            "rationale": "Rationale",
            "question": "Question",
            "answer": "Answer",
        }
    ]


def test_parse_response_trailing_comma():
    response = dedent(
        """
        [
            {
                "category": "omission",
                "original": "Test",
                "simplification": null,
                "rationale": "Rationale",
                "question": "Question",
                "answer": "Answer"
            },
        ]
        """
    ).strip()
    assert e2e_llama.parse_response(response) == [
        {
            "category": "omission",
            "original": "Test",
            "simplification": None,
            "rationale": "Rationale",
            "question": "Question",
            "answer": "Answer",
        }
    ]


def test_parse_response_incomplete():
    response = dedent(
        """
        [
            {
                "category": "omission",
                "original": "Test",
                "simplification": null,
                "rationale": "Rationale",
                "question": "Question",
                "answer": "Answer"
            },
            {
                "category":
        """
    ).strip()
    assert e2e_llama.parse_response(response) == [
        {
            "category": "omission",
            "original": "Test",
            "simplification": None,
            "rationale": "Rationale",
            "question": "Question",
            "answer": "Answer",
        }
    ]
