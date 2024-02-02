from info_loss.iaa import Span, char_span_to_token_span, char_spans_to_token_spans


def test_char_span_to_token_span():
    assert char_span_to_token_span("This is an example sentence.", 0, 4) == (0, 1)
    assert char_span_to_token_span("This is an example sentence.", 0, 5) == (
        0,
        1,
    )  # whitespace is not a token
    assert char_span_to_token_span("This is an example sentence.", 0, 6) == (
        0,
        2,
    )  # should include tokens with partial overlap
    assert char_span_to_token_span("This is an example sentence.", 0, 7) == (0, 2)
    assert char_span_to_token_span("This is an example sentence.", 0, 120) == (0, 5)

    text = "This is an example sentence."
    annotations = [
        Span("doc-0", "category_a", 0, 7),
        Span("doc-0", "category_b", 8, 18),
    ]
    expected = [
        Span("doc-0", "category_a", 0, 2),
        Span("doc-0", "category_b", 2, 4),
    ]
    assert char_spans_to_token_spans(text, annotations) == expected
