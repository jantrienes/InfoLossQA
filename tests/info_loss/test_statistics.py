from textwrap import dedent

from info_loss.statistics import Section, parse_sections


def test_parse_sections():
    text = """
    BACKGROUND.
    First section.

    METHODS.
    Section two.

    RESULTS.
    Section three.

    CONCLUSION.
    Last section...
    """
    text = dedent(text).strip()

    assert parse_sections(text) == [
        Section(
            span=(0, 28), title="BACKGROUND.", text="BACKGROUND.\nFirst section.\n\n"
        ),
        Section(span=(28, 51), title="METHODS.", text="METHODS.\nSection two.\n\n"),
        Section(span=(51, 76), title="RESULTS.", text="RESULTS.\nSection three.\n\n"),
        Section(
            span=(76, 103), title="CONCLUSION.", text="CONCLUSION.\nLast section..."
        ),
    ]
