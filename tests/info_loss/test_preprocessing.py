from textwrap import dedent

from info_loss.preprocessing import segment_abstract


def test_segment_abstract():
    t = dedent(
        """
        TITLE:
        Metabolic effect of telmisartan...

        ABSTRACT.BACKGROUND:.
        Metabolic syndrome is a cluster...

        ABSTRACT.METHODS.
        In a double-blind, parallel-group...

        ABSTRACT.RESULTS:
        Telmisartan, but not losartan...

        ABSTRACT.CONCLUSION:
        As well as providing superior...
        """
    ).strip()

    expected = dedent(
        """
        BACKGROUND.
        Metabolic syndrome is a cluster...

        METHODS.
        In a double-blind, parallel-group...

        RESULTS.
        Telmisartan, but not losartan...

        CONCLUSION.
        As well as providing superior...
        """
    ).strip()

    title, abstract, sectioned = segment_abstract(t)
    assert title == "Metabolic effect of telmisartan..."
    assert abstract == expected
    assert sectioned == True


def test_segment_text_not_sectioned():
    t = dedent(
        """
        TITLE.
        Impact upon clinical outcomes...

        ABSTRACT.
        Fluorescence in situ hybridization...
        """
    ).strip()

    title, abstract, sectioned = segment_abstract(t)
    assert title == "Impact upon clinical outcomes..."
    assert abstract == "Fluorescence in situ hybridization..."
    assert sectioned == False


def test_segment_text_empty_title():
    t = dedent(
        """
        TITLE.


        ABSTRACT.OBJECTIVE.
        Genetic variants near IRS1 are associated with features...

        ABSTRACT.RESEARCH DESIGN AND METHODS.
        Two variants near IRS1, rs1522813 and rs2943641, were...
        """
    ).strip()

    expected = dedent(
        """
        OBJECTIVE.
        Genetic variants near IRS1 are associated with features...

        RESEARCH DESIGN AND METHODS.
        Two variants near IRS1, rs1522813 and rs2943641, were...
        """
    ).strip()

    title, abstract, sectioned = segment_abstract(t)
    assert title == ""
    assert abstract == expected
    assert sectioned == True


def test_segment_text_double_abstract_title():
    t = dedent(
        """
        TITLE.
        Efficacy and Safety of Tranexamic Acid in Control...

        ABSTRACT.SUMMARY.
        Total knee arthroplasty (TKA) is generally carried...
        """
    ).strip()

    title, abstract, sectioned = segment_abstract(t)
    assert title == "Efficacy and Safety of Tranexamic Acid in Control..."
    assert abstract == "Total knee arthroplasty (TKA) is generally carried..."
    assert sectioned == False


def test_segment_text_nested_abstract_summary():
    t = dedent(
        """
        TITLE.
        Concurrent once-daily versus twice-daily...

        ABSTRACT.SUMMARY.BACKGROUND.
        Concurrent chemoradiotherapy is the standard...

        ABSTRACT.SUMMARY.METHODS.
        The CONVERT trial was an open-label, phase 3...
        """
    ).strip()

    expected = dedent(
        """
        BACKGROUND.
        Concurrent chemoradiotherapy is the standard...

        METHODS.
        The CONVERT trial was an open-label, phase 3...
        """
    ).strip()

    title, abstract, sectioned = segment_abstract(t)
    assert title == "Concurrent once-daily versus twice-daily..."
    assert abstract == expected
    assert sectioned == True
