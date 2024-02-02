from textwrap import dedent

from info_loss.prompts.question_generation import parse_response


def test_parse_response():
    response = dedent(
        """
        - Fact: "RAPID-PsA is a double-blind trial."
        - Question: What type of medical trial is the RAPID-PsA?
        - Answer: RAPID-PsA is a double-blind type of trial. In a double-blind trial, neither the patients nor the doctors know who is getting the real drug and who is getting a placebo, this helps to prevent bias in the results.

        - Fact: "All the patients had active PsA."
        - Question: What was the health condition of the patients involved in the trial?
        - Answer: All the patients that participated in the trial were experiencing active symptoms of psoriatic arthritis.
        """.strip()
    )
    qas = parse_response(response)
    assert qas[0] == {
        "fact": '"RAPID-PsA is a double-blind trial."',
        "question": "What type of medical trial is the RAPID-PsA?",
        "answer": "RAPID-PsA is a double-blind type of trial. In a double-blind trial, neither the patients nor the doctors know who is getting the real drug and who is getting a placebo, this helps to prevent bias in the results.",
    }

    assert qas[1] == {
        "fact": '"All the patients had active PsA."',
        "question": "What was the health condition of the patients involved in the trial?",
        "answer": "All the patients that participated in the trial were experiencing active symptoms of psoriatic arthritis.",
    }
