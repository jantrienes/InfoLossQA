import textwrap

from info_loss.prompts.e2e_gpt4 import parse_response

response = """
## Omissions
- Original: "oral"
- Rationale: The simplification does not mention that the Vitamin D was administered orally.
- Question: How was the medication administered?
- Answer: The vitamin D was administered orally.

- Original: "Partial Mayo Score for UC disease activity"
- Rationale: The simplification does not mention that the Partial Mayo Score was used.
- Question: Besides quality of life survey and the serum tests, what other measures did the researchers use?
- Answer: The researchers also tracked the partial mayo for UC diseases. UC stands for ulcerative colitis and is an inflammatory bowel disease. The mayo score indicates how active or severe the disease is.

## Imprecise/fuzzy concepts
- Original: "Short IBD Questionnaire (SIBDQ) for quality of life"
- Simplification: "a survey about participant's quality of life"
- Rationale: The specific type of survey (SIBDQ) is not mentioned in the simplification.
- Question: What specific survey was used to assess quality of life?
- Answer: The Short Inflammatory Bowel Disease Questionnaire (SIBDQ) was used which gives insights about the physical, social, and emotional status of patients with bowel diseases.
""".strip()


def test_parse_response():
    expected = [
        {
            "original": "oral",
            "rationale": "The simplification does not mention that the Vitamin D was administered orally.",
            "question": "How was the medication administered?",
            "answer": "The vitamin D was administered orally.",
            "category": "omission",
        },
        {
            "original": "Partial Mayo Score for UC disease activity",
            "rationale": "The simplification does not mention that the Partial Mayo Score was used.",
            "question": "Besides quality of life survey and the serum tests, what other measures did the researchers use?",
            "answer": "The researchers also tracked the partial mayo for UC diseases. UC stands for ulcerative colitis and is an inflammatory bowel disease. The mayo score indicates how active or severe the disease is.",
            "category": "omission",
        },
        {
            "original": "Short IBD Questionnaire (SIBDQ) for quality of life",
            "simplification": "a survey about participant's quality of life",
            "rationale": "The specific type of survey (SIBDQ) is not mentioned in the simplification.",
            "question": "What specific survey was used to assess quality of life?",
            "answer": "The Short Inflammatory Bowel Disease Questionnaire (SIBDQ) was used which gives insights about the physical, social, and emotional status of patients with bowel diseases.",
            "category": "concept",
        },
    ]
    actual = parse_response(response)
    assert actual == expected


def test_parse_reponse_special():
    response = textwrap.dedent(
        """
        ## Omissions
        - Original: "They were divided into four groups as follows: a, b, c, d"
        - Rationale: The simplification does not mention...
        - Question: What were the specific concentrations of remifentanil and sufentanil used in each group?
        - Answer: The concentrations were as follows: Group I had 4 ng/ml...

        ## Imprecise/fuzzy concepts
        - Original: "postoperative eye-opening and extubation time"
        - Simplification: "time taken for patients to open their eyes and to remove the breathing tube post surgery"
        - Rationale: The simplification does not accurately convey the original meaning of "postoperative eye-opening and extubation time".
        - Question: What does "postoperative eye-opening and extubation time" refer to in the study?
        - Answer: It refers to the following: time it took for patients to regain consciousness and have their breathing tube removed after surgery.
        """
    ).strip()

    assert parse_response(response) == [
        {
            "original": "They were divided into four groups as follows: a, b, c, d",
            "rationale": "The simplification does not mention...",
            "question": "What were the specific concentrations of remifentanil and sufentanil used in each group?",
            "answer": "The concentrations were as follows: Group I had 4 ng/ml...",
            "category": "omission",
        },
        {
            "original": "postoperative eye-opening and extubation time",
            "simplification": "time taken for patients to open their eyes and to remove the breathing tube post surgery",
            "rationale": 'The simplification does not accurately convey the original meaning of "postoperative eye-opening and extubation time".',
            "question": 'What does "postoperative eye-opening and extubation time" refer to in the study?',
            "answer": "It refers to the following: time it took for patients to regain consciousness and have their breathing tube removed after surgery.",
            "category": "concept",
        },
    ]
