import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Original
{original}

# Simplification
{simplification}

# Question
{question}

# Answer
{answer}

# Instruction
We consider a special QA-generation task in the context of text simplification. The QA-pairs should reveal to lay readers what information a simplified text lacks relative to its original version. The question asks for the missing information, and the answer provides it with material from the original text. Each question covers one instance of information loss. We distinguish between two types of information loss:

1. Deletions. Pieces of information which were not included in the simplification.
2. Oversimplification. Pieces of information that were simplified to the extent that they are vague or devoid of their original meaning.

For oversimplification, the location where information loss occurs in the simplified text is denoted by the <vague>...</vague> marker. Your task is to evaluate the marker position.

To what extent does the question relate to the marked span? Choose one of the ratings below.
A: Good, the question directly elaborates on the marked span
B: Unrelated, the question does not relate to the marked span
C: Missing, there should be a marker because the simplified text does discuss the topic, albeit in an oversimplified way (please specify...).
D: Correctly absent, there should not be a marker because the topic is not part of the simplified text.

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "(the rationale)",
    "rating": "(the rating)"
}}
""".strip()


def get_messages(qa_pair):
    system_prompt = SYSTEM_PROMPT

    rationale = qa_pair["target_label"]
    if rationale and len(rationale) > 0:
        rationale = rationale[0]
        r_start = rationale["start"]
        r_end = rationale["end"]
        simplification = (
            qa_pair["target"][:r_start]
            + "<vague>"
            + qa_pair["target"][r_start:r_end]
            + "</vague>"
            + qa_pair["target"][r_end:]
        )
    else:
        simplification = qa_pair["target"]

    user_prompt = USER_PROMPT.format(
        original=qa_pair["source"],
        simplification=simplification,
        question=qa_pair["question"],
        answer=qa_pair["answer"],
        rationale=rationale,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_response(response):
    if isinstance(response, dict):
        response = response["choices"][0]["message"]["content"]

    response = json.loads(response)
    answer_mapping = {
        "A": "givenness_location_1",
        "B": "givenness_location_2",
        "C": "givenness_location_3",
        "D": "givenness_location_na",
    }
    return {
        "givenness_location": answer_mapping[response["rating"]],
        "givenness_location_rationale": response["motivation"],
    }
