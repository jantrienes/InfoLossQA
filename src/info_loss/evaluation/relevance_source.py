import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Text
{text}

# Question
{question}

# Instruction
We consider a document-grounded QA-generation task. A question ("question") must be answerable with the text ("text"). It should not ask for background information that cannot be found in the text. Furthermore, the question must be specific enough to lead to a singular answer. There should be one obvious way to answer it.

Is the "question" answerable? Choose one of the ratings below.
A: Yes, and there is a single most obvious answer
B: Yes, but there could be multiple valid answers
C: No, the question is not answerable

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}
""".strip()


def get_messages(qa_pair):
    system_prompt = SYSTEM_PROMPT

    user_prompt = USER_PROMPT.format(
        text=qa_pair["source"],
        question=qa_pair["question"],
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
        "A": "relevance_source_1",
        "B": "relevance_source_2",
        "C": "relevance_source_3",
    }
    return {
        "relevance_source": answer_mapping[response["rating"]],
        "relevance_source_rationale": response["motivation"],
    }
