import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Document
{text}

# Question
{question}

# Answer
{answer}

# Instruction
We consider a document-grounded QA-generation task. Consider the above document, question and answer. Ideally a question should be specific enough so that there is a singular answer. If the question is ambiguous or vague (i.e., there are multiple valid answers), its answer has a high chance of being incomplete or only partially answering the question.

Is the question correctly answered? Choose one of the ratings below.
A: Yes
B: Partially, the answer is related but misses information
C: No

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
        answer=qa_pair["answer"],
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
        "A": "accuracy_answer_1",
        "B": "accuracy_answer_2",
        "C": "accuracy_answer_3",
    }
    return {
        "accuracy_answer": answer_mapping[response["rating"]],
        "accuracy_answer_rationale": response["motivation"],
    }
