import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Context
{context}

# Question
{question}

# Answer
{answer}

# Instruction
Please assess if the answer is standalone. That means the answer should be understood only by the given context and question. In particular it should not contain any coreferences, acronyms or abbreviations that cannot be found in the context.

Is the answer standalone? Choose one of the ratings below.
A: Yes, the answer can be understood without looking at the original.
B: No, the answer contains confusing aspects (e.g., unresolved coreferences, abbreviations/acronyms).

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}
""".strip()


def get_messages(qa_pair):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(
        context=qa_pair["target"],
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
        "A": "simplicity_standalone_1",
        "B": "simplicity_standalone_2",
    }
    return {
        "simplicity_standalone": answer_mapping[response["rating"]],
        "simplicity_standalone_rationale": response["motivation"],
    }
