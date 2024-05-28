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
Is the answer standalone? That means the answer should be understood only by the given context and question. In particular it should not contain any coreferences, acronyms or abbreviations that cannot be found in the context. Choose one of the ratings below.

A: Yes
B: No

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}
""".strip()


def generation_params():
    return {
        "model": "gpt-4-0125-preview",
        "temperature": 0,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


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
        "A": "simplicity_standalone_2",  # bad
        "B": "simplicity_standalone_1",  # good
    }
    return {
        "simplicity_standalone": answer_mapping[response["rating"]],
        "simplicity_standalone_rationale": response["motivation"],
    }
