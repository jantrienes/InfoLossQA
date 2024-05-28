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
We consider a document-grounded QA-generation task. Consider the above document, question and answer. Please assess if the provided answer has any hallucinations. Hallucinations are information or claims that cannot be traced back to the original. Disregard general background explanations and elaborations.

Does the provided answer have any hallucinations? Choose one of the ratings below.
A: Good, there are no hallucinations
B: Bad, the answer contains hallucinations

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
        "A": "hallucinations_answer_1",
        "B": "hallucinations_answer_2",
    }
    return {
        "hallucinations_answer": answer_mapping[response["rating"]],
        "hallucinations_answer_rationale": response["motivation"],
    }
