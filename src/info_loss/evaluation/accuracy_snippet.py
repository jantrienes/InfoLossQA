import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Document
{text}

# Question
{question}

# Answer Snippet
{answer_snippet}

# Instruction
We consider a document-grounded QA-generation task. Consider the above document, question and extractive "answer snippet". Ideally a question should be specific enough so that there is a singular answer. If the question is ambiguous or vague (i.e., there are multiple valid answers), its answer has a high chance of being incomplete or only partially answering the question.

Does the answer snippet correctly answer the question? Imagine it was rephrased into a fluent answer. Choose one of the ratings below.
A: Yes
B: Partially, the answer snippet is related but misses information
C: No

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
        answer_snippet=qa_pair["source_label"][0]["text"],
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
        "A": "accuracy_snippet_1",
        "B": "accuracy_snippet_2",
        "C": "accuracy_snippet_3",
    }
    return {
        "accuracy_snippet": answer_mapping[response["rating"]],
        "accuracy_snippet_rationale": response["motivation"],
    }
