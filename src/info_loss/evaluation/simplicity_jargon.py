import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Simplification
{simplification}

# Question
{question}

# Answer
{answer}

# Instruction
We consider a special QA-generation task in the context of text simplification. The QA-pairs should reveal to lay readers what information a simplified text ("simplification") lacks relative to its original version. The question asks for the missing information, and the answer provides it with material from the original text.

Please evaluate if the answer contains any jargon which an average lay person may find difficult to understand. The answer should be seen in the context of the simplification.

Does the answer contain jargon? Choose one of the ratings below.
A: The answer is jargon-free
B: The answer contains jargon but it is adequately explained in the answer
C: The answer contains jargon but it is adequately explained in the simplified text
D: The answer contains unexplained jargon


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
        simplification=qa_pair["target"],
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
        "A": "simplicity_jargon_1",
        "B": "simplicity_jargon_2",
        "C": "simplicity_jargon_3",
        "D": "simplicity_jargon_4",
    }
    return {
        "simplicity_jargon": answer_mapping[response["rating"]],
        "simplicity_jargon_rationale": response["motivation"],
    }
