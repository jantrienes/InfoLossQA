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

# Instruction
We consider a special QA-generation task in the context of text simplification. The QA-pairs should reveal to lay readers what information a simplified text lacks relative to its original version. The question asks for the missing information, and the answer provides it with material from the original text.

In this setting, answerability is defined as follows: a question ("question") must be answerable with the original text ("original") but not answerable or only vaguely answerable with the simplified text ("simplification").

To what extent is the question answerable with the simplified text? The benchmark for this is the answer on the original text. Choose one of the ratings below.
A: Fully answerable. Asking the question on the simplified text would give the same answer or a closely paraphrased answer as on the original.
B: Partly or vaguely answerable. The simplified text gives some relevant information, but is less specific or exhaustive than the original.
C: Unanswerable.

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}
""".strip()


def get_messages(qa_pair):
    system_prompt = SYSTEM_PROMPT

    user_prompt = USER_PROMPT.format(
        original=qa_pair["source"],
        simplification=qa_pair["target"],
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
        "A": "relevance_target_1",
        "B": "relevance_target_2",
        "C": "relevance_target_3",
    }
    return {
        "relevance_target": answer_mapping[response["rating"]],
        "relevance_target_rationale": response["motivation"],
    }
