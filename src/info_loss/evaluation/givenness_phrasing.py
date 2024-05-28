import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Simplification
{simplification}

# Question
{question}

# Instruction
We consider a special QA-generation task in the context of text simplification. The QA-pairs should reveal to lay readers what information a simplified text lacks relative to its original version. The question asks for the missing information, and the answer provides it with material from the original text.

Please evaluate the "Givenness" of a question. A question should be interpretable for a lay reader. It should only contain concepts (entities, events, or states) that were mentioned in the question context, or that are generally known or inferable from mentioned ones. The question context is defined as the simplified text up to and including the special [QUESTION] marker. For this evaluation, please pretend that you only see the simplification and the question.

Choose one of the ratings below.
A: Good (reader focused, no new concepts)
B: Bad (e.g., question introduces new concepts, answer leakage, hallucinations)

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}
""".strip()


def get_messages(qa_pair):
    system_prompt = SYSTEM_PROMPT

    rationale = qa_pair["target_label"]
    if rationale and len(rationale) > 0:
        end_offset = rationale[0]["end"]
        simplification = (
            qa_pair["target"][:end_offset]
            + " [QUESTION] "
            + qa_pair["target"][end_offset:]
        )
    else:
        simplification = qa_pair["target"] + " [QUESTION]"

    user_prompt = USER_PROMPT.format(
        simplification=simplification, question=qa_pair["question"]
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
        "A": "givenness_phrasing_1",
        "B": "givenness_phrasing_2",
    }
    return {
        "givenness_phrasing": answer_mapping[response["rating"]],
        "givenness_phrasing_rationale": response["motivation"],
    }
