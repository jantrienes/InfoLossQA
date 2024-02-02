import re
from typing import List

SYSTEM_PROMPT = """
## Original
{original}

## Simplification
{simplification}

## Missing facts
{facts}

The above facts are missing from the simplified text ("Simplification"). For each fact, please write a question-answer pair that would elicit the missing information from the original text ("Original"). Phrase the question in such a way that a reader can understand it without having seen the original text. It should only contain concepts (entities, events, or states) that were mentioned in the simple text, or concepts that have not been directly mentioned but are generally known or inferable from mentioned ones. The answer should be understandable by an average adult, so please explain technical jargon if necessary. Make each question-answer pair as specific as possible and make sure that they are independent of each other. Ask only about one information unit at a time. Do this for all facts, and format your output as follows:

- Fact:
- Question:
- Answer:
""".strip()


def generation_params():
    return {
        "model": "gpt-4-0613",
        "temperature": 1,
        "max_tokens": 4096,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def get_messages(original: str, simplification: str, facts: List[str]):
    facts = "\n".join(f"- {fact}" for fact in facts)
    prompt = SYSTEM_PROMPT.format(
        original=original, simplification=simplification, facts=facts
    )

    return [
        {"role": "system", "content": prompt},
    ]


def parse_response(response):
    if isinstance(response, dict):
        response = response["choices"][0]["message"]["content"]

    results = []

    for qa in response.split("\n\n"):
        if len(qa.strip()) == 0:
            continue

        fact_match = re.search(r"- *Fact:\s?(.*)", qa)
        question_match = re.search(r"- *Question:\s?(.*)", qa)
        answer_match = re.search(r"- *Answer:\s?(.*)", qa)

        results.append(
            {
                "fact": fact_match.group(1),
                "question": question_match.group(1),
                "answer": answer_match.group(1),
            }
        )
    return results
