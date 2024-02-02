import re

SYSTEM_PROMPT = """
Statement: {sent}

List all the facts we explicitly know from the statement. Make each fact as atomic as possible.
""".strip()


def generation_params():
    return {
        "model": "gpt-4-0613",
        "temperature": 0,
        "max_tokens": 512,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def get_messages(sent: str):
    prompt = SYSTEM_PROMPT.format(sent=sent)

    return [
        {"role": "system", "content": prompt},
    ]


def parse_response(response):
    if isinstance(response, dict):
        response = response["choices"][0]["message"]["content"]

    return [
        match.group(1).strip()
        for match in re.finditer(r"^\d\.(.*)", response, re.MULTILINE)
    ]
