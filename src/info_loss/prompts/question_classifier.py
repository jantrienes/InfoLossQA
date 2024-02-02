from typing import List

SYSTEM_PROMPT = """
You are a helpful assistant to classify text into categories.

## Instructions
You are asked to classify questions according to an ontology of question types. The question type reflects the nature of the question. It is NOT determined by the interrogative word of the question. There are 10 question types in total. The definition for each type is shown below. Please select the question type which is most likely for a given question. Only output the category title, not the description.

## Question Types
1. VERIFICATION: Asking for the truthfulness of an event or a concept.
- Was the study double-blinded?
- Was there a trend towards smaller increases in macular pigment for subjects with high baseline values?

2. DISJUNCTION: Asking for the true one given multiple events or concepts, where comparison among options is not needed.
- no example available, match by the description

3. CONCEPT: Asking for a definition of an event or a concept.
- What kind of result is being measured in this study?
- What were the main inclusion criteria for this study?
- What does the WCJ-III test specifically measure?
- Which areas of the brain were analyzed?

4. EXTENT: Asking for the extent or quantity of an event or a concept.
- How reliable are these results?
- How long were the participants observed?
- How much ibuprofen was in the small dose?
- How many young and old people participated in the study?

5. EXAMPLE: Asking for example(s) or instance(s) of an event or a concept.
- What kind of conditions cause corneal neovascularization?

6. COMPARISON: Asking for comparison among multiple events or concepts.
- On what results did the control group do better than the intervention group?
- How did headache of participants receiving ibuprofen compare to those participants that received a placebo?
- How more effective was the arm cranking exercise with and without electrical muscle stimulation?

7. CAUSE: Asking for the cause or reason for an event or a concept.
- What motivates this study?
- Why is EMS being investigated?

8. CONSEQUENCE: Asking for the consequences or results of an event.
- What was the effect of using ibuprofen to treat headaches?
- What were the main findings of the study?
- What does this study tell us about arm cranking with electrical muscle stimulation?

9. PROCEDURAL: Asking for the procedures, tools, or methods by which a certain outcome is achieved.
- What questionnaire was used for this study?
- What kind of lab tests were done?
- How were the patients assigned to a group?
- How were the different drugs administered to participants?

10. JUDGMENTAL: Asking for the opinions of the answerer's own.
- no example available, match by the description
""".strip()


def generation_params():
    return {
        "model": "gpt-4-0613",
        "temperature": 0,
        "max_tokens": 1024,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


def get_messages(questions: List[str]):
    prompt = "## Instances to classify\n"
    for i, question in enumerate(questions):
        prompt += f"{i+1}. {question}\n"
    prompt = prompt.strip()

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def parse_response(response):
    if isinstance(response, dict):
        response = response["choices"][0]["message"]["content"]

    response = [
        item.strip("1234567890. ").lower()
        for item in response.split("\n")
        if item.strip()
    ]
    return response
