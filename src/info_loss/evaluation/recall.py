import json

SYSTEM_PROMPT = """
You are an expert evaluator of QA-generation systems.
""".strip()

USER_PROMPT = """
# Examples

## Reference
Q: What condition did the babies in the study have?
A: The babies in the study were diagnosed with moderate-to-severe bronchiolitis.

## Generated
Q: How was the study conducted?
A: The study was a double-blind, randomized controlled trial on infants (1 to 12 months) who were diagnosed in the emergency department with moderate-to-severe bronchiolitis.

## Output
{{
    "motivation": "The reference (condition of babies) is fully recalled by the generated QA.",
    "rating": "A"
}}

## Reference
Q: What kind of tests were used to assess patients in this study?
A: Participants were assessed with laboratory tests, the United Kingdom screening test, the Michigan neuropathy screening score, and the Michigan diabetic neuropathy score.

## Generated
Q: What method was used to assess the level of neuropathy in the patients?
A: The Michigan neuropathy screening score was used to assess the level of neuropathy in all patients.

## Output
{{
    "motivation": "The generated QA only partially covers the lab tests mentioned in the reference QA.",
    "rating": "B"
}}

## Reference
Q: How much did the special questionnaire overestimate calcium intake compared to the 24-hour recall?
A: The questionnaire overestimated the average total calcium intake by 221 mg/d (milligrams per day) compared to the 24-hour recall across racial groups.

## Generated
Q: What was the variability in daily No calcium intake estimated by the 24-hour recall?
A: The variability or standard deviation in daily calcium intake estimated based on the 24-hour dietary recall was 433 milligrams per day.

## Output
{{
    "motivation": "Related topics, but the generated QA does not include the key information.",
    "rating": "C"
}}

# Input
## Reference
Q: {reference_question}
A: {reference_answer}

## Generated
Q: {predicted_question}
A: {predicted_answer}

# Instruction
You are given two question-answer (QA) pairs: a reference pair ("reference") and a model-generated pair ("generated"). Please assess the degree to which the generated QA recalls the information of the reference QA. Focus on the recall of factual reference information. That means you should disregard additional information in the generated QA. Furthermore, a reference QA may also include term definitions or background explanations which do not have to be recalled.

Please assign one of the following categories:
A: Fully recalled, the model QA gives the same information as the reference QA.
B: Partially recalled, the model QA partially answers the reference QA.
C: Not recalled, there is no overlap in the presented content.

For the rating, only give the letter nothing else. Also provide a brief motivation for your choice (one sentence). Respond according to below JSON format.
{{
    "motivation": "",
    "rating": ""
}}

""".strip()


def get_messages(sample):
    system_prompt = SYSTEM_PROMPT

    user_prompt = USER_PROMPT.format(
        reference_question=sample["reference_question"],
        reference_answer=sample["reference_answer"],
        predicted_question=sample["predicted_question"],
        predicted_answer=sample["predicted_answer"],
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
        "A": "aligned",
        "B": "partial",
        "C": "not_aligned",
    }
    return {
        "recall": answer_mapping[response["rating"]],
        "recall_rationale": response["motivation"],
    }
