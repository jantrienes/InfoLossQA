SYSTEM_PROMPT = """
You are an expert annotator for outputs of text simplification systems. This annotation task is to identify pieces of information that were lost in the simplification process. You will be given two texts: the original and a simplification. Assume that a lay reader only sees the simplification. Identify all things which the reader can NOT learn from the simplification but that they could have learnt from the original.

Please classify each unit of information into one of the following two categories:

1. Omissions: Pieces of information which were not included in the simplification.
2. Imprecise/fuzzy concepts: Pieces of information which are included in the simplification, but that have been simplified to the extent that they became imprecise or completely lost their original meaning.

Afterwards, please write a question-answer pair that would elicit the missing information from the original text. Phrase the question in such a way that a reader can understand it without having seen the original text. It should only contain concepts (entities, events, or states) that were mentioned in the simple text, or concepts that have not been directly mentioned but are generally known or inferable from mentioned ones. The answer should be understandable by an average adult, so please explain technical jargon if necessary. Make each question-answer pair as specific as possible. Ask only about one information unit at a time.

A useful heuristic to decide between omissions and imprecise/fuzzy concepts is to see if the question-answer pair clarifies/expands some topic which is discussed in the simplification. If so, classify it as imprecise/fuzzy concepts, otherwise omission.

Adhere to this output format:
- Original Fragment: <exact span in original>
- Simplification Fragment: null for Omissions OR <exact span in simplification> for Imprecise/fuzzy concepts
- Rationale: <a short rationale that describes what makes this an information loss>
- Question: <the question that elicits missing information>
- Answer: <the answer that provides the missing information from the original span>

Strictly include the above elements. There can be many omissions and imprecise concepts, so be exhaustive. Use simple language in the answer. Cite the original/simplification span EXACTLY so that span == text[text.index(span):text.index(span)+len(span)].

## Original
This study evaluates the effects of vitamin D3 on disease activity and quality of life in ulcerative colitis (UC) patients with hypovitaminosis D. The study was a prospective double-blinded, randomized trial conducted at Community Regional Medical Center, Fresno, CA from 2012–2013. Patients with UC and a serum 25(OH)D level <30 ng/ml were eligible for the study. Enrolled subjects were randomized to receive either 2,000 lU or 4,000 IU of oral vitamin D3 daily for a total of 90 days. The Short IBD Questionnaire (SIBDQ) for quality of life, the Partial Mayo Score for UC disease activity and serum lab tests were compared between the two treatment groups.

## Simplification
This study looks at whether taking vitamin D3 can affect a particular form of bowel disease (ulcerative colitis) and improve the lives of patients with low levels of vitamin D. The study was carried out at a medical center in Fresno, California, between 2012 and 2013. Patients with this disease and low vitamin D levels were included. Participants were randomly given either 2,000 IU or 4,000 IU of oral vitamin D3 daily for 90 days. Researchers used a survey about participant's quality of life and conducted lab tests between the two groups.
""".strip()

EXAMPLE_OUTPUT = """
## Omissions
- Original Fragment: "Partial Mayo Score for UC disease activity"
- Simplification Fragment: null
- Rationale: the simplification does not mention that the Partial Mayo Score was used.
- Question: Besides quality of life survey and the serum tests, what other measures did the researchers use?
- Answer: The researchers also tracked the partial mayo for UC diseases. UC stands for ulcerative colitis and is an inflammatory bowel disease. The mayo score indicates how active or severe the disease is.

## Imprecise/fuzzy concepts
- Original Fragment: "The study was a prospective double-blinded, randomized trial"
- Simplification Fragment: "The study"
- Rationale: The simplification does not explain the design of the study, it only mentions that it was a "study".
- Question: How did the study control for bias?
- Answer: The study was double-blinded, so that neither the researcher nor the participants knew which treatment each participant received, and it was randomized, meaning the participants were randomly assigned one of the treatments.

- Original Fragment: "a serum 25(OH)D level <30 ng/ml"
- Simplification Fragment: "low vitamin D levels"
- Rationale: the simplification does not explain the inclusion criteria, namely how low the vitamin D levels of eligible patients were.
- Question: How low were the vitamin D levels in patients that were included in the study?
- Answer: Participants in the study all had less than 30ng/ml of vitamin D, which is below the minimum recommendation of vitamin D levels in the body.

- Original Fragment: "Short IBD Questionnaire (SIBDQ) for quality of life"
- Simplification Fragment: "a survey about participant's quality of life"
- Rationale: the specific type of survey (SIBDQ) is not mentioned in the simplification
- Question: What survey was used to measure participants' quality of life?
- Answer: The Short Inflammatory Bowel Disease Questionnaire (SIBDQ) was used which gives insights about the physical, social, and emotional status of patients with bowel diseases.
""".strip()

USER_PROMPT = """
## Original
{original}

## Simplification
{simplification}
""".strip()


def generation_params():
    return {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "do_sample": False,
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "repetition_penalty": 1,
        "max_tokens": 2048,
    }


def get_messages(original: str, simplification: str):
    return [
        {"role": "user", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {
            "role": "user",
            "content": USER_PROMPT.format(
                original=original, simplification=simplification
            ),
        },
    ]


def parse_response(response):
    if isinstance(response, dict):
        response = response["choices"][0]["text"]

    results = []
    current_category = ""
    annotation = {}

    for line in response.split("\n"):
        if line.startswith("## Omissions"):
            current_category = "omission"
        elif line.startswith("## Imprecise/fuzzy concepts"):
            current_category = "concept"
        elif line.startswith("- Original Fragment:"):
            txt = line.removeprefix("- Original Fragment:").strip().strip('"')
            annotation["original"] = txt
        elif line.startswith("- Simplification Fragment:"):
            txt = line.removeprefix("- Simplification Fragment:").strip().strip('"')
            if txt == "null":
                annotation["simplification"] = None
            else:
                annotation["simplification"] = txt
        elif line.startswith("- Rationale:"):
            annotation["rationale"] = line.removeprefix("- Rationale:").strip()
        elif line.startswith("- Question:"):
            annotation["question"] = line.removeprefix("- Question:").strip()
        elif line.startswith("- Answer:"):
            annotation["answer"] = line.removeprefix("- Answer:").strip()
            annotation["category"] = current_category
            results.append(annotation)
            annotation = {}

    for qa in results:
        # Sanity check that all elements are present
        complete = set(qa.keys()) == set(
            [
                "category",
                "original",
                "simplification",
                "rationale",
                "question",
                "answer",
            ]
        )
        if not complete:
            raise ValueError("Failed to parse response.")

    return results
