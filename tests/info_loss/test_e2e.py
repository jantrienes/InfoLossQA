import textwrap

from info_loss.e2e import match_span, qa2thresh


def test_qa2thresh():
    original = "Enrolled subjects were randomized to receive either 2,000 lU or 4,000 IU of oral vitamin D3 daily for a total of 90 days. The Short IBD Questionnaire (SIBDQ) for quality of life, the Partial Mayo Score for UC disease activity and serum lab tests were compared between the two treatment groups."
    simplification = "Participants were randomly given either 2,000 IU or 4,000 IU of vitamin D3 daily for 90 days. Researchers used a survey about participant's quality of life and conducted lab tests between the two groups."

    qa_pairs = [
        {
            "original": "oral",
            "rationale": "The simplification does not mention that the Vitamin D was administered orally.",
            "question": "How was the medication administered?",
            "answer": "The vitamin D was administered orally.",
            "category": "omission",
        },
        {
            "original": "Partial Mayo Score for UC disease activity",
            "rationale": "The simplification does not mention that the Partial Mayo Score was used.",
            "question": "Besides quality of life survey and the serum tests, what other measures did the researchers use?",
            "answer": "The researchers also tracked the partial mayo for UC diseases. UC stands for ulcerative colitis and is an inflammatory bowel disease. The mayo score indicates how active or severe the disease is.",
            "category": "omission",
        },
        {
            "original": "Short IBD Questionnaire (SIBDQ) for quality of life",
            "simplification": "a survey about participant's quality of life",
            "rationale": "The specific type of survey (SIBDQ) is not mentioned in the simplification.",
            "question": "What specific survey was used to assess quality of life?",
            "answer": "The Short Inflammatory Bowel Disease Questionnaire (SIBDQ) was used which gives insights about the physical, social, and emotional status of patients with bowel diseases.",
            "category": "concept",
        },
    ]
    actual = qa2thresh(
        sample_id=1, original=original, simplification=simplification, qa_pairs=qa_pairs
    )
    expected = {
        "id": 1,
        "source": "Enrolled subjects were randomized to receive either 2,000 lU or 4,000 IU of oral vitamin D3 daily for a total of 90 days. The Short IBD Questionnaire (SIBDQ) for quality of life, the Partial Mayo Score for UC disease activity and serum lab tests were compared between the two treatment groups.",
        "target": "Participants were randomly given either 2,000 IU or 4,000 IU of vitamin D3 daily for 90 days. Researchers used a survey about participant's quality of life and conducted lab tests between the two groups.",
        "edits": [
            {
                "id": 0,
                "category": "omission",
                "annotation": {
                    "question": "How was the medication administered?",
                    "answer": "The vitamin D was administered orally.",
                    "rationale": "The simplification does not mention that the Vitamin D was administered orally.",
                    "input_raw": "oral",
                    "input_matched": "oral",
                    "input_matched_similarity": 1,
                },
                "input_idx": [[76, 80]],
                "prediction_errors": [],
            },
            {
                "id": 1,
                "category": "omission",
                "annotation": {
                    "question": "Besides quality of life survey and the serum tests, what other measures did the researchers use?",
                    "answer": "The researchers also tracked the partial mayo for UC diseases. UC stands for ulcerative colitis and is an inflammatory bowel disease. The mayo score indicates how active or severe the disease is.",
                    "rationale": "The simplification does not mention that the Partial Mayo Score was used.",
                    "input_raw": "Partial Mayo Score for UC disease activity",
                    "input_matched": "Partial Mayo Score for UC disease activity",
                    "input_matched_similarity": 1,
                },
                "input_idx": [[183, 225]],
                "prediction_errors": [],
            },
            {
                "id": 2,
                "category": "concept",
                "annotation": {
                    "question": "What specific survey was used to assess quality of life?",
                    "answer": "The Short Inflammatory Bowel Disease Questionnaire (SIBDQ) was used which gives insights about the physical, social, and emotional status of patients with bowel diseases.",
                    "rationale": "The specific type of survey (SIBDQ) is not mentioned in the simplification.",
                    "input_raw": "Short IBD Questionnaire (SIBDQ) for quality of life",
                    "input_matched": "Short IBD Questionnaire (SIBDQ) for quality of life",
                    "input_matched_similarity": 1,
                    "output_raw": "a survey about participant's quality of life",
                    "output_matched": "a survey about participant's quality of life",
                    "output_matched_similarity": 1,
                },
                "input_idx": [[126, 177]],
                "output_idx": [[111, 155]],
                "prediction_errors": [],
            },
        ],
    }
    assert actual == expected


def test_match_span():
    doc = "This is a test document"
    query = "this is a test"
    start, end, similarity = match_span(doc, query)
    assert doc[start:end] == "This is a test"
    assert similarity == 1

    doc = "This is a test document"
    query = "this is test"
    start, end, similarity = match_span(doc, query)
    assert doc[start:end] == "This is a test"
    assert round(similarity, 2) == 0.92

    doc = textwrap.dedent(
        """
        BACKGROUND.
        Metabolic syndrome is a cluster of common cardiovascular risk factors that includes hypertension and insulin resistance. Hypertension and diabetes mellitus are frequent comorbidities and, like metabolic syndrome, increase the risk of cardiovascular events. Telmisartan, an antihypertensive agent with evidence of partial peroxisome proliferator-activated receptor activity-gamma (PPARγ) activity, may improve insulin sensitivity and lipid profile in patients with metabolic syndrome.

        METHODS.
        In a double-blind, parallel-group, randomized study, patients with World Health Organization criteria for metabolic syndrome received once-daily doses of telmisartan (80 mg, n = 20) or losartan (50 mg, n = 20) for 3 months. At baseline and end of treatment, fasting and postprandial plasma glucose, insulin sensitivity, glycosylated haemoglobin (HBA1c) and 24-hour mean systolic and diastolic blood pressures were determined.

        RESULTS.
        Telmisartan, but not losartan, significantly (p < 0.05) reduced free plasma glucose, free plasma insulin, homeostasis model assessment of insulin resistance and HbAic. Following treatment, plasma glucose and insulin were reduced during the oral glucose tolerance test by telmisartan, but not by losartan. Telmisartan also significantly reduced 24-hour mean systolic blood pressure (p < 0.05) and diastolic blood pressure (p < 0.05) compared with losartan.

        CONCLUSION.
        As well as providing superior 24-hour blood pressure control, telmisartan, unlike losartan, displayed insulin-sensitizing activity, which may be explained by its partial PPARγ activity.
        """
    ).strip()

    query = "Partial Mayo Score for UC disease activity"
    start, end, similarity = match_span(doc, query)
    assert doc[start:end] == "its partial PPARγ activity."
    assert round(similarity, 2) == 0.55

    query = "The study was double-blind, parallel-group, randomized"
    start, end, similarity = match_span(doc, query)
    assert doc[start:end] == "a double-blind, parallel-group, randomized"
    assert round(similarity, 2) == 0.88
