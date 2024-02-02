import argparse
import hashlib
import json
import re
import string
from collections import namedtuple
from pathlib import Path
from typing import List

import nltk
import pandas as pd
import textstat


def md5(s):
    hash = hashlib.md5()
    hash.update(s.encode("utf-8"))
    return hash.hexdigest()


def load_data(path, hash_edit_ids=False):
    path = Path(path)

    if path.is_dir():
        json_files = list(path.glob("*.json"))
        json_files = sorted(json_files)
        data = []

        for json_file in json_files:
            with open(json_file, "r") as file:
                data.extend(json.load(file))
    elif path.suffix == ".json":
        with open(path, "r") as file:
            data = json.load(file)
    else:
        raise ValueError(
            "Not a valid file path. Must be a directory with *.json files, or a single .json file."
        )

    for doc in data:
        doc["sections"] = parse_sections(doc["source"])
        doc["edits"] = [edit for edit in doc["edits"] if edit["category"] != "comment"]

        if hash_edit_ids:
            for edit in doc["edits"]:
                x = edit["annotation"]
                edit["id"] = md5(
                    str(doc["id"])
                    + x.get("question", "")
                    + x.get("answer", "")
                    + x.get("comment", "")
                )
    return data


def load_data_aggregated(annotators, union=False):
    id2doc = {}

    # For each annotator, load annotations. Can either be a single json, or multiple json.
    for annotator_name, annotator_json in annotators:
        for doc in load_data(annotator_json, hash_edit_ids=True):
            if doc["id"] not in id2doc:
                id2doc[doc["id"]] = {
                    "id": doc["id"],
                    "source": doc["source"],
                    "target": doc["target"],
                    "annotations": [],
                }

            edits = {
                "annotator": annotator_name,
                "edits": doc.get("edits", []),
                "_completed": doc.get("_completed", None),
                "_duration": doc.get("_seconds_spent", -1),
            }

            if doc.get("error"):
                edits["error"] = doc["error"]
            id2doc[doc["id"]]["annotations"].append(edits)

    # Merge annotations of all annotators into one "union" annotator.
    # Concatenate names of annotators
    if union:
        for doc in id2doc.values():
            if len(doc["annotations"]) > 0:
                annotators = []
                edits = []
                for annotations in doc["annotations"]:
                    annotators.append(annotations["annotator"])
                    edits.extend(annotations["edits"])

                doc["annotations"] = [
                    {"annotator": " + ".join(annotators), "edits": edits}
                ]

    return id2doc


CLS_BACKGROUND = "Background"
CLS_METHODS = "Methods"
CLS_RESULTS = "Results"
CLS_CONCLUSION = "Conclusion"
CLS_FUNDING = "Funding"
CLS_REGISTRATION = "Registration"
CLS_OTHER = "n/a (Other)"
CLS_MULTIPLE = "n/a (Multiple)"
CLS_UNSECTIONED = "n/a (Unsectioned)"


SECTION_ORDER = {
    CLS_BACKGROUND: 0,
    CLS_METHODS: 1,
    CLS_RESULTS: 2,
    CLS_CONCLUSION: 3,
    CLS_FUNDING: 4,
    CLS_REGISTRATION: 5,
    CLS_OTHER: 6,
    CLS_MULTIPLE: 7,
    CLS_UNSECTIONED: 8,
}

SECTION_MAPPING = {
    "ABSTRACT.": CLS_BACKGROUND,
    "AIM.": CLS_BACKGROUND,
    "AIMS AND OBJECTIVES.": CLS_BACKGROUND,
    "AIMS.": CLS_BACKGROUND,
    "BACKGROUND AND AIMS.": CLS_BACKGROUND,
    "BACKGROUND AND THE PURPOSE OF THE STUDY.": CLS_BACKGROUND,
    "BACKGROUND.": CLS_BACKGROUND,
    "BACKGROUND/AIM.": CLS_BACKGROUND,
    "BACKGROUND/AIMS.": CLS_BACKGROUND,
    "CLASSIFICATION OF EVIDENCE.": CLS_OTHER,
    "CLINICAL RELEVANCE.": CLS_OTHER,
    "CLINICAL TRIALS REGISTRATION.": CLS_REGISTRATION,
    "CLINICALTRIALS.GOV REGISTRY NUMBERS.": CLS_REGISTRATION,
    "CONCLUSION AND RECOMMENDATIONS.": CLS_CONCLUSION,
    "CONCLUSION.": CLS_CONCLUSION,
    "CONCLUSIONS.": CLS_CONCLUSION,
    "CONTEXT.": CLS_BACKGROUND,
    "DESIGN AND METHODS.": CLS_METHODS,
    "DESIGN.": CLS_METHODS,
    "DISCUSSION.": CLS_CONCLUSION,
    "ELECTRONIC SUPPLEMENTARY MATERIAL.": CLS_OTHER,
    "FINDINGS.": CLS_RESULTS,
    "FUNDING.": CLS_FUNDING,
    "INTERPRETATION.": CLS_CONCLUSION,
    "INTERVENTIONS.": CLS_METHODS,
    "INTRODUCTION & OBJECTIVES.": CLS_BACKGROUND,
    "INTRODUCTION.": CLS_BACKGROUND,
    "MAIN FINDINGS.": CLS_RESULTS,
    "MATERIAL AND METHODS.": CLS_METHODS,
    "MATERIALS & METHODS.": CLS_METHODS,
    "MATERIALS AND METHODS.": CLS_METHODS,
    "MEASUREMENTS.": CLS_METHODS,
    "METHOD.": CLS_METHODS,
    "METHODS AND FINDINGS.": CLS_MULTIPLE,
    "METHODS AND MATERIALS.": CLS_METHODS,
    "METHODS AND RESULTS.": CLS_MULTIPLE,
    "METHODS.": CLS_METHODS,
    "OBJECTIVE AND DESIGN.": CLS_BACKGROUND,
    "OBJECTIVE.": CLS_BACKGROUND,
    "OBJECTIVES.": CLS_BACKGROUND,
    "PARTICIPANTS.": CLS_METHODS,
    "PATIENTS AND METHODS.": CLS_METHODS,
    "PATIENTS DESIGN AND MEASUREMENTS.": CLS_METHODS,
    "PURPOSE.": CLS_BACKGROUND,
    "RESEARCH DESIGN AND METHODS.": CLS_METHODS,
    "RESULT.": CLS_RESULTS,
    "RESULTS.": CLS_RESULTS,
    "RESULTS: .": CLS_RESULTS,
    "SETTING.": CLS_METHODS,
    "SETTINGS AND DESIGN.": CLS_METHODS,
    "STATISTICAL ANALYSIS USED.": CLS_METHODS,
    "STATISTICS.": CLS_METHODS,
    "SUBJECT AND METHODS.": CLS_METHODS,
    "TRIAL REGISTRATION NUMBER.": CLS_REGISTRATION,
    "TRIAL REGISTRATION.": CLS_REGISTRATION,
}

Section = namedtuple(
    "Section",
    ["span", "title", "text"],
)


def parse_sections(text: str) -> List[Section]:
    current_start = 0
    sections = []
    for boundary in re.finditer("\n\n", text):
        end = boundary.end()
        section_text = text[current_start:end]
        title = section_text.split("\n", maxsplit=1)[0]
        sections.append(
            Section(span=(current_start, end), title=title, text=section_text)
        )
        current_start = end

    section_text = text[current_start : len(text)]
    title = section_text.split("\n", maxsplit=1)[0]
    sections.append(
        Section(span=(current_start, len(text)), title=title, text=section_text)
    )

    return sections


def span_overlap(x1, x2, y1, y2):
    return x1 < y2 and y1 < x2


def qa_stats(doc, qa):
    question_len = len(nltk.word_tokenize(qa["annotation"]["question"]))
    answer_len = len(nltk.word_tokenize(qa["annotation"]["answer"]))

    input_spans = [doc["source"][start:end] for (start, end) in qa.get("input_idx", [])]
    output_spans = [
        doc["target"][start:end] for (start, end) in qa.get("output_idx", [])
    ]

    input_span_len = (
        len(nltk.word_tokenize(" ".join(input_spans))) if input_spans else None
    )
    output_span_len = (
        len(nltk.word_tokenize(" ".join(output_spans))) if output_spans else None
    )

    sections = []
    if not qa.get("input_idx"):
        sections = ["n/a (parsing error)"]
    elif len(doc["sections"]) == 1:
        sections = [CLS_UNSECTIONED]
    elif len(doc["sections"]) > 1:
        for span in qa["input_idx"]:
            q_start, q_end = span
            for section in doc["sections"]:
                s_start, s_end = section.span
                if span_overlap(q_start, q_end, s_start, s_end):
                    title_normalized = SECTION_MAPPING[section.title]
                    sections.append(title_normalized)

    if len(sections) == 1 or len(set(sections)) == 1:
        primary_section = sections[0]
    elif len(set(sections)) > 1:
        # questions can be in multiple sections:
        # (a) multi-part highlight
        # (b) highlight that crosses section boundary
        # if that is the case, we assign a 'multiple' category
        primary_section = CLS_MULTIPLE

    return {
        "doc_id": doc["id"],
        "category": qa["category"],
        "question_len": question_len,
        "answer_len": answer_len,
        "input_spans": input_spans,
        "output_spans": output_spans,
        "input_span_len": input_span_len,
        "output_span_len": output_span_len,
        "sections": sections,
        "primary_section": primary_section,
    }


def summary_statistics(docs):
    df_qa = pd.DataFrame(qa_stats(doc, qa) for doc in docs for qa in doc["edits"])

    df_stats = {
        "num_documents": len(docs),
        "num_question_answer_pairs": len(df_qa),
        **df_qa["category"]
        .value_counts(normalize=True)
        .rename("num_{}".format)
        .to_dict(),
        "avg_questions_per_document": df_qa.groupby("doc_id").size().mean(),
        "avg_question_length": df_qa["question_len"].mean(),
        "avg_answer_length": df_qa["answer_len"].mean(),
        "avg_input_span_length": df_qa["input_span_len"].mean(),
        "avg_output_span_length": df_qa["output_span_len"].mean(),
    }
    by_category = df_qa.groupby("category").agg(
        num_documents=("doc_id", "nunique"),
        num_question_answer_pairs=("doc_id", "count"),
        avg_questions_per_document=("doc_id", lambda x: x.count() / x.nunique()),
        avg_question_length=("question_len", "mean"),
        avg_answer_length=("answer_len", "mean"),
        avg_input_span_length=("input_span_len", "mean"),
        avg_output_span_length=("output_span_len", "mean"),
    )

    df_stats = pd.DataFrame([df_stats], index=["all"])
    df_stats = pd.concat([df_stats, by_category]).T
    return df_stats


def corpus_statistics(docs):
    df_docs = pd.DataFrame(document_stats(doc) for doc in docs)
    stats = df_docs.mean()
    stats.loc["num_docs"] = len(docs)
    return stats


def document_stats(doc):
    # Pre-tokenize with NLTK
    source = nltk.word_tokenize(doc["source"])
    target = nltk.word_tokenize(doc["target"])

    return {
        "src_len": len(source),
        "tgt_len": len(target),
        "compression": 1 - len(target) / len(source),
        "novelty_uni": novelty(" ".join(source), " ".join(target), n=1),
        "novelty_bi": novelty(" ".join(source), " ".join(target), n=2),
        "src_fre": textstat.flesch_reading_ease(doc["source"]),  # pylint: disable=E1101
        "tgt_fre": textstat.flesch_reading_ease(doc["target"]),  # pylint: disable=E1101
    }


def clean(s):
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s\s+", " ", s)
    s = s.lower()
    return s


def novel_ngrams(a: str, b: str, n=2):
    """Count number of n-grams in b but not in a."""
    a, b = clean(a), clean(b)
    a = set(nltk.ngrams(a.split(), n=n))
    b = set(nltk.ngrams(b.split(), n=n))
    novel = len(b - a)
    total = len(b)
    return novel, total


def novelty(a, b, n):
    """
    Fraction of n-grams in b but not in a.

    If there are no n-grams in b, novelty=0.
    """
    novel, total = novel_ngrams(a, b, n=n)
    if total == 0:
        return 0
    return novel / total


def section_distribution(docs):
    df_qa = pd.DataFrame(qa_stats(doc, qa) for doc in docs for qa in doc["edits"])
    df_filtered = df_qa[df_qa["primary_section"].isin(SECTION_ORDER.keys())]
    df_secs = (
        df_filtered.groupby("category")["primary_section"]
        .value_counts(normalize=True)
        .unstack(level=1)
    )
    df_secs.columns.name = None
    df_secs.index.name = None
    df_secs = df_secs.reindex(
        columns=sorted(df_secs.columns, key=lambda x: SECTION_ORDER[x])
    )
    df_secs.loc["all"] = df_filtered["primary_section"].value_counts(normalize=True)
    df_secs = df_secs.sort_index()
    return df_secs


def main(args):
    docs = load_data(args.input_path)
    df_summary = summary_statistics(docs)
    df_summary = df_summary.round(1)
    print(df_summary.to_string())

    df_secs = section_distribution(docs)
    df_secs = (df_secs * 100).round(1)
    print(df_secs.to_string())


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_path",
        help="Either a directory with *.json files or a single .json file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(arg_parser())
