"""
Metrics for inter-annotator agreement.

Classification:
- Randolph's Kappa
- Fleiss Kappa
- Krippendorff's Alpha

Span-labeling:
- Exact match
- Partial match (NLTK for tokenization)
- Sentence-level agreement (NLTK for sentence splitting)
"""

from collections import defaultdict, namedtuple
from typing import List

import numpy as np
from krippendorff import alpha
from nltk import PunktSentenceTokenizer, WhitespaceTokenizer
from nltk.metrics import masi_distance
from nltk.metrics.agreement import AnnotationTask
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats import inter_rater as irr


def label_encode(raters):
    le = LabelEncoder()
    le.fit([v for rater in raters for v in rater])  # flatten
    raters = [le.transform(rater) for rater in raters]
    raters = np.column_stack(raters)
    return raters


def kappa(raters, method="fleiss"):
    """Calculate Kappa (Fleiss, Randolphs) with statsmodels.

    Parameters
    ----------
    raters : tuple
        Each item in the tuple should be a list of ratings by one rater.
    method : str, optional
        Valid values: {'fleiss', 'randolph'}.
        See: https://www.statsmodels.org/dev/generated/statsmodels.stats.inter_rater.fleiss_kappa.html
    """
    raters = label_encode(raters)
    counts, cats = irr.aggregate_raters(raters)
    return irr.fleiss_kappa(counts, method=method)


def krippendorffs_alpha(raters, level_of_measurement="nominal"):
    """
    Parameters
    ----------
    raters : tuple
        Each item in the tuple should be a list of ratings by one rater.
    level_of_measurement : str or callable
        Steven's level of measurement of the variable.
        It must be one of "nominal", "ordinal", "interval", "ratio", or a callable.
        See: https://github.com/pln-fing-udelar/fast-krippendorff/
    """
    raters = label_encode(raters)
    counts, cats = irr.aggregate_raters(raters)
    return alpha(value_counts=counts, level_of_measurement=level_of_measurement)


Span = namedtuple("Span", ["doc_id", "label", "start", "end"])


def span_overlap(x1, x2, y1, y2):
    return x1 < y2 and y1 < x2


def exact_span_match(annotator_1: List[List[Span]], annotator_2: List[List[Span]]):
    """
    Exact match: the offsets and label of spans have to be identical.

    Like in NER evaluation, we compare the spans of one annotator with the spans of another annotator. Two spans are a match iff their offsets and type match exactly. We take one annotator as the "gold" and the other annotator as "prediction" and calculate F1. Since the F1 score is symmetric, it does not matter which annotator we take as gold/predicted.
    """
    # Flatten annotations, each span has a document_id so we can tell apart identical spans in different docs.
    annotator_1 = set(ann for doc in annotator_1 for ann in doc)
    annotator_2 = set(ann for doc in annotator_2 for ann in doc)
    labels = set(annotation.label for annotation in annotator_1 | annotator_2)

    # Group by label such that we can calculate a per-label score
    ann1_dict = defaultdict(set)
    ann2_dict = defaultdict(set)
    for ann in annotator_1:
        ann1_dict[ann.label].add(ann)
    for ann in annotator_2:
        ann2_dict[ann.label].add(ann)

    label_scores = {}
    for label in labels:
        a = ann1_dict[label]
        b = ann2_dict[label]

        tp = len(a & b)
        fp = len(b - a)
        fn = len(a - b)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        label_scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_average = {
        "precision": np.mean([v["precision"] for v in label_scores.values()]),
        "recall": np.mean([v["recall"] for v in label_scores.values()]),
        "f1": np.mean([v["f1"] for v in label_scores.values()]),
    }

    return {"labels": label_scores, "macro_average": macro_average}


def partial_span_match(
    annotator_1: List[List[Span]], annotator_2: List[List[Span]], threshold=0.5
):
    """
    Fuzzy span-level matching. Two spans are a match if the token-level IOU is greater than a threshold (here, 0.5). When threshold = 1, this is identical to exact span-level matching.

    References:
    - DeYoung et al. (2020). https://doi.org/10.18653/v1/2020.acl-main.408
    - Briakou and Carpuat (2020). https://doi.org/10.18653/v1/2020.emnlp-main.121
    """
    ann1_grouped = defaultdict(set)
    ann2_grouped = defaultdict(set)

    for doc in annotator_1:
        for ann in doc:
            ann1_grouped[(ann.doc_id, ann.label)].add(ann)
    for doc in annotator_2:
        for ann in doc:
            ann2_grouped[(ann.doc_id, ann.label)].add(ann)

    doc_ids = set(doc_id for doc_id, label in ann1_grouped.keys() | ann2_grouped.keys())
    labels = set(label for doc_id, label in ann1_grouped.keys() | ann2_grouped.keys())

    label_scores = {}
    for label in labels:
        tp, fp, fn = 0, 0, 0
        for doc in doc_ids:
            anns_gold = ann1_grouped[(doc, label)]
            anns_pred = ann2_grouped[(doc, label)]

            matched_gold = set()
            matched_pred = set()

            for ann_pred in anns_pred:
                pred = set(range(ann_pred.start, ann_pred.end))
                for ann_gold in anns_gold:
                    gold = set(range(ann_gold.start, ann_gold.end))
                    iou = len(pred & gold) / len(pred | gold)
                    if iou >= threshold:
                        matched_gold.add(ann_gold)
                        matched_pred.add(ann_pred)

            tp += len(matched_gold)
            fp += len(anns_pred - matched_pred)
            fn += len(anns_gold - matched_gold)

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        label_scores[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    macro_average = {
        "precision": np.mean([v["precision"] for v in label_scores.values()]),
        "recall": np.mean([v["recall"] for v in label_scores.values()]),
        "f1": np.mean([v["f1"] for v in label_scores.values()]),
    }

    return {"labels": label_scores, "macro_average": macro_average}


def char_span_to_token_span(text, start_char, end_char, return_tokens=False):
    """
    Converts a given span with character-based indexing to token-indexing. Spans are always right-exclusive.
    For tokenization as simple whitespace tokenizer is used.
    """
    tokenizer = WhitespaceTokenizer()
    token_spans = list(tokenizer.span_tokenize(text))

    start_token_index = None
    end_token_index = None
    for i, (token_start, token_end) in enumerate(token_spans):
        has_overlap = span_overlap(start_char, end_char, token_start, token_end)

        if start_token_index is None and has_overlap:
            start_token_index = i
            end_token_index = i + 1
        elif has_overlap:
            end_token_index = i + 1

    return (start_token_index, end_token_index)


def char_spans_to_token_spans(text, annotations: List[Span]):
    result = []
    for ann in annotations:
        # would be more efficient if we tokenized here
        (start, end) = char_span_to_token_span(text, ann.start, ann.end)
        result.append(ann._replace(start=start, end=end))
    return result


def sentence_level(
    texts: List[str], annotator_1: List[List[Span]], annotator_2: List[List[Span]]
):
    """
    Project spans to sentence-level labels. If a sentence has a span of a given label, it receives a positive binary label, otherwise negative.

    References:
    - Goyal et al. (2022). https://doi.org/10.18653/v1/2022.emnlp-main.29
    """

    def project_labels(text: str, annotations: List[Span]):
        sents = list(PunktSentenceTokenizer().span_tokenize(text))

        sents_labels = []
        for sent_start, sent_end in sents:
            labels = []
            for ann in annotations:
                if span_overlap(sent_start, sent_end, ann.start, ann.end):
                    labels.append(ann.label)
            sents_labels.append(labels)

        return sents_labels

    def flatten(lists):
        return [e for l in lists for e in l]

    ann1_labeled_sentences = flatten(
        [project_labels(text, anns) for text, anns in zip(texts, annotator_1)]
    )
    ann2_labeled_sentences = flatten(
        [project_labels(text, anns) for text, anns in zip(texts, annotator_2)]
    )

    # NLTK format: (coder, item, labels: frozenset(str))
    binary_data = []
    labels_data = []

    for i, (ann1_labels, ann2_labels) in enumerate(
        zip(ann1_labeled_sentences, ann2_labeled_sentences)
    ):
        ann1_labels = frozenset(ann1_labels)
        ann2_labels = frozenset(ann2_labels)

        # Binary agreement: if sentence has an annotation, we assign 1, otherwise 0
        binary_data.append(("coder1", f"item{i}", int(len(ann1_labels) > 0)))
        binary_data.append(("coder2", f"item{i}", int(len(ann2_labels) > 0)))

        # Set-agreement: add labels, if one annotator did not assign any, we add a 'null' label
        labels_data.append(
            (
                "coder1",
                f"item{i}",
                ann1_labels if len(ann1_labels) > 0 else frozenset(["null"]),
            )
        )
        labels_data.append(
            (
                "coder2",
                f"item{i}",
                ann2_labels if len(ann2_labels) > 0 else frozenset(["null"]),
            )
        )

    binary = AnnotationTask()
    binary.load_array(binary_data)

    multilabel = AnnotationTask(distance=masi_distance)
    multilabel.load_array(labels_data)
    return {"binary": binary.alpha(), "multilabel": multilabel.alpha()}
