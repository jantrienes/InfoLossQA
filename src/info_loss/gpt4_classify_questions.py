"""
Classify questions according to question taxonomy.

Reference:
- Shuyang Cao and Lu Wang (2021). https://aclanthology.org/2021.acl-long.502/
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import dotenv
import openai
from tqdm.auto import tqdm

from info_loss import utils
from info_loss.prompts import question_classifier
from info_loss.statistics import load_data


def predict_batched(questions: List[str], batch_size, cache_dir):
    predictions = []
    for i in tqdm(
        range(0, len(questions), batch_size),
        total=math.ceil(len(questions) / batch_size),
    ):
        batch = questions[i : i + batch_size]
        params = question_classifier.generation_params()
        messages = question_classifier.get_messages(batch)
        response = utils.openai_request(params, messages, cache_dir=cache_dir)
        response = question_classifier.parse_response(response)
        predictions.extend(response)
    return predictions


def main(args):
    docs = load_data(args.input_path)
    cache_dir = Path(args.output_json).parent / "questions-predictions/"
    samples = []
    for doc in docs:
        for edit in doc["edits"]:
            if edit["category"] != "comment":
                samples.append(
                    {
                        "doc_id": doc["id"],
                        "edit_id": edit["id"],
                        "question": edit["annotation"]["question"],
                        "category": edit["category"],
                    }
                )

    questions = [sample["question"] for sample in samples]
    predictions = predict_batched(questions, batch_size=32, cache_dir=cache_dir)
    for y, question in zip(predictions, samples):
        question["label"] = y
    with open(args.output_json, "w") as fout:
        json.dump(samples, fout)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        help="Either a directory with *.json files or a single .json file.",
    )
    parser.add_argument(
        "--output_json",
        help="File to write classified questions to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    dotenv.load_dotenv()
    main(arg_parser())
