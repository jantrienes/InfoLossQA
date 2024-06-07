import argparse
import json
import logging
import os
import unicodedata
from collections import namedtuple
from pathlib import Path

import Levenshtein
import openai
import pandas as pd
import together
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import AutoTokenizer, pipeline

from info_loss import utils
from info_loss.prompts import e2e_gpt4, e2e_llama, e2e_mistral

Match = namedtuple("Match", ["start", "end", "similarity"])


class GPT4:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def predict(self, sample_id, original, simplification):
        try:
            params = e2e_gpt4.generation_params()
            messages = e2e_gpt4.get_messages(original, simplification)
            response = utils.openai_request(
                params, messages, cache_dir=self.cache_dir, cache_id=f"{sample_id}"
            )
            qa_pairs = e2e_gpt4.parse_response(response)
            result = qa2thresh(sample_id, original, simplification, qa_pairs)
        except Exception as e:
            logging.exception(f"WARNING: Inference error (sample = {sample_id})")
            return {
                "id": sample_id,
                "source": original,
                "target": simplification,
                "edits": [],
                "error": str(e),
            }
        return result


class LLama:
    def __init__(self, cache_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-70b-chat-hf", token=os.environ["HF_TOKEN"]
        )
        self.cache_dir = cache_dir
        together.api_key = os.environ["TOGETHER_API_KEY"]

    def predict(self, sample_id, original, simplification):
        try:
            params = e2e_llama.generation_params()
            messages = e2e_llama.get_messages(original, simplification)
            response = utils.together_request(
                params,
                messages,
                self.tokenizer,
                cache_dir=self.cache_dir,
                cache_id=f"{sample_id}",
            )
            response = e2e_llama.parse_response(response)
            result = qa2thresh(sample_id, original, simplification, response)
        except Exception as e:
            logging.exception(f"WARNING: Inference error (sample = {sample_id})")
            return {
                "id": sample_id,
                "source": original,
                "target": simplification,
                "edits": [],
                "error": str(e),
            }
        return result


class Mistral:
    def __init__(self, cache_dir):
        self.params = e2e_mistral.generation_params()
        self.tokenizer = AutoTokenizer.from_pretrained(self.params["model"])
        self.pipeline = pipeline(
            "text-generation",
            model=self.params["model"],
            return_full_text=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.pipeline = None
        self.cache_dir = cache_dir

    def predict(self, sample_id, original, simplification):
        try:
            messages = e2e_mistral.get_messages(original, simplification)
            response = utils.huggingface_generate(
                self.params,
                messages,
                self.tokenizer,
                self.pipeline,
                cache_dir=self.cache_dir,
                cache_id=f"{sample_id}",
            )
            response = e2e_mistral.parse_response(response)
            result = qa2thresh(sample_id, original, simplification, response)
        except Exception as e:
            logging.exception(f"WARNING: Inference error (sample = {sample_id})")
            return {
                "id": sample_id,
                "source": original,
                "target": simplification,
                "edits": [],
                "error": str(e),
            }
        return result


def match_span(text, query):
    if not query:
        return Match(-1, -1, -1)
    text = unicodedata.normalize("NFKD", text).lower()
    query = unicodedata.normalize("NFKD", query).lower()

    try:
        start = text.index(query)
        end = start + len(query)
        return Match(start, end, 1)
    except ValueError:
        start, end, similarity = match_fuzzy(text, query)
        return Match(start, end, similarity)


def match_fuzzy(text, query):
    text_words = text.split(" ")
    query_words = query.split(" ")

    best_similarity = 0
    best_start = 0
    best_end = 0

    # Iterate over windows (len(query) +- 3 words)
    min_window = max(len(query_words) - 3, 1)
    for window_size in range(min_window, len(query_words) + 3):
        for i in range(len(text_words) - window_size + 1):
            window = text_words[i : i + window_size]
            window_str = " ".join(window)
            similarity = Levenshtein.ratio(window_str, query)
            if similarity > best_similarity:
                best_similarity = similarity
                best_start = i
                best_end = i + window_size

    # Lookup character-based index
    best_str = " ".join(text_words[best_start:best_end])
    start = text.index(best_str)
    end = start + len(best_str)
    return start, end, best_similarity


def qa2thresh(sample_id, original, simplification, qa_pairs):
    result = {
        "id": sample_id,
        "source": original,
        "target": simplification,
        "edits": [],
    }

    for i, qa in enumerate(qa_pairs):
        edit = {
            "id": i,
            "category": qa["category"],
            "annotation": {
                "question": qa["question"],
                "answer": qa["answer"],
                "rationale": qa["rationale"],
            },
            "prediction_errors": [],
        }

        # Get input/output offsets.
        # Assign following error codes:
        # - Missing input (omission, concept)
        # - Lookup error input (omission, concept)
        # - Spurious output (omission)
        # - Missing output (concept)
        # - Lookup error output (concept)

        if qa.get("original"):
            edit["annotation"]["input_raw"] = qa["original"]

            (start, end, similarity) = match_span(original, qa["original"])
            if similarity >= 0.8:
                edit["input_idx"] = [[start, end]]
                edit["annotation"]["input_matched"] = original[start:end]
                edit["annotation"]["input_matched_similarity"] = similarity
            else:
                edit["prediction_errors"].append("invalid_input_localization")
        else:
            edit["prediction_errors"].append("missing_input_localization")

        if qa.get("simplification"):
            edit["annotation"]["output_raw"] = qa["simplification"]

            if qa["category"] == "omission":
                edit["prediction_errors"].append("spurious_output_localization")
            else:
                (start, end, similarity) = match_span(
                    simplification, qa["simplification"]
                )
                if similarity >= 0.8:
                    edit["annotation"]["output_matched"] = simplification[start:end]
                    edit["annotation"]["output_matched_similarity"] = similarity
                    edit["output_idx"] = [[start, end]]
                else:
                    edit["prediction_errors"].append("invalid_output_localization")
        elif not qa.get("simplification") and qa["category"] == "concept":
            edit["prediction_errors"].append("missing_output_localization")

        result["edits"].append(edit)
    return result


def main(args):
    output_path = Path(args.output_path)
    cache_path = output_path / "predictions"
    if args.model == "gpt4":
        model = GPT4(cache_path)
    elif args.model == "llama":
        model = LLama(cache_path)
    elif args.model == "mistral":
        model = Mistral(cache_path)
    else:
        raise ValueError("Unsupported model.")

    df = pd.read_json(args.input_json)
    samples = df[["PMCID", "abstract", "simplification"]].values
    predictions = [model.predict(*sample) for sample in tqdm(samples)]
    with open(output_path / "predictions.json", "w") as fout:
        json.dump(predictions, fout)
    utils.report_usage(cache_path)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="data/processed/documents.json",
        help="Path to input JSON.",
    )
    parser.add_argument(
        "--output_path", help="Path to write results to.", required=True
    )
    parser.add_argument(
        "--model",
        help="Model to use.",
        choices=["gpt4", "llama", "mistral"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    main(arg_parser())
