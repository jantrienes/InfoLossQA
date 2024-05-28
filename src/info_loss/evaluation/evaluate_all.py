import time
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
import openai
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count

from info_loss.utils import openai_request
from info_loss.evaluation import (
    accuracy_answer,
    accuracy_snippet,
    givenness_location,
    givenness_phrasing,
    hallucinations_answer,
    relevance_source,
    relevance_target,
    simplicity_jargon,
    simplicity_standalone,
)

PROMPTS = {
    "accuracy_answer": accuracy_answer,
    "accuracy_snippet": accuracy_snippet,
    "givenness_location": givenness_location,
    "givenness_phrasing": givenness_phrasing,
    "hallucinations_answer": hallucinations_answer,
    "relevance_source": relevance_source,
    "relevance_target": relevance_target,
    "simplicity_jargon": simplicity_jargon,
    "simplicity_standalone": simplicity_standalone,
}

MAX_WORKERS = 8


def process_prompt(args):
    qa_pair, criterion, output_path = args
    prompt = PROMPTS[criterion]
    cache_dir = Path(output_path) / criterion

    params = prompt.generation_params()
    messages = prompt.get_messages(qa_pair)
    response = openai_request(params, messages, cache_dir=cache_dir, cooldown=0.1)
    response = prompt.parse_response(response)
    response["edit_id"] = qa_pair["edit_id"]
    return response, criterion


def main(args):
    with open(args.input_json) as fin:
        data = json.load(fin)
    order = {qa_pair["edit_id"]: i for i, qa_pair in enumerate(data)}

    arg_list = []
    for criterion in PROMPTS.keys():
        for qa_pair in data:
            arg_list.append((qa_pair, criterion, args.output_path))

    results = {criterion: [] for criterion in PROMPTS.keys()}
    with Pool(processes=MAX_WORKERS) as pool:
        with tqdm(total=len(arg_list), desc="Processing QA pairs") as progress:
            for result, criterion in pool.imap_unordered(process_prompt, arg_list):
                results[criterion].append(result)
                progress.update(1)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for criterion, ratings in results.items():
        ratings = sorted(ratings, key=lambda qa_pair: order[qa_pair["edit_id"]])
        with open(output_path / f"{criterion}.json", "w") as fout:
            json.dump(ratings, fout)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_json", help="Path to JSON with QA-pairs.")
    parser.add_argument(
        "--output_path",
        default="output/infolossqa-eval/gpt-4-0125-preview-zero-shot/",
        help="Root of output. Will store cache and final predictions there.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main(arg_parser())
