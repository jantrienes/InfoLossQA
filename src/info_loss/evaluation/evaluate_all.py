import logging
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from json.decoder import JSONDecodeError

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
from info_loss.utils import openai_request, litellm_request

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

MAX_WORKERS = 1

logger = logging.getLogger(__name__)


def generate(messages, cache_dir, model):
    if model == "gpt4":
        params = {
            "model": "gpt-4o-2024-05-13",
            "temperature": 0,
            "max_tokens": 256,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = openai_request(params, messages, cache_dir=cache_dir, cooldown=0.1)

    elif model == "llama3":
        params = {
            "model": "together_ai/meta-llama/Llama-3-70b-chat-hf",
            "temperature": 0,
            "max_tokens": 256,
            "top_p": 1,
            "frequency_penalty": 0,
        }
        response = litellm_request(params, messages, cache_dir=cache_dir, cooldown=0.1)
    else:
        raise ValueError(f"Invalid model {model}.")

    return response


def process_prompt(args):
    qa_pair, criterion, output_path, model = args
    prompt = PROMPTS[criterion]
    cache_dir = Path(output_path) / criterion

    messages = prompt.get_messages(qa_pair)
    response = generate(messages, cache_dir=cache_dir, model=model)
    try:
        respnose = response["choices"][0]["message"]["content"]
        response = prompt.parse_response(response)
    except JSONDecodeError as e:
        log = f"WARN: could not parse response for edit_id={qa_pair['edit_id']}, criterion={criterion}. Raw:\n{respnose}"
        logger.warning(log)
        response = {f"{criterion}": None, f"{criterion}_rationale": None}

    response["edit_id"] = qa_pair["edit_id"]
    return response, criterion


def main(args):
    with open(args.input_json) as fin:
        data = json.load(fin)
    order = {qa_pair["edit_id"]: i for i, qa_pair in enumerate(data)}

    arg_list = []
    for criterion in PROMPTS.keys():
        for qa_pair in data:
            arg_list.append((qa_pair, criterion, args.output_path, args.model))

    results = {criterion: [] for criterion in PROMPTS.keys()}
    with Pool(processes=args.max_workers) as pool:
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
    parser.add_argument(
        "--input_json", help="Path to JSON with QA-pairs.", required=True
    )
    parser.add_argument(
        "--output_path",
        help="Root of output. Will store cache and final predictions there.",
        required=True,
    )
    parser.add_argument("--model", choices=["gpt4", "llama3"])
    parser.add_argument("--max_workers", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main(arg_parser())
