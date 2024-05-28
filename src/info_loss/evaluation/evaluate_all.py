import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from json.decoder import JSONDecodeError
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from info_loss.evaluation import (
    accuracy_answer,
    accuracy_snippet,
    givenness_location,
    givenness_phrasing,
    hallucinations_answer,
    recall,
    relevance_source,
    relevance_target,
    simplicity_jargon,
    simplicity_standalone,
)
from info_loss.utils import litellm_request, openai_request

ACCURACY_PROMPTS = {
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

RECALL_PROMPTS = {
    "recall": recall,
}
PROMPTS = {**ACCURACY_PROMPTS, **RECALL_PROMPTS}

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
    sample, prompt_name, output_path, model = args
    prompt = PROMPTS[prompt_name]
    cache_dir = Path(output_path) / prompt_name

    messages = prompt.get_messages(sample)
    response = generate(messages, cache_dir=cache_dir, model=model)
    try:
        response = response["choices"][0]["message"]["content"]
        response = prompt.parse_response(response)
    except JSONDecodeError as e:
        log = f"WARN: could not parse response for edit_id={sample['edit_id']}, criterion={prompt_name}. Raw:\n{response}"
        logger.warning(log)
        response = {f"{prompt_name}": None, f"{prompt_name}_rationale": None}

    response["edit_id"] = sample["edit_id"]
    return response, prompt_name


def main(args):
    with open(args.input_json) as fin:
        data = json.load(fin)
        data = data[:10]

    if args.evaluate_recall:
        prompts = RECALL_PROMPTS.keys()
    else:
        prompts = ACCURACY_PROMPTS.keys()

    arg_list = []
    for prompt_name in prompts:
        for sample in data:
            arg_list.append((sample, prompt_name, args.output_path, args.model))

    results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        r = list(tqdm(pool.map(process_prompt, arg_list), total=len(arg_list)))
        for result, criterion in r:
            results[criterion].append(result)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for criterion, ratings in results.items():
        with open(output_path / f"{criterion}.json", "w") as fout:
            json.dump(ratings, fout)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_json", help="Path to JSON with samples.", required=True
    )
    parser.add_argument(
        "--output_path",
        help="Root of output. Will store cache and final predictions there.",
        required=True,
    )
    parser.add_argument("--model", choices=["gpt4", "llama3"])
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--evaluate_recall", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main(arg_parser())
