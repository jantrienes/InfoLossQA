import hashlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import openai
import together
import litellm


def text2hash(string: str) -> str:
    hash_object = hashlib.sha256(string.encode("utf-8"))
    hex_dig = hash_object.hexdigest()
    return hex_dig


def get_cache_path(generation_params, messages, cache_dir, cache_id=None) -> Path:
    if cache_id:
        filename = f"{cache_id}.json"
    else:
        filename = f"{text2hash(str(generation_params) + str(messages))}.json"
    return cache_dir / filename


def openai_request(
    generation_params,
    messages,
    cache_id=None,
    cache_dir=Path("../output/openai_cache/"),
    overwrite_cache=False,
    retries=5,
    cooldown=0,
):
    cache_dir = Path(cache_dir)
    cache_file = get_cache_path(generation_params, messages, cache_dir, cache_id)

    if not overwrite_cache and cache_dir is not None and cache_file.exists():
        with open(cache_file) as fin:
            return json.load(fin)

    for retry in range(retries):
        try:
            response = openai.ChatCompletion.create(
                **generation_params,
                messages=messages,
            )
            response = response.to_dict_recursive()
            response = {**response, **generation_params}
            response["messages"] = messages
            if cooldown > 0:
                time.sleep(cooldown)
            break
        except Exception as exc:
            if retry < retries - 1:
                sleep_seconds = 5
                logging.warning(
                    f"Exception in OpenAI API call. Retry in {sleep_seconds} secs...",
                    exc_info=True,
                )
                time.sleep(sleep_seconds)
            else:
                raise Exception(
                    f"OpenAI API failed for {retries} times. Please try again later."
                ) from exc

    cache_dir.mkdir(exist_ok=True, parents=True)
    with open(cache_file, "w") as fout:
        json.dump(response, fout)
    return response


def together_request(
    generation_params,
    messages,
    tokenizer,
    cache_id=None,
    cache_dir=Path("../output/together_cache/"),
    overwrite_cache=False,
    retries=5,
):
    cache_dir = Path(cache_dir)
    cache_file = get_cache_path(generation_params, messages, cache_dir, cache_id)
    if not overwrite_cache and cache_dir is not None and cache_file.exists():
        with open(cache_file) as fin:
            return json.load(fin)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if not generation_params.get("max_tokens"):
        # NOTE: max length is hardcoded for LLama, different models may need different value.
        generation_params["max_tokens"] = 4096 - prompt_len - 2

    for retry in range(retries):
        try:
            response = together.Complete.create(
                prompt=prompt,
                **generation_params,
            )
            response["choices"] = response["output"]["choices"]
            del response["output"]
            message = response["choices"][0]
            output_len = len(tokenizer(message["text"])["input_ids"])
            message["finish_reason"] = (
                "stop" if output_len < generation_params["max_tokens"] else "length"
            )
            response["usage"] = {
                "prompt_tokens": prompt_len,
                "completion_tokens": output_len,
                "total_tokens": prompt_len + output_len,
            }
            break
        except Exception as exc:
            if retry < retries - 1:
                sleep_seconds = 5
                logging.warning(
                    f"Exception in Together API call. Retry in {sleep_seconds} secs...",
                    exc_info=True,
                )
                time.sleep(sleep_seconds)
            else:
                raise Exception(
                    f"Together API failed for {retries} times. Please try again later."
                ) from exc

    cache_dir.mkdir(exist_ok=True, parents=True)
    with open(cache_file, "w") as fout:
        json.dump(response, fout)
    return response


def litellm_request(
    generation_params,
    messages,
    cache_id=None,
    cache_dir=Path("../output/litellm_cache/"),
    overwrite_cache=False,
    retries=5,
    cooldown=0,
):
    cache_dir = Path(cache_dir)
    cache_file = get_cache_path(generation_params, messages, cache_dir, cache_id)

    if not overwrite_cache and cache_dir is not None and cache_file.exists():
        with open(cache_file) as fin:
            return json.load(fin)

    for retry in range(retries):
        try:
            response = litellm.completion(
                **generation_params,
                messages=messages,
            )
            response = response.to_dict()
            response = {**response, **generation_params}
            response["messages"] = messages
            if cooldown > 0:
                time.sleep(cooldown)
            break
        except Exception as exc:
            if retry < retries - 1:
                sleep_seconds = 5
                logging.warning(
                    f"Exception in API call. Retry in {sleep_seconds} secs...",
                    exc_info=True,
                )
                time.sleep(sleep_seconds)
            else:
                raise Exception(
                    f"API failed for {retries} times. Please try again later."
                ) from exc

    cache_dir.mkdir(exist_ok=True, parents=True)
    with open(cache_file, "w") as fout:
        json.dump(response, fout)
    return response


def huggingface_generate(
    generation_params,
    messages,
    tokenizer,
    pipeline,
    cache_id=None,
    cache_dir=Path("../output/huggingface_cache/"),
    overwrite_cache=False,
):
    cache_dir = Path(cache_dir)
    cache_file = get_cache_path(generation_params, messages, cache_dir, cache_id)
    if not overwrite_cache and cache_dir is not None and cache_file.exists():
        with open(cache_file) as fin:
            return json.load(fin)

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    generation = pipeline(
        prompt,
        max_new_tokens=generation_params["max_tokens"],
        generation_kwargs=generation_params,
        pad_token_id=tokenizer.eos_token_id,
    )
    generation = generation[0]["generated_text"]
    output_len = len(tokenizer(generation)["input_ids"])
    finish_reason = "stop" if output_len < generation_params["max_tokens"] else "length"

    # mimic outputs of OpenAI/Together API
    result = {
        **generation_params,
        "choices": [
            {
                "role": "assistant",
                "text": generation,
                "finish_reason": finish_reason,
            }
        ],
        "messages": messages,
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": output_len,
            "total_tokens": prompt_len + output_len,
        },
    }

    cache_dir.mkdir(exist_ok=True, parents=True)
    with open(cache_file, "w") as fout:
        json.dump(result, fout)
    return result


def report_usage(predictions_path):
    COSTS = {
        # per 1000 tokens
        "gpt-4-0613-prompt": 0.03,
        "gpt-4-0613-completion": 0.06,
        "togethercomputer/llama-2-70b-chat-prompt": 0.0009,
        "togethercomputer/llama-2-70b-chat-completion": 0.0009,
        "mistralai/Mistral-7B-Instruct-v0.1-prompt": 0,  # running locally, just stubbing for report_usage
        "mistralai/Mistral-7B-Instruct-v0.1-completion": 0,  # running locally, just stubbing for report_usage
    }

    predictions = list(Path(predictions_path).glob("*.json"))
    n_requests = len(predictions)
    token_limit = []
    token_tally = defaultdict(int)

    for f in predictions:
        with open(f) as fin:
            data = json.load(fin)
            if data["choices"][0]["finish_reason"] != "stop":
                token_limit.append(f)

            model = data["model"]
            token_tally[f"{model}-prompt"] += data["usage"]["prompt_tokens"]
            token_tally[f"{model}-completion"] += data["usage"]["completion_tokens"]

    if len(token_limit) > 0:
        print(
            "WARNING: Generation terminated early due to token limit for following files:"
        )
        print(token_limit)

    print("Costs:")
    total = 0
    for model, n_tokens in token_tally.items():
        costs = n_tokens / 1000 * COSTS[model]
        total += costs
        print(f"- {model} ({n_tokens:,}/1000*{COSTS[model]}) = {costs:.1f}$")
    print(f"Total: {total:.1f}$")
    print(f"Requests: {n_requests}")
