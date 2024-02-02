"""
Identify missing information through fact-based NLI.

1. Extract "atomic" facts from original. (GPT-4)
2. Classify each fact into entailed/neutral/contradicted by the simple text (off-the-shelf NLI model)
3. For each neutral fact, generate q/a. (GPT-4)
"""

import json
import os
from pathlib import Path
from typing import List

import openai
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from nltk.tokenize import PunktSentenceTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from info_loss.prompts import fact_extraction, question_generation
from info_loss.utils import openai_request, report_usage

CACHE_DIR = Path("output/gpt-4-0613-nli/predictions/")
OUTPUT_DIR = Path("output/gpt-4-0613-nli/")
NLI_MODEL_NAME_OR_PATH = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def extract_facts(document_id, text: str):
    """Splits text into sentences, extracts atomic facts per sentence."""
    params = fact_extraction.generation_params()
    sentences = []
    split_sentences = PunktSentenceTokenizer().span_tokenize(text)
    for i, (start, end) in enumerate(split_sentences):
        sent = text[start:end]
        messages = fact_extraction.get_messages(sent)
        response = openai_request(
            params,
            messages,
            cache_id=f"{document_id}-{i}",
            cache_dir=CACHE_DIR,
        )
        try:
            response = fact_extraction.parse_response(response)
        except:
            print(f"Failed to parse response for {document_id}")
            print(f"Response:\n{response}")
            raise

        sentences.append(
            {
                "sentence": sent,
                "start": start,
                "end": end,
                "facts": [{"fact": fact} for fact in response],
            }
        )
    return sentences


def classify_facts(premises: List[str], hypotheses: List[str], tokenizer, model):
    """
    Classify facts as per NLI: entail/neutral/contradict
    The hypothesis is the fact, the premise is the simple text.
    Neutral facts are considered omissions and we disregard contradictions.
    """
    dataset = Dataset.from_dict({"premise": premises, "hypothesis": hypotheses})
    dataset = dataset.map(
        lambda examples: tokenizer(examples["premise"], examples["hypothesis"]),
        batched=True,
    )
    dataset = dataset.remove_columns(["premise", "hypothesis"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=32)

    model.eval()
    labels, probas = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classify facts (batches)"):
            batch = batch.to(model.device)
            logits = model(**batch).logits
            preds = torch.softmax(logits, dim=-1)
            labels.extend(preds.argmax(dim=-1).cpu().tolist())
            probas.extend(preds.cpu().tolist())
    return labels, probas


def generate_qa(doc):
    # Collect missing facts (label == 1 == neutral)
    flat_facts = [
        fact
        for sent in doc["original_sentences"]
        for fact in sent["facts"]
        if fact.get("label") == 1
    ]

    if not flat_facts:
        return doc

    messages = question_generation.get_messages(
        doc["original"],
        doc["simplification"],
        facts=[fact["fact"] for fact in flat_facts],
    )
    params = question_generation.generation_params()
    response = openai_request(
        params,
        messages,
        cache_id=f"{doc['id']}-qa",
        cache_dir=CACHE_DIR,
    )
    try:
        qas = question_generation.parse_response(response)
    except:
        print(f"Failed to parse response for {doc['id']}")
        print(f"Response:\n{response}")
        raise
    for fact, qa in zip(flat_facts, qas):
        fact["question"] = qa["question"]
        fact["answer"] = qa["answer"]
    return doc


def qa2thresh(doc):
    result = {
        "id": doc["id"],
        "source": doc["original"],
        "target": doc["simplification"],
        "edits": [],
    }

    i = 0
    for sent in doc["original_sentences"]:
        for fact in sent["facts"]:
            if "question" in fact:
                edit = {
                    "id": i,
                    "category": "omission",  # this method does not support a concept category
                    "annotation": {
                        "question": fact["question"],
                        "answer": fact["answer"],
                        "nli_fact": fact["fact"],
                        "nli_label": fact["label"],
                        "nli_proba": fact["proba"],
                    },
                    "input_idx": [[sent["start"], sent["end"]]],
                }
                i += 1
                result["edits"].append(edit)
    return result


def main():
    # The main datastructure looks as follows:
    # doc = {
    #     "id": "",
    #     "original": "",
    #     "simplification": "",
    #     "original_sentences": [
    #         {
    #             "sentence": "",
    #             "start": 0,
    #             "end": 0,
    #             "facts": [
    #                 {
    #                     "fact": "",
    #                     "label": "",
    #                     "proba": "",
    #                     "question": "",
    #                     "answer": "",
    #                 }
    #             ],
    #         }
    #     ],
    # }
    df = pd.read_json("data/processed/documents.json")
    samples = df[["PMCID", "abstract", "simplification"]].values

    # 1. Extract facts
    docs = []
    for doc_id, original, simplification in tqdm(samples, desc="Extract facts"):
        docs.append(
            {
                "id": doc_id,
                "original": original,
                "simplification": simplification,
                "original_sentences": extract_facts(doc_id, original),
            }
        )

    # 2. Classify facts as per NLI: entail/neutral/contradict
    # We first accumulates all facts across all documents to benefit from batching.
    flat_facts = []  # Pointers to the fact dictionaries
    premises, hypotheses = [], []
    for doc in docs:
        for sent in doc["original_sentences"]:
            for fact in sent["facts"]:
                flat_facts.append(fact)
                premises.append(doc["simplification"])
                hypotheses.append(fact["fact"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME_OR_PATH)
    model.to(device)
    labels, probas = classify_facts(premises, hypotheses, tokenizer, model)
    for fact, label, proba in zip(flat_facts, labels, probas):
        fact["label"] = label
        fact["proba"] = proba

    del model  # free GPU memory, we do not need it anymore.

    # Write classification result to disk for manual analysis for facts.
    with open(OUTPUT_DIR / "nli_results.json", "w") as fout:
        json.dump(docs, fout)

    # 3. For each missing/neutral fact, generate a Q/A pair.
    docs = [generate_qa(doc) for doc in tqdm(docs, desc="Generate Q/As")]

    # Convert intermediate format into standard form and write out to disk.
    results = [qa2thresh(doc) for doc in docs]
    with open(OUTPUT_DIR / "predictions.json", "w") as fout:
        json.dump(results, fout)

    report_usage(CACHE_DIR)


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]
    main()
