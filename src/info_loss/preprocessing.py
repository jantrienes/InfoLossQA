import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

from openai import OpenAI

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

SYSTEM_PROMPT = "Please simplify the following technical abstract into plain language that an average adult would understand. If the abstract has sections, keep them."
GPT_MODEL = "gpt-4-0613"


def load_text(path):
    with open(path) as fin:
        return fin.read()


def write_text(txt, path):
    with open(path, "w") as fout:
        fout.write(txt)


def segment_abstract(text):
    text = text.strip()
    sections = text.split("\n\n")

    title = sections[0]
    title = re.sub("TITLE[:.]", "", title).strip()
    sections = sections[1:]
    sections = [s.strip() for s in sections if s.strip()]
    text_formatted = ""
    sectioned = False

    if len(sections) == 1:
        text_formatted = re.sub(r"^ABSTRACT\.\s?\n", "", sections[0], re.MULTILINE)
        text_formatted = re.sub(
            r"^ABSTRACT\.SUMMARY\.\s?\n", "", text_formatted, re.MULTILINE
        )
    elif len(sections) >= 1:
        sectioned = True
        formatted = []
        for section in sections:
            if "\n" not in section:
                print(f"WARNING: {section}")
                continue

            match = re.match(
                r"^ABSTRACT(?:\.SUMMARY)?\.(.+)\s?\n", section, re.MULTILINE
            )
            sec_title = match.group(1).rstrip(".:") + "." + "\n"
            section = re.sub(match.group(0), sec_title, section)
            formatted.append(section)
        text_formatted = "\n\n".join(formatted)
    else:
        raise ValueError

    return title, text_formatted, sectioned


def openai_request(prompt):
    response = client.chat.completions.create(model=GPT_MODEL,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0)
    # get the first message in response (should only be one)
    result = response.choices[0].message.content
    return result


def main(args):
    if Path(args.output_json).exists():
        print(f"SKIP: {args.output_json} does already exist.")
        exit(0)
    Path(args.cache_path).mkdir(exist_ok=True, parents=True)
    Path(args.output_json).parent.mkdir(exist_ok=True, parents=True)

    df_raw = pd.read_csv(args.input_csv, index_col=0)
    df_processed = df_raw.apply(
        lambda row: segment_abstract(row["Abstract"]), axis=1, result_type="expand"
    )
    df_processed.columns = ["title", "abstract", "sectioned"]
    df_processed["PMCID"] = df_raw["PMCID"]

    outputs = []
    for _, row in tqdm(df_processed.iterrows(), total=len(df_processed)):
        result_file = Path(args.cache_path) / f"PMCID{row['PMCID']}.gpt4.txt"
        try:
            txt = load_text(result_file)
        except FileNotFoundError:
            txt = openai_request(row["abstract"])
            write_text(txt, result_file)
        outputs.append(txt)

    # Write full JSON
    df_processed["simplification"] = outputs
    df_processed.to_json(args.output_json, orient="records")

    def normalize(x):
        # avoid bugs in thresh
        x = unicodedata.normalize("NFKC", x)
        x = x.replace("<", "≺")
        x = x.replace(">", "≻")
        return x

    df_processed["abstract"] = df_processed["abstract"].apply(normalize)
    df_processed["simplification"] = df_processed["simplification"].apply(normalize)

    # Write thresh tasks
    task_path = Path(args.thresh_tasks_path)
    task_path.mkdir(exist_ok=True, parents=True)
    for index, row in df_processed.iterrows():
        d = {
            "id": row["PMCID"],
            "source": row["abstract"],
            "target": row["simplification"],
        }
        with open(task_path / f"PMCID{row['PMCID']}.json", "w") as fout:
            json.dump([d], fout)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        default="data/raw/documents.csv",
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output_json",
        default="data/processed/documents.json",
        help="Path to write results to.",
    )
    parser.add_argument(
        "--cache_path",
        default="output/gpt-4-simplification/",
        help="Cache intermediate GPT-4 responses.",
    )
    parser.add_argument(
        "--thresh_tasks_path",
        default="data/processed/thresh-tasks/",
        help="Path to write thresh tasks to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    main(arg_parser())
