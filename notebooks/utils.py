import json
import pandas as pd
from IPython.display import display
from pathlib import Path


def display_sample(df, N=10, random_state=42, left_align=True):
    """Print a sample of a DataFrame without limits on column width.

    Strings are not truncated and line breaks are rendered.

    Parameters
    ----------
    N : int
        Number of rows to display. If None or larger than df, it displays all rows.
    """
    with pd.option_context("display.max_colwidth", None):
        if not N or N > len(df):
            sample = df
        else:
            sample = df.sample(N, random_state=random_state)

        styler = sample.style.set_properties(**{"white-space": "pre-wrap"})
        if left_align:
            styler = styler.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )
        display(styler)


def labelstudio_import(client, config, tasks, title):
    existing_projects = [p.get_params()["title"] for p in client.get_projects()]
    if title in existing_projects:
        print(f"WARNING: Project {title} already exists. Skipping...")
        return

    with open(config) as fin:
        config = fin.read()
    with open(tasks) as fin:
        tasks = json.load(fin)

    project = client.start_project(title=title, label_config=config)
    project.import_tasks(tasks)


def labelstudio_export(client, project_id, out_file):
    out_file = Path(out_file)
    if out_file.exists():
        with open(out_file) as fin:
            annotations = json.load(fin)
    else:
        annotations = client.get_project(project_id).export_tasks("JSON_MIN")
        with open(out_file, "w") as fout:
            json.dump(annotations, fout)

    return annotations
