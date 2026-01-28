#  Copyright (c) 2026 Constantinos Eleftheriou <Constantinos.Eleftheriou@ed.ac.uk>.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies
#  or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import os
import json
import click
import datetime
import pandas as pd


@click.command("session")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True),
    default=".",
    help="Output directory for report and trials files.",
)
def session_cmd(**kwargs):
    """Generate a session report and extract trials from a Visiomode JSON file."""
    out_dir = summarise(**kwargs)
    click.echo(f"Files saved under {out_dir}")


def summarise(path: str, output_dir: str = ".") -> str:
    """Generate a session summary report and trials file from a raw Visiomode JSON.

    Args:
        path (str): Path to Visiomode JSON file.
        output_dir (str, optional): Output directory for report and trials files. Defaults to ".".

    Returns:
        str: Returns directory under which files were saved
    """
    extract_trials(path, to_csv=True, output_dir=output_dir)
    generate_report(path, output_dir=output_dir)

    return output_dir


def extract_trials(path: str, to_csv: bool = False, output_dir: str = ".") -> pd.DataFrame:
    """Parse a Visiomode JSON file and return a trials dataframe. Optionally save to CSV.

    Args:
        path (str): Path to Visiomode JSON file.
        to_csv (bool, optional): Save trials dataframe as a CSV. If True, indicate which directory to save in via `output_dir`. Defaults to False.
        output_dir (str, optional): Output directory for trials CSV file. Only used if `to_csv` is True. Defaults to current directory.

    Returns:
        pd.DataFrame: Session trials dataframe, where each row is a trial.
    """
    animal_id = path.split("/")[-1].split("_")[0].strip("sub-")
    experiment = path.split("/")[-1].split("_")[1].replace("exp-", "")

    session_date = datetime.datetime.strptime(path.split(os.sep)[-1].split("_")[2].strip("ses-")[:8], "%Y%m%d")
    interaction = path.split(os.sep)[-1].split("_")[-1].replace("behaviour-", "").replace(".json", "")
    trials = []
    protocol = None
    with open(f, "r") as fp:
        session_data = json.load(fp)
        protocol = session_data["protocol"]
        trials = session_data["trials"]
    session = [
        {
            "animal_id": animal_id,
            "session_date": session_date,
            "protocol": protocol,
            "interaction": interaction,
            "experiment": experiment,
            **trial,
        }
        for trial in trials
    ]

    df = pd.DataFrame(session)

    if to_csv:
        out_path = (
            f"{output_dir}{os.sep}{animal_id}_{experiment}_{str(session_date)}_behaviour-{interaction}_trials.csv"
        )
        df.to_csv(out_path)

    return df


def generate_report(path: str, output_dir: str = ".") -> str:
    return ""
