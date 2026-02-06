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
import numpy as np

from pathlib import Path
from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import select_autoescape

from collections.abc import Iterator


SESSION_REPORT_TEMPLATE = "session.html"

HIT = "hit"
MISS = "miss"
FALSE_ALARM = "false_alarm"
CORRECT_REJECTION = "correct_rejection"


env = Environment(loader=PackageLoader("visiomode_analysis.reports", "templates"), autoescape=select_autoescape())


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


def get_metadata(path: str) -> dict:
    with open(path, "r") as fp:
        session_data = json.load(fp)

        # Prefer the JSON for the following
        session_start_time = session_data.get("timestamp")
        protocol = session_data.get("protocol", "unknown")
        version = session_data.get("version", "unknown")
        duration = session_data.get("duration", "unknown")
        response_device = session_data.get("spec", {}).get("response_device", "unknown")
        reward_profile = session_data.get("spec", {}).get("reward_profile", "unknown")
        stimulus_duration = float(session_data.get("spec", {}).get("stimulus_duration", -1))
        iti = float(session_data.get("spec", {}).get("iti", -1))
        corrections_enabled = session_data.get("spec", {}).get("corrections_enabled", "unknown")
        stimuli = {
            "target": session_data.get("spec", {}).get("target", None),
            "distractor": session_data.get("spec", {}).get("distractor", None),
        }
        device = session_data.get("device", "unknown")
        notes = session_data.get("notes")

        # Prefer file name for the following, or defer to JSON if mangled

        if "sub-" in path.split(os.sep)[-1]:
            animal_id = path.split(os.sep)[-1].split("_")[0].strip("sub-")
        else:
            animal_id = session_data.get("animal_id", "unknown")

        if "exp-" in path.split(os.sep)[-1]:
            experiment = path.split(os.sep)[-1].split("_")[1].replace("exp-", "")
        else:
            experiment = session_data.get("experiment", "unknown")

        if "ses-" in path.split(os.sep)[-1]:
            session_date = datetime.datetime.strptime(
                path.split(os.sep)[-1].split("_")[2].strip("ses-")[:8], "%Y%m%d"
            ).date()
        else:
            session_date = datetime.datetime.fromisoformat(session_start_time).date()

        if "behaviour-" in path.split(os.sep)[-1]:
            interaction = path.split(os.sep)[-1].split("_")[-1].replace("behaviour-", "").replace(".json", "")
        else:
            interaction = session_data.get("interaction", "unknown")

    return {
        "animal_id": animal_id,
        "experiment": experiment,
        "session_date": session_date,
        "interaction": interaction,
        "protocol": protocol,
        "version": version,
        "duration": duration,
        "session_start_time": session_start_time,
        "response_device": response_device,
        "reward_profile": reward_profile,
        "stimulus_duration": stimulus_duration,
        "iti": iti,
        "corrections_enabled": corrections_enabled,
        "device": device,
        "stimuli": stimuli,
        "notes": notes,
    }


def extract_trials(path: str, to_csv: bool = False, output_dir: str = ".") -> pd.DataFrame:
    """Parse a Visiomode JSON file and return a trials dataframe. Optionally save to CSV.

    Args:
        path (str): Path to Visiomode JSON file.
        to_csv (bool, optional): Save trials dataframe as a CSV. If True, indicate which directory to save in via `output_dir`. Defaults to False.
        output_dir (str, optional): Output directory for trials CSV file. Only used if `to_csv` is True. Defaults to current directory.

    Returns:
        pd.DataFrame: Session trials dataframe, where each row is a trial.
    """
    metadata = get_metadata(path)

    trials: Iterator[dict]
    with open(path, "r") as fp:
        session_data = json.load(fp)
        trials = _flatten_trials(session_data, metadata=metadata)

    session = [
        {
            "animal_id": metadata.get("animal_id"),
            "session_date": metadata.get("session_date"),
            "protocol": metadata.get("protocol"),
            "interaction": metadata.get("interaction"),
            "experiment": metadata.get("experiment"),
            **trial,
        }
        for trial in trials
    ]

    df = pd.DataFrame(session)

    # Convert legacy outcomes if they're still about
    df = df.replace({"hit": "correct", "false_alarm": "incorrect", "miss": "no_response"})

    if to_csv:
        out_path = f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_ses-{str(metadata.get('session_date'))}_behaviour-{metadata.get('interaction')}_trials.csv"
        df.to_csv(out_path)

    return df


def generate_report(path: str, output_dir: str = ".") -> str:
    template = env.get_template(SESSION_REPORT_TEMPLATE)

    template_identifiers = {
        "subject_id": ...,
        "session_date": ...,
        "experiment_id": ...,
        "duration": ...,
        "trials_num": ...,
        "protocol": ...,
        "response_device": ...,
        "reward_profile": ...,
        "iti": ...,
        "si": ...,
        "corrections_enabled": ...,
        "stimuli": ...,
    }

    out_path = output_dir / Path(path.split("/")[-1].replace(".h5", "_report.html"))
    out_path.write_text(template.render(template_identifiers), encoding="utf-8")
    return str(out_path)


def _flatten_trials(session: dict, metadata: dict) -> Iterator[dict]:
    session_start_time = datetime.datetime.fromisoformat(metadata.get("session_start_time", ""))

    for trial in session.get("trials", []):
        start_time = (datetime.datetime.fromisoformat(trial["timestamp"]) - session_start_time).total_seconds()

        stimulus_duration = metadata.get("stimulus_duration", -1) / 1000

        stop_time = start_time + trial["iti"] + stimulus_duration
        if trial.get("response") and trial.get("response", {}).get("timestamp"):
            stop_time = (
                datetime.datetime.fromisoformat(trial["response"]["timestamp"]) - session_start_time
            ).total_seconds()

        response = trial.get("response").get("name") if trial.get("response") else None
        response_time = trial["response_time"]

        pos_x = trial.get("response").get("pos_x", 0) if trial.get("response") else None
        pos_y = trial.get("response").get("pos_y", 0) if trial.get("response") else None
        dist_x = trial.get("response").get("dist_x", 0) if trial.get("response") else None
        dist_y = trial.get("response").get("dist_y", 0) if trial.get("response") else None

        stimulus: dict = {}
        if trial.get("stimulus"):
            if trial.get("stimulus") == "None":
                stimulus = {}
            elif trial.get("stimulus").get("common_name"):
                # handle single stimulus on screen tasks
                stimulus = {f"stim_{key}": value for key, value in trial.get("stimulus").items()}
            elif trial.get("stimulus").get("target"):
                # 2AFC / nAFC
                target_stim = {f"target_{key}": value for key, value in trial.get("stimulus").get("target").items()}
                distractor_stim = {
                    f"distractor_{key}": value for key, value in trial.get("stimulus").get("distractor").items()
                }
                stimulus = {**target_stim, **distractor_stim}
        else:
            stimulus = {
                "target_id": metadata.get("stimuli", {}).get("target"),
                "distractor_id": metadata.get("stimuli", {}).get("distractor"),
            }

        cue_onset = start_time + trial["iti"] if stimulus else "NA"

        sdt_type = None
        if trial.get("sdt_type"):
            sdt_type = trial.get("sdt_type")
        elif metadata.get("protocol") == "gonogo":
            if trial.get("response") and trial.get("outcome") == "correct":
                sdt_type = HIT
            elif trial.get("response") and trial.get("outcome") == "incorrect":
                sdt_type = FALSE_ALARM
            elif not trial.get("response") and trial.get("outcome") == "correct":
                sdt_type = CORRECT_REJECTION
            elif not trial.get("response") and trial.get("outcome") == "incorrect":
                sdt_type = MISS
            else:
                sdt_type = "NA"

        yield {
            "start_time": start_time,
            "stop_time": stop_time,
            "cue_onset": cue_onset,
            "response": response,
            "response_time": response_time,
            "outcome": trial["outcome"],
            "correction": trial["correction"],
            "pos_x": pos_x,
            "pos_y": pos_y,
            "dist_x": dist_x,
            "dist_y": dist_y,
            "sdt_type": sdt_type,
            **stimulus,
        }
