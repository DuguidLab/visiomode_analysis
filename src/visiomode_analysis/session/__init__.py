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

from visiomode_analysis.session import metrics


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
    out_dir = preprocess_session(**kwargs)
    click.echo(f"Files saved under {out_dir}")


def preprocess_session(path: str, output_dir: str = ".") -> str:
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
            "target_id": session_data.get("spec", {}).get("target"),
            **{
                f"target_{key.replace('t_', '')}": value
                for key, value in session_data.get("spec", {}).items()
                if key.startswith("t_")
            },
            "distractor_id": session_data.get("spec", {}).get("distractor"),
            **{
                f"distractor_{key.replace('d_', '')}": value
                for key, value in session_data.get("spec", {}).items()
                if key.startswith("d_")
            },
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
            environment = path.split(os.sep)[-1].split("_")[-1].replace("behaviour-", "").replace(".json", "")
        else:
            environment = session_data.get("environment", "unknown")

    return {
        "animal_id": animal_id,
        "experiment": experiment,
        "session_date": session_date,
        "environment": environment,
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
            "environment": metadata.get("environment"),
            "experiment": metadata.get("experiment"),
            **trial,
        }
        for trial in trials
    ]

    df = pd.DataFrame(session)

    # Convert legacy outcomes if they're still about
    df.outcome = df.outcome.replace({"hit": "correct", "false_alarm": "incorrect", "miss": "no_response"})

    if to_csv:
        out_path = f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_ses-{str(metadata.get('session_date'))}_behaviour-{metadata.get('environment')}_trials.csv"
        df.to_csv(out_path)

    return df


def summarise(path: str) -> dict:
    metadata = get_metadata(path)
    df = extract_trials(path)

    # Trial counts
    precued = len(df[(df.outcome == "precued")])

    correct = len(df[(df.outcome == "correct") & (df.correction == False)])  # noqa: E712
    correct_wc = len(df[(df.outcome == "correct")])
    incorrect = len(df[(df.outcome == "incorrect") & (df.correction == False)])  # noqa: E712
    incorrect_wc = len(df[(df.outcome == "incorrect")])

    correction_trials = len(df[(df.outcome == "incorrect") & (df.correction == True)])  # noqa: E712

    hits = len(df[(df.sdt_type == "hit") & (df.correction == False)])  # noqa: E712
    hits_wc = len(df[(df.sdt_type == "hit")])

    false_alarms = len(df[(df.sdt_type == "false_alarm") & (df.correction == False)])  # noqa: E712
    false_alarms_wc = len(df[(df.sdt_type == "false_alarm")])

    correct_rejections = len(df[(df.sdt_type == "correct_rejection") & (df.correction == False)])  # noqa: E712
    correct_rejections_wc = len(df[(df.sdt_type == "correct_rejection")])

    misses = len(df[(df.sdt_type == "miss") & (df.correction == False)])  # noqa: E712
    misses_wc = len(df[(df.sdt_type == "miss")])

    cued = hits + misses + false_alarms + correct_rejections
    cued_wc = hits_wc + misses_wc + false_alarms_wc + correct_rejections_wc

    total = cued_wc + precued

    # Trial ratios
    percentage_correct = metrics.percentage_correct(num_correct=correct, num_cued=cued)
    percentage_correct_wc = metrics.percentage_correct(num_correct=correct_wc, num_cued=cued_wc)

    cued_ratio = cued / precued if precued > 0 else 1.0
    correction_ratio = correction_trials / incorrect if incorrect > 0 else 0.0

    # Signal detection theory metrics
    is_2afc = True if metadata.get("protocol", "").contains("afc") else False

    hit_rate = (hits + 0.5) / (hits + misses + 1.0)
    hit_rate_wc = (hits_wc + 0.5) / (hits_wc + misses_wc + 1.0)
    fa_rate = (false_alarms + 0.5) / (correct_rejections + false_alarms + 1.0)
    fa_rate_wc = (false_alarms_wc + 0.5) / (correct_rejections_wc + false_alarms_wc + 1.0)

    d_prime = metrics.d_prime(hit_rate, fa_rate, afc_correction=is_2afc)
    d_prime_wc = metrics.d_prime(hit_rate_wc, fa_rate_wc, afc_correction=is_2afc)

    bias = metrics.bias(hit_rate, fa_rate)
    bias_wc = metrics.bias(hit_rate_wc, fa_rate_wc)

    # Perseveration
    perseveration = metrics.perseveration(num_correction_trials=correction_trials, num_incorrect=incorrect_wc)

    # Reaction time metrics
    rt = np.median(df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"])  # noqa: E712
    rt_wc = np.median(df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"])

    rt_hits = np.median(df[(df.sdt_type == "hit") & (df.correction == False)]["response_time"])  # noqa: E712
    rt_hits_wc = np.median(df[(df.sdt_type == "hit")]["response_time"])

    rt_false_alarms = np.median(df[(df.sdt_type == "false_alarm") & (df.correction == False)]["response_time"])  # noqa: E712
    rt_false_alarms_wc = np.median(df[(df.sdt_type == "false_alarm")]["response_time"])

    return {}


def generate_report(path: str, output_dir: str = ".") -> str:
    template = env.get_template(SESSION_REPORT_TEMPLATE)

    metadata = get_metadata(path)

    template_identifiers = {
        "subject_id": metadata.get("animal_id"),
        "session_date": str(metadata.get("session_date")),
        "experiment_id": metadata.get("experiment_id"),
        "duration": metadata.get("duration"),
        "trials_num": ...,
        "protocol": metadata.get("protocol"),
        "response_device": metadata.get("response_device"),
        "reward_profile": metadata.get("reward_profile"),
        "iti": metadata.get("iti"),
        "si": metadata.get("si"),
        "corrections_enabled": metadata.get("corrections_enabled"),
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

        response = trial.get("response").get("name") or "unknown" if trial.get("response") else None
        if response == "unknown":
            if metadata.get("environment") == "hf":
                response = "leverpush"
            elif metadata.get("environment") == "freelymoving":
                response = "touch"

        response_time = trial["response_time"]
        # disregard negative response times
        response_time = np.nan if response_time < 0 else response_time

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
        else:  # handle older versions of visiomode
            if metadata.get("protocol") == "gonogo" or metadata.get("protocol") == "targetonly":
                if (trial.get("response") and trial.get("outcome") == "correct") or (
                    not trial.get("response") and trial.get("outcome") == "incorrect"
                ):
                    stimulus = {
                        **{
                            f"stim_{key.replace('target_', '')}": value
                            for key, value in metadata.get("stimuli", {}).items()
                            if key.startswith("target_")
                        },
                    }
                elif (trial.get("response") and trial.get("outcome") == "incorrect") or (
                    not trial.get("response") and trial.get("outcome") == "correct"
                ):
                    stimulus = {
                        **{
                            f"stim_{key.replace('distractor_', '')}": value
                            for key, value in metadata.get("stimuli", {}).items()
                            if key.startswith("distractor_")
                        },
                    }
            else:  # 2AFC
                stimulus = {key: value for key, value in metadata.get("stimuli", {}).items()}

        cue_onset = start_time + trial["iti"] if stimulus else np.nan

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
