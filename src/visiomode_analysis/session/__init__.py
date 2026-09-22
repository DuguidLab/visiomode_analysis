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
import h5py
import pandas as pd
import numpy as np
import numpy.typing as npt

from pathlib import Path
from jinja2 import Environment
from jinja2 import PackageLoader
from jinja2 import select_autoescape

from collections.abc import Iterator
from typing import Any

from visiomode_analysis.session import metrics, plots
import visiomode_analysis.session.regressor as rgr


SESSION_REPORT_TEMPLATE = "session.html"

HIT = "hit"
MISS = "miss"
FALSE_ALARM = "false_alarm"
CORRECT_REJECTION = "correct_rejection"

# Protocol identifiers that describe a target-only task (a single target stimulus, no distractor).
# "singletarget" is the name older Visiomode versions used for what is now "targetonly".
TARGETONLY_PROTOCOLS = ("targetonly", "singletarget")

# Older Visiomode versions wrote SDT-style labels as the trial outcome; map them to the current
# outcome vocabulary so downstream logic only has to deal with one set of labels.
LEGACY_OUTCOME_LABELS = {"hit": "correct", "false_alarm": "incorrect", "miss": "no_response"}


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
@click.option(
    "--no-report",
    is_flag=True,
    default=False,
    help="If set, do not generate a session report.",
)
@click.option(
    "--with-regressors",
    is_flag=True,
    default=False,
    help="If set, generate regressors for the session.",
)
@click.option(
    "--regressor-timestamps",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a CSV/TXT file of ISO timestamps, or a mesoscopy H5 with /timestamps_aligned, for regressor generation.",
)
def session_cmd(**kwargs):
    """Generate a session report and extract trials from a Visiomode JSON file."""
    out_dir = preprocess_session(**kwargs)
    click.echo(f"Files saved under {out_dir}")


@click.command("regressors")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(dir_okay=True),
    default=".",
    help="Output directory for regressors files.",
)
@click.option(
    "--regressor-timestamps",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to a CSV/TXT file of ISO timestamps, or a mesoscopy H5 with /timestamps_aligned, for regressor generation.",
)
def regressors_cmd(**kwargs):
    """Generate regressors for a session based on the protocol."""
    trials_df = get_trials(kwargs["path"])
    meta = get_metadata(kwargs["path"])
    generate_regressors(trials_df, meta, kwargs["regressor_timestamps"], output_dir=kwargs["output_dir"])
    click.echo(f"Regressors saved under {kwargs['output_dir']}")


@click.command("session-start-time")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False),
)
def session_start_time_cmd(path: str):
    """Print the behaviour session start time (the JSON `timestamp` key) and nothing else."""
    click.echo(get_metadata(path)["session_start_time"])


def preprocess_session(
    path: str,
    output_dir: str = ".",
    no_report: bool = False,
    with_regressors: bool = False,
    regressor_timestamps: str | None = None,
) -> str:
    """Generate a session summary report and trials file from a raw Visiomode JSON.

    Args:
        path (str): Path to Visiomode JSON file.
        output_dir (str, optional): Output directory for report and trials files. Defaults to ".".
        no_report (bool, optional): Whether to generate a session report. Defaults to False.
        with_regressors (bool, optional): Whether to generate regressors for the session. Defaults to False.
        regressor_timestamps (str, optional): Path to a CSV/TXT file of ISO timestamps, or a mesoscopy H5, for regressor generation. Defaults to None.
    Returns:
        str: Returns directory under which files were saved
    """
    trials_df = get_trials(path, to_csv=True, output_dir=output_dir)
    meta = get_metadata(path)

    if not no_report:
        generate_report(path, output_dir=output_dir)

    if with_regressors:
        if not regressor_timestamps:
            raise ValueError("Regressor timestamps file must be provided when generating regressors.")
        generate_regressors(trials_df, meta, regressor_timestamps, output_dir=output_dir)

    return output_dir


def is_targetonly(protocol: str | None) -> bool:
    """Whether a protocol identifier refers to a target-only task (see `TARGETONLY_PROTOCOLS`)."""
    return protocol in TARGETONLY_PROTOCOLS


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
        corrections_enabled = session_data.get("spec", {}).get("corrections_enabled", False)
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
        "behaviour_session": Path(path).stem,
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


def get_trials(path: str, to_csv: bool = False, output_dir: str = ".") -> pd.DataFrame:
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

    # Legacy outcomes are already normalised in `_flatten_trials`; repeat here so a DataFrame
    # built any other way is treated the same.
    df["outcome"] = df["outcome"].replace(LEGACY_OUTCOME_LABELS)

    if to_csv:
        out_path = f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_ses-{str(metadata.get('session_date')).replace('-', '')}_behaviour-{metadata.get('protocol')}_trials.csv"
        df.to_csv(out_path)

    return df


def get_rts(path: str, sdt_type=None, include_corrections=True) -> npt.NDArray:
    trials = get_trials(path=path)

    if not include_corrections:
        trials = trials[trials.correction == False]  # noqa: E712

    if sdt_type:
        return np.array(trials[(trials.response.notnull()) & (trials.sdt_type == sdt_type)].response_time.values)

    return np.array(trials[(trials.response.notnull()) & (trials.cue_onset.notnull())].response_time.values)


def summary(path: str) -> dict:
    """Summarise session from a JSON or trials.csv file

    Args:
        path (str): Path to (preprocessed) trials.csv or raw JSON

    Returns:
        dict: Summary dictionary
    """
    if path.endswith(".json"):
        metadata = get_metadata(path)
        df = get_trials(path)
    else:
        df = pd.read_csv(path)
        metadata = {
            "animal_id": df["animal_id"][0],
            "session_date": df["session_date"][0],
            "protocol": df["protocol"][0],
            "environment": df["environment"][0],
            "experiment": df["experiment"][0],
        }

    # Trial counts
    correct = len(df[(df["outcome"] == "correct") & (df["correction"] == False)])  # noqa: E712
    correct_wc = len(df[(df["outcome"] == "correct")])
    incorrect = len(df[(df["outcome"] == "incorrect") & (df["correction"] == False)])  # noqa: E712
    incorrect_wc = len(df[(df["outcome"] == "incorrect")])

    correction_trials = len(df[(df["outcome"] == "incorrect") & (df["correction"] == True)])  # noqa: E712

    hits = len(df[(df["sdt_type"] == "hit") & (df["correction"] == False)])  # noqa: E712
    hits_wc = len(df[(df["sdt_type"] == "hit")])

    false_alarms = len(df[(df["sdt_type"] == "false_alarm") & (df["correction"] == False)])  # noqa: E712
    false_alarms_wc = len(df[(df["sdt_type"] == "false_alarm")])

    correct_rejections = len(df[(df["sdt_type"] == "correct_rejection") & (df["correction"] == False)])  # noqa: E712
    correct_rejections_wc = len(df[(df["sdt_type"] == "correct_rejection")])

    misses = len(df[(df["sdt_type"] == "miss") & (df["correction"] == False)])  # noqa: E712
    misses_wc = len(df[(df["sdt_type"] == "miss")])

    cued = hits + misses + false_alarms + correct_rejections
    cued_wc = hits_wc + misses_wc + false_alarms_wc + correct_rejections_wc

    precued = len(df[(df["outcome"] == "precued")])

    total = cued_wc + precued

    # Trial ratios
    percentage_correct = (correct / cued) * 100 if cued > 0 else 100
    percentage_correct_wc = (correct_wc / cued_wc) * 100 if cued_wc > 0 else 100

    cued_ratio = cued / precued if precued > 0 else 1.0
    correction_ratio = cued / correction_trials if correction_trials > 0 else 0.0

    # Signal detection theory metrics
    _is_2afc = True if "afc" in metadata.get("protocol", "") else False

    hit_rate = (hits + 0.5) / (hits + misses + 1.0)
    hit_rate_wc = (hits_wc + 0.5) / (hits_wc + misses_wc + 1.0)
    fa_rate = (false_alarms + 0.5) / (correct_rejections + false_alarms + 1.0)
    fa_rate_wc = (false_alarms_wc + 0.5) / (correct_rejections_wc + false_alarms_wc + 1.0)

    d_prime = metrics.d_prime(hit_rate, fa_rate, afc_correction=_is_2afc)
    d_prime_wc = metrics.d_prime(hit_rate_wc, fa_rate_wc, afc_correction=_is_2afc)

    decision_criterion = metrics.criterion(hit_rate, fa_rate)
    decision_criterion_wc = metrics.criterion(hit_rate_wc, fa_rate_wc)

    # Perseveration
    perseveration = metrics.perseveration(num_correction_trials=correction_trials, num_incorrect=incorrect_wc)

    # Reaction time metrics
    rt = float(
        np.median(df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"])
    )  # noqa: E712
    rt_wc = float(np.median(df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"]))

    rt_hits = float(np.median(df[(df.sdt_type == "hit") & (df.correction == False)]["response_time"]))  # noqa: E712
    rt_hits_wc = float(np.median(df[(df.sdt_type == "hit")]["response_time"]))

    rt_false_alarms = float(np.median(df[(df.sdt_type == "false_alarm") & (df.correction == False)]["response_time"]))  # noqa: E712
    rt_false_alarms_wc = float(np.median(df[(df.sdt_type == "false_alarm")]["response_time"]))

    try:
        rt_iqr = float(
            np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"], 75
            )
            - np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"],
                25,
            )
        )

        rt_iqr_wc = float(
            np.percentile(df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"], 75)
            - np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"],
                25,
            )
        )

        # interdecile range (10th to 90th percentile)
        rt_idr = float(
            np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"], 90
            )
            - np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued") & (df.correction == False)]["response_time"], 10
            )
        )
        rt_idr_wc = float(
            np.percentile(df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"], 90)
            - np.percentile(
                df[(df.response.notnull()) & (df.outcome != "precued")]["response_time"],
                10,
            )
        )
    except IndexError:
        rt_iqr = np.nan
        rt_iqr_wc = np.nan
        rt_idr = np.nan
        rt_idr_wc = np.nan

    return {
        "animal_id": metadata.get("animal_id"),
        "session_date": metadata.get("session_date"),
        "protocol": metadata.get("protocol"),
        "environment": metadata.get("environment"),
        "experiment": metadata.get("experiment"),
        "correct": correct,
        "correct_wc": correct_wc,
        "incorrect": incorrect,
        "incorrect_wc": incorrect_wc,
        "correction_trials": correction_trials,
        "hits": hits,
        "hits_wc": hits_wc,
        "false_alarms": false_alarms,
        "false_alarms_wc": false_alarms_wc,
        "correct_rejections": correct_rejections,
        "correct_rejections_wc": correct_rejections_wc,
        "misses": misses,
        "misses_wc": misses_wc,
        "cued": cued,
        "cued_wc": cued_wc,
        "precued": precued,
        "total": total,
        "percentage_correct": percentage_correct,
        "percentage_correct_wc": percentage_correct_wc,
        "cued_ratio": cued_ratio,
        "correction_ratio": correction_ratio,
        "hit_rate": hit_rate,
        "hit_rate_wc": hit_rate_wc,
        "fa_rate": fa_rate,
        "fa_rate_wc": fa_rate_wc,
        "d_prime": d_prime,
        "d_prime_wc": d_prime_wc,
        "decision_criterion": decision_criterion,
        "decision_criterion_wc": decision_criterion_wc,
        "perseveration": perseveration,
        "rt": rt,
        "rt_wc": rt_wc,
        "rt_hits": rt_hits,
        "rt_hits_wc": rt_hits_wc,
        "rt_false_alarms": rt_false_alarms,
        "rt_false_alarms_wc": rt_false_alarms_wc,
        "rt_iqr": rt_iqr,
        "rt_iqr_wc": rt_iqr_wc,
        "rt_idr": rt_idr,
        "rt_idr_wc": rt_idr_wc,
    }


def generate_regressors(
    trials_df: pd.DataFrame, metadata: dict, regressor_timestamps_path: str, output_dir: str = "."
) -> str:
    """Generate regressors for the session based on the protocol.

    CSV/TXT inputs hold absolute ISO timestamps, which are recalculated relative to the session start time in the
    metadata. H5 inputs (written by `mesoscopy align`) hold already-aligned timestamps in `/timestamps_aligned`,
    which are used as-is after checking that the file's `session_start_time` attribute matches the session.

    Args:
        trials_df (pd.DataFrame): A DataFrame containing trial data with columns for trial type,
            start time, and stop time.
        metadata (dict): A dictionary containing session metadata.
        regressor_timestamps_path (str): Path to a CSV or TXT file of ISO timestamps, or a mesoscopy H5 file with a
            `/timestamps_aligned` dataset. Typically corresponds to the frame times of an imaging session or other
            continuous recording.
        output_dir (str, optional): Output directory for saving regressors. Defaults to ".".

    Returns:
        str: Path to the generated regressors file.

    Raises:
        ValueError: If the timestamps file has an unsupported extension, or (for H5) the `/timestamps_aligned`
            dataset is missing or its `session_start_time` attribute does not match the session.
        NotImplementedError: If the protocol specified in the metadata is not supported for regressor generation.

    Note:
        The output regressors are saved as a .npz file containing `regressors`, `labels`, `timestamps` (aligned to
        behaviour start), `trial_idx`, `session_start_time` and `behaviour_session`.
    """
    session_start_time = datetime.datetime.fromisoformat(metadata.get("session_start_time", ""))

    if regressor_timestamps_path.endswith(".h5"):
        timestamps = _read_aligned_timestamps(regressor_timestamps_path, session_start_time)
    else:
        if regressor_timestamps_path.endswith(".csv"):
            source_timestamps = pd.read_csv(regressor_timestamps_path).to_numpy().flatten()
        elif regressor_timestamps_path.endswith(".txt"):
            source_timestamps = np.loadtxt(regressor_timestamps_path, dtype=str)
        else:
            raise ValueError("Regressor timestamps file must be in CSV, TXT or H5 format.")

        # recalculate timestamps to align to behaviour
        timestamps = np.array(
            [
                (datetime.datetime.fromisoformat(timestamp) - session_start_time).total_seconds()
                for timestamp in source_timestamps
            ]
        )

    outpath = f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_ses-{str(metadata.get('session_date')).replace('-', '')}_behaviour-{metadata.get('protocol')}_regressors.npz"

    if metadata.get("protocol") == "gonogo":
        regressors, labels, trial_idx = rgr.generate_gonogo_regressors(trials_df, timestamps, trial_epoch_only=True)
        np.savez(
            outpath,
            regressors=regressors,
            labels=np.array(list(labels.values())),
            timestamps=timestamps,
            trial_idx=trial_idx,
            session_start_time=metadata.get("session_start_time", ""),
            behaviour_session=metadata.get("behaviour_session", ""),
        )
    else:
        raise NotImplementedError(f"Regressor generation not implemented for protocol {metadata.get('protocol')}.")

    return outpath


ALIGNED_TIMESTAMPS_DATASET = "timestamps_aligned"


def _read_aligned_timestamps(path: str, session_start_time: datetime.datetime) -> npt.NDArray[np.float64]:
    """Read `/timestamps_aligned` from a mesoscopy H5, checking its `session_start_time` attribute against the session."""
    with h5py.File(path, "r") as h5:
        if ALIGNED_TIMESTAMPS_DATASET not in h5:
            raise ValueError(f"Dataset /{ALIGNED_TIMESTAMPS_DATASET} not found in {path}.")
        dataset = h5[ALIGNED_TIMESTAMPS_DATASET]
        h5_start_time = dataset.attrs.get("session_start_time")
        if h5_start_time is None:
            raise ValueError(f"/{ALIGNED_TIMESTAMPS_DATASET} in {path} has no session_start_time attribute.")
        if isinstance(h5_start_time, bytes):
            h5_start_time = h5_start_time.decode()
        if datetime.datetime.fromisoformat(str(h5_start_time)) != session_start_time:
            raise ValueError(
                f"session_start_time mismatch: {path} has {h5_start_time}, behaviour session is "
                f"{session_start_time.isoformat()}."
            )
        return np.asarray(dataset[()], dtype=np.float64)


def generate_report(path: str, output_dir: str = ".") -> str:
    template = env.get_template(SESSION_REPORT_TEMPLATE)

    metadata = get_metadata(path)
    trials = get_trials(path)
    session_summary = summary(path=path)

    template_identifiers = {
        "subject_id": metadata.get("animal_id"),
        "session_date": str(metadata.get("session_date")),
        "experiment_id": metadata.get("experiment_id"),
        "duration": metadata.get("duration"),
        "trials_num": session_summary.get("total"),
        "protocol": metadata.get("protocol"),
        "is_targetonly": is_targetonly(metadata.get("protocol")),
        "response_device": metadata.get("response_device"),
        "reward_profile": metadata.get("reward_profile"),
        "iti": metadata.get("iti"),
        "stimulus_duration": metadata.get("stimulus_duration"),
        "corrections_enabled": metadata.get("corrections_enabled"),
        "stimuli": metadata.get("stimuli"),
        "notes": metadata.get("notes"),
        "summary": session_summary,
        "fig_success_pie": plots.plot_success_pie(
            session_summary.get("correct", 0),
            session_summary.get("incorrect", 0),
            session_summary.get("miss", 0),
            as_html=True,
        ),
        "fig_success_pie_wc": plots.plot_success_pie(
            session_summary.get("correct_wc", 0),
            session_summary.get("incorrect_wc", 0),
            session_summary.get("miss_wc", 0),
            as_html=True,
        ),
        "fig_cued_pie": plots.plot_cued_pie(session_summary.get("cued"), session_summary.get("precued"), as_html=True),
        "fig_cued_pie_wc": plots.plot_cued_pie(
            session_summary.get("cued_wc"), session_summary.get("precued"), as_html=True
        ),
        "fig_rt_median": plots.plot_rt_median(
            get_rts(path=path, include_corrections=False),
            stimulus_duration=metadata.get("stimulus_duration", 4000) / 1000,
            as_html=True,
        )
        if is_targetonly(metadata.get("protocol"))
        else plots.plot_rt_medians_from_dict(
            {
                "all": get_rts(path=path, include_corrections=False),
                "hits": get_rts(path=path, sdt_type="hit", include_corrections=False),
                "false_alarms": get_rts(path=path, sdt_type="false_alarm", include_corrections=False),
            },
            stimulus_duration=metadata.get("stimulus_duration", 4000) / 1000,
            as_html=True,
        ),
        "fig_rt_median_wc": plots.plot_rt_median(
            get_rts(path=path, include_corrections=True),
            stimulus_duration=metadata.get("stimulus_duration", 4000) / 1000,
            as_html=True,
        )
        if is_targetonly(metadata.get("protocol"))
        else plots.plot_rt_medians_from_dict(
            {
                "all": get_rts(path=path, include_corrections=True),
                "hits": get_rts(path=path, sdt_type="hit", include_corrections=True),
                "false_alarms": get_rts(path=path, sdt_type="false_alarm", include_corrections=True),
            },
            stimulus_duration=metadata.get("stimulus_duration", 4000) / 1000,
            as_html=True,
        ),
        "fig_presentation_breakdown": plots.plot_correction_pie(
            session_summary.get("cued", 0), session_summary.get("correction_trials"), as_html=True
        ),
        "fig_presentation_ratio": plots.plot_single_yvalue(
            session_summary.get("correction_ratio", 0), 0, 5, as_html=True
        ),
        "fig_perseveration_index": plots.plot_single_yvalue(
            session_summary.get("perseveration", 0), 0, 1, as_html=True
        ),
        "fig_roc": plots.plot_roc(
            hit_rate=session_summary.get("hit_rate", 0),
            fa_rate=session_summary.get("fa_rate", 0),
            hit_rate_wc=session_summary.get("hit_rate_wc", 0),
            fa_rate_wc=session_summary.get("fa_rate_wc", 0),
            as_html=True,
        ),
        "fig_d_prime": plots.plot_dprime(
            d_prime=session_summary.get("d_prime", 0), d_prime_wc=session_summary.get("d_prime_wc", None), as_html=True
        ),
        "fig_decision_criterion": plots.plot_criterion(
            criterion=session_summary.get("decision_criterion", 0),
            criterion_wc=session_summary.get("decision_criterion_wc", None),
            as_html=True,
        ),
        "fig_response_sdt": plots.plot_sdt_pie(
            num_hits=session_summary.get("hits", 0),
            num_false_alarms=session_summary.get("false_alarms", 0),
            num_correct_rejections=session_summary.get("correct_rejections", 0),
            num_misses=session_summary.get("misses", 0),
            as_html=True,
        ),
        "fig_response_sdt_wc": plots.plot_sdt_pie(
            num_hits=session_summary.get("hits_wc", 0),
            num_false_alarms=session_summary.get("false_alarms_wc", 0),
            num_correct_rejections=session_summary.get("correct_rejections_wc", 0),
            num_misses=session_summary.get("misses_wc", 0),
            as_html=True,
        ),
        "fig_response_timeseries": plots.plot_trial_timeseries(trials=trials, as_html=True),
        "fig_sdt_timeseries": plots.plot_trial_timeseries(trials=trials, use_sdt=True, as_html=True),
    }

    out_path = Path(
        f"{output_dir}{os.sep}sub-{metadata.get('animal_id')}_exp-{metadata.get('experiment')}_ses-{str(metadata.get('session_date')).replace('-', '')}_behaviour-{metadata.get('protocol')}_report-session.html"
    )
    out_path.write_text(template.render(template_identifiers), encoding="utf-8")
    return str(out_path)


def _normalise_legacy_outcome(trial: Any) -> Any:
    """Return a copy of a raw trial with any legacy outcome label mapped to the current vocabulary.

    Done up front so the stimulus reconstruction and SDT inference in `_flatten_trials` see the same
    labels regardless of which Visiomode version wrote the session.
    """
    return {**trial, "outcome": LEGACY_OUTCOME_LABELS.get(trial["outcome"], trial["outcome"])}


def _flatten_trials(session: dict, metadata: dict) -> Iterator[dict]:
    session_start_time = datetime.datetime.fromisoformat(metadata.get("session_start_time", ""))

    for trial in session.get("trials", []):
        trial = _normalise_legacy_outcome(trial)

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
            if metadata.get("protocol") == "gonogo":
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
            elif is_targetonly(metadata.get("protocol")):
                if (trial.get("response") and (trial.get("outcome") == "correct")) or (
                    trial.get("outcome") == "no_response"
                ):
                    stimulus = {
                        **{
                            f"stim_{key.replace('target_', '')}": value
                            for key, value in metadata.get("stimuli", {}).items()
                            if key.startswith("target_")
                        },
                    }
            else:  # 2AFC
                stimulus = {key: value for key, value in metadata.get("stimuli", {}).items()}

        cue_onset = start_time + trial["iti"] if stimulus else np.nan

        sdt_type = None
        if trial.get("sdt_type"):
            sdt_type = trial.get("sdt_type")
        elif metadata.get("protocol") == "gonogo" or is_targetonly(metadata.get("protocol")):
            if trial.get("response") and trial.get("outcome") == "correct":
                sdt_type = HIT
            elif trial.get("response") and trial.get("outcome") == "incorrect":
                sdt_type = FALSE_ALARM
            elif not trial.get("response") and trial.get("outcome") == "correct":
                sdt_type = CORRECT_REJECTION
            elif not trial.get("response") and (
                trial.get("outcome") == "incorrect" or trial.get("outcome") == "no_response"
            ):
                sdt_type = MISS
            else:
                sdt_type = None

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
