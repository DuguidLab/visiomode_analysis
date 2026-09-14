import datetime
import json
import os
import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest

from visiomode_analysis import session

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def gonogo_session_json_path() -> str:
    """Path to a real (anonymised) Go/NoGo session recording, used as an integration fixture."""
    return str(REPO_ROOT / "exploratory" / "test_data" / "example-gonogo-leverpush.json")


def _gonogo_regressor_timestamps(gonogo_session_json_path) -> list[str]:
    """Absolute ISO timestamps that land just after the cue onset of real trials in
    `gonogo_session_json_path`, so regressor generation has something to find."""
    trials = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    session_start = datetime.datetime.fromisoformat(metadata["session_start_time"])

    offsets = trials["cue_onset"].dropna().head(3) + 0.2
    return [(session_start + datetime.timedelta(seconds=offset)).isoformat() for offset in offsets]


@pytest.fixture
def gonogo_regressor_timestamps_csv_path(tmp_path, gonogo_session_json_path) -> str:
    """A `--regressor-timestamps` CSV built from `_gonogo_regressor_timestamps`."""
    timestamps = _gonogo_regressor_timestamps(gonogo_session_json_path)

    path = tmp_path / "regressor_timestamps.csv"
    path.write_text("timestamp\n" + "\n".join(timestamps) + "\n")
    return str(path)


@pytest.fixture
def gonogo_regressor_timestamps_txt_path(tmp_path, gonogo_session_json_path) -> str:
    """The whitespace-delimited TXT equivalent of `gonogo_regressor_timestamps_csv_path`."""
    timestamps = _gonogo_regressor_timestamps(gonogo_session_json_path)

    path = tmp_path / "regressor_timestamps.txt"
    path.write_text("\n".join(timestamps) + "\n")
    return str(path)


@pytest.fixture
def write_aligned_h5(gonogo_session_json_path):
    """Factory fixture that writes a synthetic mesoscopy-style H5 following the alignment contract:
    a float64 `/timestamps_aligned` dataset (seconds from behaviour start) with `session_start_time`,
    `behaviour_session` and `offset_s` attributes. Defaults to timestamps that land inside real trials
    of `gonogo_session_json_path`."""

    def _write(path, timestamps=None, session_start_time=None, behaviour_session=None, include_dataset=True):
        metadata = session.get_metadata(gonogo_session_json_path)
        if timestamps is None:
            trials = session.get_trials(gonogo_session_json_path)
            timestamps = (trials["cue_onset"].dropna().head(3) + 0.2).to_numpy(dtype=np.float64)
        if session_start_time is None:
            session_start_time = metadata["session_start_time"]
        if behaviour_session is None:
            behaviour_session = metadata["behaviour_session"]

        with h5py.File(path, "w") as h5:
            if include_dataset:
                dataset = h5.create_dataset("timestamps_aligned", data=np.asarray(timestamps, dtype=np.float64))
                dataset.attrs["session_start_time"] = session_start_time
                dataset.attrs["behaviour_session"] = behaviour_session
                dataset.attrs["offset_s"] = float(timestamps[0])
        return str(path)

    return _write


@pytest.fixture
def write_trials_csv():
    """Factory fixture that writes a minimal preprocessed trials.csv with just enough columns
    for `session.summary` (and therefore `subject.collate_sessions`) to run on."""

    def _write(directory, filename, animal_id, session_date, protocol, experiment, environment="unknown"):
        rows = [
            dict(outcome="correct", correction=False, sdt_type="hit", response_time=0.5, response="leverpush"),
            dict(
                outcome="incorrect", correction=False, sdt_type="false_alarm", response_time=0.3, response="leverpush"
            ),
            dict(
                outcome="correct", correction=False, sdt_type="correct_rejection", response_time=np.nan, response=None
            ),
            dict(outcome="incorrect", correction=True, sdt_type=None, response_time=np.nan, response=None),
            dict(outcome="precued", correction=False, sdt_type=None, response_time=np.nan, response=None),
        ]
        df = pd.DataFrame(rows)
        df["animal_id"] = animal_id
        df["session_date"] = session_date
        df["protocol"] = protocol
        df["experiment"] = experiment
        df["environment"] = environment

        path = os.path.join(directory, filename)
        df.to_csv(path, index=False)
        return path

    return _write


@pytest.fixture
def write_legacy_singletarget_json():
    """Factory fixture that writes a session JSON in the shape older Visiomode versions produced
    for the "singletarget" protocol (now "targetonly"): no `spec`, no per-trial `stimulus` or
    `response` objects, and SDT-style outcome labels ("miss") instead of "no_response". Every
    trial is a miss, so the session has no reaction times at all."""

    def _write(directory, num_trials=5, filename="sub-A1_exp-expX_ses-20210429_optoevents.json"):
        start = datetime.datetime(2021, 4, 29, 16, 16, 33)
        trials = [
            {
                "outcome": "miss",
                "iti": 10.0,
                "response_time": -1,
                "duration": -1,
                "pos_x": -1,
                "pos_y": -1,
                "dist_x": -1,
                "dist_y": -1,
                "timestamp": (start + datetime.timedelta(seconds=2 + 14 * i)).isoformat(),
                "correction": False,
            }
            for i in range(num_trials)
        ]
        data = {
            "animal_id": "A1",
            "experiment": "expX",
            "duration": 20.0,
            "protocol": "singletarget",
            "complete": True,
            "timestamp": start.isoformat(),
            "notes": "",
            "device": "rig-1",
            "trials": trials,
        }
        path = pathlib.Path(directory) / filename
        path.write_text(json.dumps(data))
        return str(path)

    return _write


@pytest.fixture
def flatten_trial():
    """Factory fixture that runs a single trial dict through the private
    `session._flatten_trials` generator, with sensible default session metadata that individual
    tests can override via `metadata_overrides`."""

    def _flatten(trial, metadata_overrides=None, session_start_time="2022-01-01T00:00:00"):
        metadata = {
            "session_start_time": session_start_time,
            "stimulus_duration": 4000,
            "environment": "unknown",
            "protocol": "gonogo",
            "stimuli": {
                "target_id": "movinggrating",
                "target_contrast": "1.0",
                "distractor_id": "isoluminantgray",
            },
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return next(iter(session._flatten_trials({"trials": [trial]}, metadata=metadata)))

    return _flatten
