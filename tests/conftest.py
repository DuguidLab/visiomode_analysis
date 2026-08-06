import datetime
import os
import pathlib

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
