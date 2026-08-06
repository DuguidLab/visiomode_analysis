import os

import pandas as pd
import pytest

from visiomode_analysis import subject


def test_collate_sessions_raises_when_no_trials_csv_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trials.csv files found"):
        subject.collate_sessions(directory=str(tmp_path), output_dir=None)


def test_collate_sessions_orders_by_date_and_ranks_task_sessions_per_protocol(tmp_path, write_trials_csv):
    # Sessions are written out of date order, and span two protocols, to exercise both the
    # chronological sort and the per-(animal, protocol) task_session ranking.
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-03", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "c_trials.csv", "A1", "2022-01-02", "targetonly", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["session_date"]) == ["2022-01-01", "2022-01-02", "2022-01-03"]
    assert list(subject_df["session_id"]) == [1, 2, 3]

    by_date = subject_df.set_index("session_date")
    # The two gonogo sessions are the 1st and 2nd gonogo sessions chronologically...
    assert by_date.loc["2022-01-01", "task_session"] == 1.0
    assert by_date.loc["2022-01-03", "task_session"] == 2.0
    # ...while the lone targetonly session is the 1st (and only) session of its protocol.
    assert by_date.loc["2022-01-02", "task_session"] == 1.0


def test_collate_sessions_does_not_write_csv_when_output_dir_is_falsy(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")

    subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert os.listdir(tmp_path) == ["a_trials.csv"]


def test_collate_sessions_writes_summary_csv_with_expected_name(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=str(tmp_path))

    out_path = tmp_path / "sub-A1_exp-expX_behaviour-summary.csv"
    assert out_path.exists()

    written = pd.read_csv(out_path, index_col=0)
    assert list(written["animal_id"]) == list(subject_df["animal_id"])
    assert list(written["session_id"]) == list(subject_df["session_id"])


def test_preprocess_subject_returns_output_dir_and_writes_summary(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")

    out_dir = subject.preprocess_subject(directory=str(tmp_path), output_dir=str(tmp_path))

    assert out_dir == str(tmp_path)
    assert (tmp_path / "sub-A1_exp-expX_behaviour-summary.csv").exists()
