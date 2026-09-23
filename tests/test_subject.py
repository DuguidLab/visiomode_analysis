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


# -- Excluded sessions (marked with a suffix after .csv, e.g. `..._trials.csv.ignore`) --


def test_collate_sessions_lists_excluded_sessions_without_metrics(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.ignore", "A1", "2022-01-02", "gonogo", "expX", environment="rig2")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["excluded"]) == [False, True]
    excluded = subject_df.set_index("session_date").loc["2022-01-02"]
    # Metadata is kept...
    assert excluded["animal_id"] == "A1"
    assert excluded["protocol"] == "gonogo"
    assert excluded["environment"] == "rig2"
    assert excluded["experiment"] == "expX"
    # ...but every metric is left empty.
    metric_columns = [c for c in subject_df.columns if c not in (*subject.METADATA_FIELDS, "excluded")]
    metric_columns = [c for c in metric_columns if c not in ("session_id", "task_session")]
    assert metric_columns
    assert excluded[metric_columns].isna().all()


def test_collate_sessions_does_not_summarise_excluded_sessions(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    # Metadata only, no trial columns: `session.summary` would fail on this file.
    pd.DataFrame(
        [dict(animal_id="A1", session_date="2022-01-02", protocol="gonogo", environment="unknown", experiment="expX")]
    ).to_csv(tmp_path / "b_trials.csv.exclude", index=False)

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["excluded"]) == [False, True]


def test_collate_sessions_counts_excluded_sessions_towards_task_session(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.ignore", "A1", "2022-01-02", "gonogo", "expX")
    write_trials_csv(tmp_path, "c_trials.csv", "A1", "2022-01-03", "gonogo", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["session_id"]) == [1, 2, 3]
    assert list(subject_df["task_session"]) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("suffix", [".ignore", ".ignored", ".exclude", ".excluded", ".skip", ".IGNORE", ".Skip"])
def test_collate_sessions_recognises_exclude_suffixes(tmp_path, write_trials_csv, suffix):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, f"b_trials.csv{suffix}", "A1", "2022-01-02", "gonogo", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["excluded"]) == [False, True]


def test_collate_sessions_skips_files_with_other_suffixes(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.bak", "A1", "2022-01-02", "gonogo", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None)

    assert list(subject_df["session_date"]) == ["2022-01-01"]
    assert list(subject_df["excluded"]) == [False]


def test_collate_sessions_drops_excluded_sessions_when_not_ignoring(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.ignore", "A1", "2022-01-02", "gonogo", "expX")
    write_trials_csv(tmp_path, "c_trials.csv", "A1", "2022-01-03", "gonogo", "expX")

    subject_df = subject.collate_sessions(directory=str(tmp_path), output_dir=None, ignore=False)

    assert list(subject_df["session_date"]) == ["2022-01-01", "2022-01-03"]
    assert list(subject_df["session_id"]) == [1, 2]
    assert list(subject_df["task_session"]) == [1.0, 2.0]
    assert list(subject_df["excluded"]) == [False, False]


def test_collate_sessions_raises_when_all_sessions_are_excluded_and_not_ignoring(tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv.ignore", "A1", "2022-01-01", "gonogo", "expX")

    with pytest.raises(FileNotFoundError, match="not marked as excluded"):
        subject.collate_sessions(directory=str(tmp_path), output_dir=None, ignore=False)
