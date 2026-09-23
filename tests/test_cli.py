import os

import pandas as pd
import pytest
from click.testing import CliRunner

from visiomode_analysis import cli, group, session, subject
from visiomode_analysis.__about__ import __version__


@pytest.fixture
def runner():
    return CliRunner()


# -- `visiomode-analysis session` --


def test_session_cmd_writes_trials_csv_and_report(runner, gonogo_session_json_path, tmp_path):
    result = runner.invoke(session.session_cmd, [gonogo_session_json_path, "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Files saved under {tmp_path}" in result.output

    written = os.listdir(tmp_path)
    assert any(name.endswith("_trials.csv") for name in written)
    assert any(name.endswith("_report-session.html") for name in written)


def test_session_cmd_handles_an_all_miss_legacy_singletarget_session(runner, write_legacy_singletarget_json, tmp_path):
    path = write_legacy_singletarget_json(tmp_path)

    result = runner.invoke(session.session_cmd, [path, "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    written = os.listdir(tmp_path)
    assert any(name.endswith("_behaviour-singletarget_trials.csv") for name in written)
    assert any(name.endswith("_behaviour-singletarget_report-session.html") for name in written)


def test_session_cmd_no_report_skips_report_generation(runner, gonogo_session_json_path, tmp_path):
    result = runner.invoke(session.session_cmd, [gonogo_session_json_path, "-o", str(tmp_path), "--no-report"])

    assert result.exit_code == 0, result.output

    written = os.listdir(tmp_path)
    assert any(name.endswith("_trials.csv") for name in written)
    assert not any(name.endswith("_report-session.html") for name in written)


def test_session_cmd_with_regressors_requires_timestamps_option(runner, gonogo_session_json_path, tmp_path):
    result = runner.invoke(session.session_cmd, [gonogo_session_json_path, "-o", str(tmp_path), "--with-regressors"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)


def test_session_cmd_with_regressors_writes_regressors_file(
    runner, gonogo_session_json_path, gonogo_regressor_timestamps_csv_path, tmp_path
):
    result = runner.invoke(
        session.session_cmd,
        [
            gonogo_session_json_path,
            "-o",
            str(tmp_path),
            "--with-regressors",
            "--regressor-timestamps",
            gonogo_regressor_timestamps_csv_path,
        ],
    )

    assert result.exit_code == 0, result.output
    assert any(name.endswith("_regressors.npz") for name in os.listdir(tmp_path))


def test_session_cmd_rejects_a_path_that_does_not_exist(runner, tmp_path):
    result = runner.invoke(session.session_cmd, [str(tmp_path / "missing.json")])

    assert result.exit_code == 2
    assert "does not exist" in result.output


# -- `visiomode-analysis regressors` --


def test_regressors_cmd_writes_regressors_file(
    runner, gonogo_session_json_path, gonogo_regressor_timestamps_csv_path, tmp_path
):
    result = runner.invoke(
        session.regressors_cmd,
        [
            gonogo_session_json_path,
            "-o",
            str(tmp_path),
            "--regressor-timestamps",
            gonogo_regressor_timestamps_csv_path,
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Regressors saved under {tmp_path}" in result.output
    assert any(name.endswith("_regressors.npz") for name in os.listdir(tmp_path))


def test_regressors_cmd_accepts_h5_timestamps(runner, gonogo_session_json_path, write_aligned_h5, tmp_path):
    h5_path = write_aligned_h5(tmp_path / "aligned.h5")

    result = runner.invoke(
        session.regressors_cmd, [gonogo_session_json_path, "-o", str(tmp_path), "--regressor-timestamps", h5_path]
    )

    assert result.exit_code == 0, result.output
    assert any(name.endswith("_regressors.npz") for name in os.listdir(tmp_path))


def test_session_cmd_with_regressors_accepts_h5_timestamps(runner, gonogo_session_json_path, write_aligned_h5, tmp_path):
    h5_path = write_aligned_h5(tmp_path / "aligned.h5")

    result = runner.invoke(
        session.session_cmd,
        [gonogo_session_json_path, "-o", str(tmp_path), "--no-report", "--with-regressors", "--regressor-timestamps", h5_path],
    )

    assert result.exit_code == 0, result.output
    assert any(name.endswith("_regressors.npz") for name in os.listdir(tmp_path))


def test_regressors_cmd_requires_regressor_timestamps_option(runner, gonogo_session_json_path):
    result = runner.invoke(session.regressors_cmd, [gonogo_session_json_path])

    assert result.exit_code == 2
    assert "regressor-timestamps" in result.output.lower()


# -- `visiomode-analysis session-start-time` --


def test_session_start_time_cmd_prints_only_the_timestamp(runner, gonogo_session_json_path):
    result = runner.invoke(session.session_start_time_cmd, [gonogo_session_json_path])

    assert result.exit_code == 0, result.output
    assert result.output == session.get_metadata(gonogo_session_json_path)["session_start_time"] + "\n"


# -- `visiomode-analysis subject` --


def test_subject_cmd_writes_summary_csv(runner, tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")

    result = runner.invoke(subject.subject_cmd, [str(tmp_path), "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert f"Files saved under {tmp_path}" in result.output
    assert (tmp_path / "sub-A1_exp-expX_behaviour-summary.csv").exists()


def test_subject_cmd_lists_excluded_sessions_by_default(runner, tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.ignore", "A1", "2022-01-02", "gonogo", "expX")

    result = runner.invoke(subject.subject_cmd, [str(tmp_path), "-o", str(tmp_path)])

    assert result.exit_code == 0, result.output
    written = pd.read_csv(tmp_path / "sub-A1_exp-expX_behaviour-summary.csv", index_col=0)
    assert list(written["excluded"]) == [False, True]


def test_subject_cmd_no_ignore_drops_excluded_sessions(runner, tmp_path, write_trials_csv):
    write_trials_csv(tmp_path, "a_trials.csv", "A1", "2022-01-01", "gonogo", "expX")
    write_trials_csv(tmp_path, "b_trials.csv.ignore", "A1", "2022-01-02", "gonogo", "expX")

    result = runner.invoke(subject.subject_cmd, [str(tmp_path), "-o", str(tmp_path), "--no-ignore"])

    assert result.exit_code == 0, result.output
    written = pd.read_csv(tmp_path / "sub-A1_exp-expX_behaviour-summary.csv", index_col=0)
    assert list(written["session_date"]) == ["2022-01-01"]
    assert list(written["excluded"]) == [False]


def test_subject_cmd_raises_for_a_directory_with_no_trials_csv(runner, tmp_path):
    result = runner.invoke(subject.subject_cmd, [str(tmp_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)


# -- `visiomode-analysis group` (currently an unimplemented stub) --


def test_group_cmd_is_a_no_op(runner):
    result = runner.invoke(group.group_cmd, [])

    assert result.exit_code == 0
    assert result.output == ""


# -- Top-level `cli` group wiring --


def test_cli_reports_its_version(runner):
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_lists_all_subcommands(runner):
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    for command_name in ("session", "regressors", "session-start-time", "subject", "group"):
        assert command_name in result.output


def test_cli_dispatches_to_session_subcommand(runner, gonogo_session_json_path, tmp_path):
    result = runner.invoke(cli, ["session", gonogo_session_json_path, "-o", str(tmp_path), "--no-report"])

    assert result.exit_code == 0, result.output
    assert any(name.endswith("_trials.csv") for name in os.listdir(tmp_path))
