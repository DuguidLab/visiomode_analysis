import datetime
import json
import warnings
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from visiomode_analysis import session


def test_get_metadata_reads_spec_and_filename_derived_fields(gonogo_session_json_path):
    metadata = session.get_metadata(gonogo_session_json_path)

    assert metadata["protocol"] == "gonogo"
    assert metadata["response_device"] == "leverpush"
    assert metadata["reward_profile"] == "waterreward"
    assert metadata["stimulus_duration"] == 4000.0
    assert metadata["iti"] == 5000.0
    assert metadata["session_date"] == datetime.date(2022, 3, 9)
    assert metadata["stimuli"]["target_id"] == "movinggrating"
    assert metadata["stimuli"]["distractor_id"] == "isoluminantgray"


def test_get_metadata_prefers_bids_style_filename_over_json_contents(gonogo_session_json_path, tmp_path):
    # The example fixture's own filename has none of the sub-/exp-/ses-/behaviour- markers, so
    # get_metadata falls back to the JSON's own animal_id/experiment/environment fields for it.
    # Copying it to a BIDS-style name exercises the filename-parsing branches instead, which take
    # priority over those JSON fields.
    bids_path = tmp_path / "sub-ZZ99_exp-testexp_ses-20230115_behaviour-hf.json"
    shutil.copy(gonogo_session_json_path, bids_path)

    metadata = session.get_metadata(str(bids_path))

    assert metadata["animal_id"] == "ZZ99"
    assert metadata["experiment"] == "testexp"
    assert metadata["session_date"] == datetime.date(2023, 1, 15)
    assert metadata["environment"] == "hf"


def test_get_trials_returns_one_row_per_input_trial(gonogo_session_json_path):
    with open(gonogo_session_json_path) as fp:
        num_source_trials = len(json.load(fp)["trials"])

    trials = session.get_trials(gonogo_session_json_path)

    assert isinstance(trials, pd.DataFrame)
    assert len(trials) == num_source_trials


def test_get_trials_normalises_legacy_outcome_labels(gonogo_session_json_path):
    trials = session.get_trials(gonogo_session_json_path)

    # Legacy "hit"/"miss"/"false_alarm" outcomes are remapped to correct/incorrect/no_response,
    # while the finer-grained SDT classification survives separately in `sdt_type`.
    assert set(trials["outcome"].unique()) <= {"correct", "incorrect", "precued"}
    assert set(trials["sdt_type"].dropna().unique()) <= {"hit", "miss", "false_alarm", "correct_rejection"}


def test_get_rts_only_includes_cued_trials_with_a_response(gonogo_session_json_path):
    rts = session.get_rts(gonogo_session_json_path)

    assert isinstance(rts, np.ndarray)
    assert len(rts) > 0
    assert np.all(rts >= 0)


def test_summary_computes_consistent_trial_counts(gonogo_session_json_path):
    result = session.summary(gonogo_session_json_path)

    assert result["animal_id"] == "MM229"
    assert result["protocol"] == "gonogo"
    assert result["total"] == result["cued_wc"] + result["precued"]
    assert result["cued_wc"] == result["hits_wc"] + result["misses_wc"] + result["false_alarms_wc"] + result["correct_rejections_wc"]
    assert result["correct_wc"] == result["hits_wc"] + result["correct_rejections_wc"]

    # Rates are Hautus-corrected proportions, so they're always strictly between 0 and 1.
    assert 0.0 < result["hit_rate"] < 1.0
    assert 0.0 < result["fa_rate"] < 1.0


def test_summary_classifies_every_trial_of_an_all_miss_legacy_singletarget_session(
    write_legacy_singletarget_json, tmp_path
):
    path = write_legacy_singletarget_json(tmp_path, num_trials=7)

    result = session.summary(path)

    assert result["protocol"] == "singletarget"
    assert result["total"] == 7
    assert result["cued"] == 7
    assert result["misses"] == 7
    assert result["hits"] == result["false_alarms"] == result["correct_rejections"] == 0
    assert np.isnan(result["rt"])
    assert np.isnan(result["rt_iqr"])


def test_generate_report_succeeds_for_an_all_miss_legacy_singletarget_session(
    write_legacy_singletarget_json, tmp_path
):
    # Regression: this used to raise IndexError from np.percentile on the empty RT array.
    path = write_legacy_singletarget_json(tmp_path)

    report_path = session.generate_report(path, output_dir=str(tmp_path))

    assert report_path.endswith("_behaviour-singletarget_report-session.html")
    html = pathlib.Path(report_path).read_text(encoding="utf-8")
    # Target-only sessions have no distractor, so the SDT-only sections are omitted.
    assert "SDT metrics" not in html
    assert "SDT timeseries" not in html


def test_summary_reports_nan_iqr_when_no_trial_has_a_response(tmp_path):
    # The IQR is computed from trials with a non-null response; when there are none at all,
    # np.percentile raises on the empty selection and summary() falls back to NaN instead of
    # propagating the error.
    rows = [
        dict(outcome="correct", correction=False, sdt_type="hit", response_time=np.nan, response=None),
        dict(outcome="incorrect", correction=False, sdt_type="false_alarm", response_time=np.nan, response=None),
        dict(outcome="correct", correction=False, sdt_type="correct_rejection", response_time=np.nan, response=None),
        dict(outcome="precued", correction=False, sdt_type=None, response_time=np.nan, response=None),
    ]
    df = pd.DataFrame(rows)
    df["animal_id"] = "A1"
    df["session_date"] = "2022-01-01"
    df["protocol"] = "gonogo"
    df["experiment"] = "expX"
    df["environment"] = "unknown"

    csv_path = tmp_path / "trials.csv"
    df.to_csv(csv_path, index=False)

    result = session.summary(str(csv_path))

    assert np.isnan(result["rt_iqr"])
    assert np.isnan(result["rt_iqr_wc"])


def test_generate_regressors_accepts_txt_timestamps(
    gonogo_session_json_path, gonogo_regressor_timestamps_txt_path, tmp_path
):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)

    out_path = session.generate_regressors(trials_df, metadata, gonogo_regressor_timestamps_txt_path, output_dir=str(tmp_path))

    assert out_path.endswith("_regressors.npz")
    assert (tmp_path / "sub-MM229_exp-109hrb21d_ses-20220309_behaviour-gonogo_regressors.npz").exists()


def test_generate_regressors_txt_output_has_expected_keys(
    gonogo_session_json_path, gonogo_regressor_timestamps_txt_path, tmp_path
):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)

    out_path = session.generate_regressors(trials_df, metadata, gonogo_regressor_timestamps_txt_path, output_dir=str(tmp_path))

    with np.load(out_path) as npz:
        assert set(npz.files) == {
            "regressors",
            "labels",
            "timestamps",
            "trial_idx",
            "session_start_time",
            "behaviour_session",
        }
        assert str(npz["session_start_time"]) == metadata["session_start_time"]
        assert str(npz["behaviour_session"]) == "example-gonogo-leverpush"
        # TXT timestamps are still recalculated relative to behaviour start.
        expected = session.get_trials(gonogo_session_json_path)["cue_onset"].dropna().head(3) + 0.2
        np.testing.assert_allclose(npz["timestamps"], expected.to_numpy(), atol=1e-6)


def test_generate_regressors_h5_uses_aligned_timestamps_as_is(gonogo_session_json_path, write_aligned_h5, tmp_path):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    timestamps = (trials_df["cue_onset"].dropna().head(3) + 0.2).to_numpy()
    h5_path = write_aligned_h5(tmp_path / "aligned.h5", timestamps=timestamps)

    out_path = session.generate_regressors(trials_df, metadata, h5_path, output_dir=str(tmp_path))

    with np.load(out_path) as npz:
        np.testing.assert_array_equal(npz["timestamps"], timestamps)
        assert npz["timestamps"].dtype == np.float64
        assert str(npz["session_start_time"]) == metadata["session_start_time"]
        assert str(npz["behaviour_session"]) == metadata["behaviour_session"]
        assert npz["regressors"].shape[0] == len(timestamps)


def test_generate_regressors_h5_accepts_equivalent_iso_spellings(gonogo_session_json_path, write_aligned_h5, tmp_path):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    # Same instant, different ISO 8601 spelling (space separator instead of "T").
    respelled = datetime.datetime.fromisoformat(metadata["session_start_time"]).isoformat(sep=" ")
    assert respelled != metadata["session_start_time"]
    h5_path = write_aligned_h5(tmp_path / "aligned.h5", session_start_time=respelled)

    out_path = session.generate_regressors(trials_df, metadata, h5_path, output_dir=str(tmp_path))

    assert out_path.endswith("_regressors.npz")


def test_generate_regressors_h5_rejects_mismatched_session_start_time(
    gonogo_session_json_path, write_aligned_h5, tmp_path
):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    h5_path = write_aligned_h5(tmp_path / "aligned.h5", session_start_time="2000-01-01T00:00:00")

    with pytest.raises(ValueError, match="session_start_time mismatch"):
        session.generate_regressors(trials_df, metadata, h5_path, output_dir=str(tmp_path))


def test_generate_regressors_h5_rejects_missing_dataset(gonogo_session_json_path, write_aligned_h5, tmp_path):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    h5_path = write_aligned_h5(tmp_path / "aligned.h5", include_dataset=False)

    with pytest.raises(ValueError, match="timestamps_aligned"):
        session.generate_regressors(trials_df, metadata, h5_path, output_dir=str(tmp_path))


def test_generate_regressors_h5_rejects_missing_session_start_time_attr(gonogo_session_json_path, tmp_path):
    import h5py

    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    h5_path = tmp_path / "aligned.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("timestamps_aligned", data=np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="no session_start_time attribute"):
        session.generate_regressors(trials_df, metadata, str(h5_path), output_dir=str(tmp_path))


def test_generate_regressors_rejects_unsupported_timestamps_file_extension(gonogo_session_json_path, tmp_path):
    trials_df = session.get_trials(gonogo_session_json_path)
    metadata = session.get_metadata(gonogo_session_json_path)
    bad_path = tmp_path / "timestamps.xyz"
    bad_path.write_text("2022-01-01T00:00:01\n")

    with pytest.raises(ValueError, match="CSV, TXT or H5"):
        session.generate_regressors(trials_df, metadata, str(bad_path), output_dir=str(tmp_path))


def test_generate_regressors_rejects_unsupported_protocol(tmp_path):
    trials_df = pd.DataFrame(
        [dict(start_time=0.0, stop_time=1.0, cue_onset=0.5, stim_id="x", outcome="correct", response=None)]
    )
    metadata = {
        "protocol": "unsupported-protocol",
        "animal_id": "MM229",
        "experiment": "exp",
        "session_date": "20220101",
        "session_start_time": "2022-01-01T00:00:00",
    }
    timestamps_path = tmp_path / "timestamps.csv"
    timestamps_path.write_text("2022-01-01T00:00:01\n")

    with pytest.raises(NotImplementedError):
        session.generate_regressors(trials_df, metadata, str(timestamps_path), output_dir=str(tmp_path))


# -- Lever push durations (from a sibling `_lever-durations.csv`) --


def _lever_push_mask(trials):
    return (trials["outcome"] == "precued") | trials["sdt_type"].isin(["hit", "false_alarm"])


def test_find_lever_durations_pairs_bids_named_json_with_sibling_csv(gonogo_session_with_lever_durations):
    json_path, durations_path = gonogo_session_with_lever_durations

    assert session.find_lever_durations(json_path) == durations_path


def test_find_lever_durations_returns_none_without_a_sibling_csv(gonogo_session_json_path):
    assert session.find_lever_durations(gonogo_session_json_path) is None


def test_get_trials_assigns_lever_durations_to_push_trials_in_order(gonogo_session_with_lever_durations):
    json_path, _ = gonogo_session_with_lever_durations

    trials = session.get_trials(json_path)

    is_push = _lever_push_mask(trials)
    assert list(trials.loc[is_push, "lever_duration"]) == [100.0 + i for i in range(is_push.sum())]
    assert trials.loc[~is_push, "lever_duration"].isna().all()


def test_get_trials_omits_lever_duration_column_without_a_durations_file(gonogo_session_json_path):
    trials = session.get_trials(gonogo_session_json_path)

    assert "lever_duration" not in trials.columns


def test_get_trials_uses_explicit_lever_durations_path_over_sibling(gonogo_session_with_lever_durations, tmp_path):
    json_path, durations_path = gonogo_session_with_lever_durations
    durations = pd.read_csv(durations_path)
    override_path = tmp_path / "override.csv"
    durations.assign(duration=durations["duration"] + 1000).to_csv(override_path, index=False)

    trials = session.get_trials(json_path, lever_durations=str(override_path))

    assert trials["lever_duration"].min() == 1100.0


def test_get_trials_matches_lever_durations_by_push_id_not_file_order(gonogo_session_with_lever_durations):
    json_path, durations_path = gonogo_session_with_lever_durations
    durations = pd.read_csv(durations_path)
    durations.iloc[::-1].to_csv(durations_path, index=False)

    trials = session.get_trials(json_path)

    assert list(trials.loc[_lever_push_mask(trials), "lever_duration"]) == list(durations["duration"])


def test_get_trials_warns_and_leaves_lever_durations_empty_on_count_mismatch(gonogo_session_with_lever_durations):
    json_path, durations_path = gonogo_session_with_lever_durations
    pd.read_csv(durations_path).iloc[:-1].to_csv(durations_path, index=False)

    with pytest.warns(UserWarning, match="lever pushes"):
        trials = session.get_trials(json_path)

    assert "lever_duration" in trials.columns
    assert trials["lever_duration"].isna().all()


def _write_lever_duration_trials_csv(tmp_path, write_trials_csv):
    rows = [
        dict(outcome="correct", correction=False, sdt_type="hit", lever_duration=80.0),
        dict(outcome="correct", correction=False, sdt_type="hit", lever_duration=100.0),
        dict(outcome="correct", correction=True, sdt_type="hit", lever_duration=200.0),
        dict(outcome="incorrect", correction=False, sdt_type="false_alarm", lever_duration=120.0),
        dict(outcome="correct", correction=False, sdt_type="correct_rejection", lever_duration=np.nan),
        dict(outcome="precued", correction=False, sdt_type=None, lever_duration=60.0),
    ]
    for row in rows:
        row["response"] = "leverpush" if not np.isnan(row["lever_duration"]) else None
        row["response_time"] = 0.5 if row["sdt_type"] in ("hit", "false_alarm") else np.nan
    return write_trials_csv(tmp_path, "trials.csv", "A1", "2022-01-01", "gonogo", "expX", rows=rows)


def _iqr(values):
    return np.percentile(values, 75) - np.percentile(values, 25)


@pytest.mark.parametrize(
    "subset, durations, durations_wc",
    [
        ("", [80.0, 100.0, 120.0, 60.0], [80.0, 100.0, 200.0, 120.0, 60.0]),
        ("_cued", [80.0, 100.0, 120.0], [80.0, 100.0, 200.0, 120.0]),
        ("_hits", [80.0, 100.0], [80.0, 100.0, 200.0]),
        ("_false_alarms", [120.0], [120.0]),
        ("_precued", [60.0], [60.0]),
    ],
)
def test_summary_reports_lever_duration_stats_per_subset(tmp_path, write_trials_csv, subset, durations, durations_wc):
    result = session.summary(_write_lever_duration_trials_csv(tmp_path, write_trials_csv))

    for suffix, values in (("", durations), ("_wc", durations_wc)):
        assert result[f"lever_duration_mean{subset}{suffix}"] == pytest.approx(np.mean(values))
        assert result[f"lever_duration_median{subset}{suffix}"] == pytest.approx(np.median(values))
        assert result[f"lever_duration_iqr{subset}{suffix}"] == pytest.approx(_iqr(values))
        if len(values) > 1:
            assert result[f"lever_duration_sd{subset}{suffix}"] == pytest.approx(np.std(values, ddof=1))
        else:
            assert np.isnan(result[f"lever_duration_sd{subset}{suffix}"])


def test_summary_reports_nan_lever_duration_stats_for_empty_subsets(tmp_path, write_trials_csv):
    csv_path = _write_lever_duration_trials_csv(tmp_path, write_trials_csv)
    df = pd.read_csv(csv_path)
    df[df["sdt_type"] != "false_alarm"].to_csv(csv_path, index=False)

    result = session.summary(csv_path)

    for stat in session.LEVER_DURATION_STATS:
        assert np.isnan(result[f"lever_duration_{stat}_false_alarms"])
        assert np.isnan(result[f"lever_duration_{stat}_false_alarms_wc"])


def test_summary_has_forty_lever_duration_fields_when_durations_are_present(gonogo_session_with_lever_durations):
    json_path, _ = gonogo_session_with_lever_durations

    result = session.summary(json_path)

    # 4 stats x 5 subsets x with/without corrections.
    assert len([key for key in result if key.startswith("lever_duration_")]) == 40


def test_summary_omits_lever_duration_fields_without_durations(gonogo_session_json_path):
    result = session.summary(gonogo_session_json_path)

    assert not any(key.startswith("lever_duration") for key in result)


def test_generate_report_includes_lever_duration_plot_only_when_durations_are_present(
    gonogo_session_with_lever_durations, gonogo_session_json_path, tmp_path
):
    json_path, _ = gonogo_session_with_lever_durations

    with_durations = pathlib.Path(session.generate_report(json_path, output_dir=str(tmp_path)))
    without_durations = pathlib.Path(session.generate_report(gonogo_session_json_path, output_dir=str(tmp_path)))

    assert "Median lever push duration" in with_durations.read_text(encoding="utf-8")
    assert "Median lever push duration" not in without_durations.read_text(encoding="utf-8")


def test_get_trials_warns_and_leaves_lever_durations_empty_without_expected_columns(
    gonogo_session_with_lever_durations,
):
    json_path, durations_path = gonogo_session_with_lever_durations
    pd.read_csv(durations_path).rename(columns={"push_id": "id"}).to_csv(durations_path, index=False)

    with pytest.warns(UserWarning, match="missing column.*push_id"):
        trials = session.get_trials(json_path)

    assert trials["lever_duration"].isna().all()


@pytest.mark.parametrize("push_ids", ["duplicated", "gapped"])
def test_get_trials_warns_and_leaves_lever_durations_empty_on_bad_push_ids(
    gonogo_session_with_lever_durations, push_ids
):
    json_path, durations_path = gonogo_session_with_lever_durations
    durations = pd.read_csv(durations_path)
    # Same number of pushes as lever push trials, but the IDs can't be trusted to line up with them.
    durations.loc[1, "push_id"] = 0 if push_ids == "duplicated" else len(durations)
    durations.to_csv(durations_path, index=False)

    with pytest.warns(UserWarning, match="push_id"):
        trials = session.get_trials(json_path)

    assert trials["lever_duration"].isna().all()


def test_summary_accepts_a_trials_dataframe(gonogo_session_with_lever_durations):
    json_path, _ = gonogo_session_with_lever_durations

    from_dataframe = session.summary(session.get_trials(json_path))
    from_json = session.summary(json_path)

    assert from_dataframe.keys() == from_json.keys()
    for key, value in from_json.items():
        # NaN != NaN, so missing values have to be compared separately.
        assert from_dataframe[key] == value or (pd.isna(from_dataframe[key]) and pd.isna(value)), key


def test_preprocess_session_uses_lever_durations_override_throughout(gonogo_session_with_lever_durations, tmp_path):
    json_path, durations_path = gonogo_session_with_lever_durations
    override_path = tmp_path / "override.csv"
    pd.read_csv(durations_path).to_csv(override_path, index=False)
    # A stale sibling that would warn if anything fell back to auto-detection.
    pd.read_csv(durations_path).iloc[:-1].to_csv(durations_path, index=False)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        session.preprocess_session(json_path, output_dir=str(out_dir), lever_durations=str(override_path))

    (report_path,) = out_dir.glob("*_report-session.html")
    assert "Median lever push duration" in report_path.read_text(encoding="utf-8")


def test_generate_report_omits_lever_duration_plot_when_durations_could_not_be_matched(
    gonogo_session_with_lever_durations, tmp_path
):
    json_path, durations_path = gonogo_session_with_lever_durations
    pd.read_csv(durations_path).iloc[:-1].to_csv(durations_path, index=False)

    with pytest.warns(UserWarning):
        report_path = pathlib.Path(session.generate_report(json_path, output_dir=str(tmp_path)))

    assert "Median lever push duration" not in report_path.read_text(encoding="utf-8")


def test_get_trials_warns_and_leaves_lever_durations_empty_on_empty_file(gonogo_session_with_lever_durations):
    json_path, durations_path = gonogo_session_with_lever_durations
    pathlib.Path(durations_path).write_text("")

    with pytest.warns(UserWarning, match="Can't match"):
        trials = session.get_trials(json_path)

    assert trials["lever_duration"].isna().all()


def test_get_trials_warns_and_leaves_lever_durations_empty_on_non_numeric_duration(
    gonogo_session_with_lever_durations,
):
    json_path, durations_path = gonogo_session_with_lever_durations
    durations = pd.read_csv(durations_path)
    # Right number of pushes with valid IDs, so the non-numeric value is the only problem.
    durations["duration"] = durations["duration"].astype(object)
    durations.loc[1, "duration"] = "not-a-number"
    durations.to_csv(durations_path, index=False)

    with pytest.warns(UserWarning, match="not-a-number"):
        trials = session.get_trials(json_path)

    assert trials["lever_duration"].isna().all()


def test_summary_reads_metadata_from_a_filtered_trials_dataframe(gonogo_session_json_path):
    trials = session.get_trials(gonogo_session_json_path)
    # Dropping the first trial leaves an index that no longer starts at 0.
    result = session.summary(trials.iloc[1:])

    assert result["animal_id"] == "MM229"
    assert result["protocol"] == "gonogo"


def test_summary_ignores_none_responses_in_trials_csvs_written_by_older_versions(tmp_path, write_trials_csv):
    # Trials CSVs written before 0.4.1 kept Visiomode's "none" response with the timeout as its response time.
    rows = [
        dict(outcome="correct", correction=False, sdt_type="hit", response_time=0.5, response="leverpush"),
        dict(outcome="incorrect", correction=False, sdt_type="false_alarm", response_time=0.7, response="leverpush"),
        dict(outcome="no_response", correction=False, sdt_type="miss", response_time=10.0, response="none"),
        dict(outcome="correct", correction=False, sdt_type="correct_rejection", response_time=4.0, response="none"),
    ]
    csv_path = write_trials_csv(tmp_path, "trials.csv", "A1", "2022-01-01", "gonogo", "expX", rows=rows)

    result = session.summary(csv_path)

    assert result["rt"] == pytest.approx(0.6)
    assert result["rt_wc"] == pytest.approx(0.6)
    assert result["rt_iqr"] == pytest.approx(0.1)
    assert result["misses"] == 1 and result["correct_rejections"] == 1
