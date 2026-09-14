import datetime
import json
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
