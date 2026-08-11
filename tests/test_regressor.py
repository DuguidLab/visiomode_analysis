import numpy as np
import pandas as pd
import pytest

from visiomode_analysis.session import regressor as rgr


@pytest.fixture
def gonogo_trials() -> pd.DataFrame:
    """Three hand-crafted Go/NoGo trials covering a hit, a false alarm, and a correct rejection.

    Trial 0: hit on the "go" stimulus, lever pushed at RT=0.5s (sets the median RT used for
        the "hold" and "nogo" windows below).
    Trial 1: false alarm on the "nogo" stimulus, lever pushed at RT=0.3s.
    Trial 2: correct rejection on the "nogo" stimulus, no lever push.
    """
    return pd.DataFrame(
        [
            dict(
                start_time=0.0,
                stop_time=5.0,
                cue_onset=1.0,
                stim_id="movinggrating",
                outcome="correct",
                response="leverpush",
                sdt_type="hit",
                response_time=0.5,
            ),
            dict(
                start_time=5.0,
                stop_time=10.0,
                cue_onset=6.0,
                stim_id="isoluminantgray",
                outcome="incorrect",
                response="leverpush",
                sdt_type="false_alarm",
                response_time=0.3,
            ),
            dict(
                start_time=10.0,
                stop_time=15.0,
                cue_onset=11.0,
                stim_id="isoluminantgray",
                outcome="correct",
                response=None,
                sdt_type="correct_rejection",
                response_time=np.nan,
            ),
        ]
    )


def test_generate_gonogo_regressors_flags_expected_events(gonogo_trials):
    # leverpush_rt = median hit response_time = 0.5s, so windows below are relative to that.
    timestamps = [
        0.5,  # trial 0, before cue onset -> nothing
        1.5,  # trial 0, during the go stimulus -> stim_go
        4.95,  # trial 0, in the final 0.08s before stop -> cued lever push
        5.2,  # trial 1, ITI following a correct trial -> reward
        6.3,  # trial 1, within the nogo/leverpush-RT window -> stim_nogo (and still within reward window)
        9.95,  # trial 1, in the final 0.08s before stop -> cued lever push
        11.45,  # trial 2, in the expected-push hold window, but no push -> resp_hold
        20.0,  # outside every trial window -> excluded entirely
    ]

    regressors, labels, trial_entries = rgr.generate_gonogo_regressors(
        gonogo_trials, timestamps, trial_epoch_only=True
    )

    assert labels == {
        0: "stim_go",
        1: "stim_nogo",
        2: "resp_cuedpush",
        3: "resp_uncuedpush",
        4: "resp_hold",
        5: "reward",
    }

    # The last timestamp falls after every trial's stop_time, so it's dropped from the entries.
    np.testing.assert_array_equal(trial_entries, [True, True, True, True, True, True, True, False])

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(regressors, expected)


def test_generate_gonogo_regressors_pads_zero_rows_outside_trial_epochs(gonogo_trials):
    timestamps = [0.5, 20.0]

    regressors, _, trial_entries = rgr.generate_gonogo_regressors(gonogo_trials, timestamps, trial_epoch_only=False)

    # Unlike trial_epoch_only=True, every input timestamp gets a row, zero-filled outside trials.
    assert regressors.shape == (len(timestamps), 6)
    np.testing.assert_array_equal(regressors[1], [0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(trial_entries, [True, False])


def test_generate_gonogo_regressors_raises_when_no_timestamp_falls_in_a_trial(gonogo_trials):
    with pytest.raises(ValueError, match="No trials found"):
        rgr.generate_gonogo_regressors(gonogo_trials, [100.0, 200.0])
