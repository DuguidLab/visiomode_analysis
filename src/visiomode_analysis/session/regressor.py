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

import numpy as np
import pandas as pd


def generate_gonogo_regressors(
    trials_df: pd.DataFrame,
    timestamps: np.ndarray | list,
    go_stim_id: str = "movinggrating",
    nogo_stim_id: str = "isoluminantgray",
    uncued_push_id: str = "uncued",
    correct_outcome_id: str = "correct",
    lever_push_duration: float = 0.07,
    stimulus_duration: float = 0.15,
    reward_duration: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Generate regressors for the Go/NoGo paradigm for a custom set of timestamps.

    Args:
        trials (pd.DataFrame): A DataFrame containing trial data with columns for trial type,
            start time, and stop time.
        timestamps (np.ndarray | list): An array or list of timestamps at which to evaluate the
            regressors. This would typically correspond to the timestamps of an imaging session or other continuous recording, relative to the start time of the behavioural session. Timestamps should be in seconds and aligned to the start of the behaviour.
        go_stim_id (str, optional): The identifier for the Go stimulus. Defaults to "movinggrating".
        nogo_stim_id (str, optional): The identifier for the No-Go stimulus.
            Defaults to "isoluminantgray".
        uncued_push_id (str, optional): The identifier for uncued push responses. Defaults to "uncued".
        correct_outcome_id (str, optional): The identifier for correct trial outcomes. Defaults to "correct".
        lever_push_duration (float, optional): The duration of a lever push in seconds. Defaults to 0.07,  which is lever push duration from Dacre et al. 2021.
        stimulus_duration (float, optional): The duration of the stimulus in seconds. Defaults to 0.15.
        reward_duration (float, optional): The duration of the reward in seconds. Defaults to 1.5.

    Returns:
        np.ndarray: A 2D array where each row corresponds to a timestamp and each column
            corresponds to a regressor (e.g., stimulus, response, reward).
        dict: A dictionary mapping regressor names to their corresponding column indices in the
            output array.
    """
    timestamps = np.asarray(timestamps, dtype=float)

    response_times = np.array(trials_df[trials_df.response.notna()].response_time.values)
    leverpush_rt = np.nanmedian(response_times)

    start_time = trials_df["start_time"].to_numpy(dtype=float)
    stop_time = trials_df["stop_time"].to_numpy(dtype=float)
    cue_onset = trials_df["cue_onset"].to_numpy(dtype=float)
    stim_id = trials_df["stim_id"].to_numpy()
    outcome = trials_df["outcome"].to_numpy()
    leverpush = trials_df["response"].notna().to_numpy()

    # For every timestamp find the index of the trial that contains it (start_time < ts < stop_time).
    ts_indexes = np.searchsorted(start_time, timestamps, side="right") - 1
    trial_entries = ts_indexes >= 0
    trial_entries[trial_entries] &= start_time[ts_indexes[trial_entries]] < timestamps[trial_entries]
    trial_entries[trial_entries] &= stop_time[ts_indexes[trial_entries]] > timestamps[trial_entries]

    trial_idx = np.nonzero(trial_entries)[0]

    if len(trial_idx) == 0:
        raise ValueError("No trials found for the provided timestamps.")

    ts = timestamps[trial_idx]
    trial_idx = ts_indexes[trial_idx]

    # Stimulus regressors
    regr_stim_go = (
        (stim_id[trial_idx] == go_stim_id)
        & (ts >= cue_onset[trial_idx])
        & (ts <= cue_onset[trial_idx] + stimulus_duration)
    ).astype(int)
    regr_stim_nogo = (
        (stim_id[trial_idx] == nogo_stim_id)
        & (ts >= cue_onset[trial_idx])
        & (ts <= cue_onset[trial_idx] + stimulus_duration)
    ).astype(int)

    # Response regressors
    push_window_start = cue_onset[trial_idx] + leverpush_rt - lever_push_duration
    push_window_end = cue_onset[trial_idx] + leverpush_rt
    regr_resp_cuedpush = (
        leverpush[trial_idx]
        & (outcome[trial_idx] != uncued_push_id)
        & (ts >= push_window_start)
        & (ts <= push_window_end)
    ).astype(int)
    regr_resp_uncuedpush = (
        (outcome[trial_idx] == uncued_push_id)
        & (ts >= stop_time[trial_idx] - lever_push_duration)
        & (ts <= stop_time[trial_idx])
    ).astype(int)
    regr_resp_hold = (~leverpush[trial_idx] & (ts >= push_window_start) & (ts <= push_window_end)).astype(int)

    # Reward regressor, depends on previous trial
    has_previous = trial_idx > 0
    previous_outcome = np.where(has_previous, outcome[np.maximum(trial_idx - 1, 0)], None)  # type: ignore
    regr_reward = (
        has_previous & (previous_outcome == correct_outcome_id) & (ts <= start_time[trial_idx] + reward_duration)
    ).astype(int)

    regressors = np.stack(
        [
            regr_stim_go,
            regr_stim_nogo,
            regr_resp_cuedpush,
            regr_resp_uncuedpush,
            regr_resp_hold,
            regr_reward,
        ],
        axis=1,
    )

    regressor_names = {
        0: "stim_go",
        1: "stim_nogo",
        2: "resp_cuedpush",
        3: "resp_uncuedpush",
        4: "resp_hold",
        5: "reward",
    }

    return regressors, regressor_names
