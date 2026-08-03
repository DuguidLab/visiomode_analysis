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
    trials: pd.DataFrame,
    timestamps: np.ndarray | list,
    go_stim_id: str = "movinggrating",
    nogo_stim_id: str = "isoluminantgray",
    ucued_push_id: str = "precued",
    correct_outcome_id: str = "correct",
) -> tuple[np.ndarray, dict]:
    """Generate regressors for a Go/No-Go task based on trial data and metadata.

    Args:
        trials (pd.DataFrame): A DataFrame containing trial data with columns for trial type,
            start time, and stop time.
        timestamps (np.ndarray | list): An array or list of timestamps at which to evaluate the
            regressors. This would typically correspond to the timestamps of an imaging session or other continuous recording, relative to the start time of the behavioural session.
        go_stim_id (str, optional): The identifier for the Go stimulus. Defaults to "movinggrating".
        nogo_stim_id (str, optional): The identifier for the No-Go stimulus.
            Defaults to "isoluminantgray".
        ucued_push_id (str, optional): The identifier for uncued push responses. Defaults to "precued".
        correct_outcome_id (str, optional): The identifier for correct trial outcomes. Defaults to "correct".

    Returns:
        np.ndarray: A 2D array where each row corresponds to a timestamp and each column
            corresponds to a regressor (e.g., stimulus, response, reward).
        dict: A dictionary mapping regressor names to their corresponding column indices in the
            output array.
    """

    return np.array([]), dict()
