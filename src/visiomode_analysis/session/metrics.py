import pandas as pd

from scipy.stats import norm


def d_prime(df, session_type):
    """Function to calculate d' from a session dataframe.

    Args: 
        df: Pandas dataframe with session data.
        session_type: Go/NoGo or 2AFC
    
    Returns:
        d_prime: Float representing the session d'.

    Example:
        dprime = d_prime(df, session_type="gonogo")
    """
    # Calculate the hit rate
    hits = len(
        df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)]
    )
    misses = len(
        df[(df.outcome == "incorrect") & (df.response.isnull()) & (df.correction == False)]
    )
    hit_rate = (hits + 0.5) / (hits + misses + 1.0)

    # Calculate FA rate
    false_alarms = len(
        df[(df.outcome == "incorrect") & (df.response.notnull()) & (df.correction == False)]
    )

    correct_reject = len(
        df[(df.outcome == "correct") & (df.response.isnull()) & (df.correction == False)]
    )

    fa_rate = (false_alarms + 0.5) / (false_alarms + correct_reject + 1)

    # Calculate d'
    # TODO: Switch d' calculation if task is 2afc
    if session_type == "gonogo":
        d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    else:
        d_prime = (1/sqrt(2))*(norm.ppf(hit_rate) - norm.ppf(fa_rate))

    # Return d' 
    return d_prime

# Calculate mean RT for all trial types
mean_rt_hits = df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)]["response_time"].mean()

mean_rt_false_alarms = df[(df.outcome == "incorrect") & (df.response.notnull()) & (df.correction == False)]["response_time"].mean()

# Calculate median RT for all trial types
median_rt_hits = df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)]["response_time"].median()

median_rt_false_alarms = df[(df.outcome == "incorrect") & (df.response.notnull()) & (df.correction == False)]["response_time"].median()
