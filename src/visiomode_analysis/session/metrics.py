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

def rt_metric(df, trial_type, metric_type):
    '''Defines a function to calculate mean or median RT for different trial types
    
    Args:
        df: Pandas dataframe with session data.
        trial_type: hits or false_alarms or both (default is both)
        metric type: mean or median (default is median)
    
    Output: 
        float representing the mean/median RT for a all hit/FA trials in a session
    
    Example: 
        hit_mean_RT = rt_metric(df, trial_type='hit', metric_type='mean')'''
    

    if trial_type == "hits":
        if metric_type == "mean":
            return df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)]["response_time"].mean()
        else:
            return df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)]["response_time"].median()
    elif trial_type == "false_alarms":
        if metric_type == "mean":
            return df[(df.outcome == "incorrect") & (df.response.notnull()) & (df.correction == False)]["response_time"].mean()
        else:
            return df[(df.outcome == "incorrect") & (df.response.notnull()) & (df.correction == False)]["response_time"].median()
    else:
        if metric_type == "mean":
            return df[(df.response.notnull()) & (df.correction == False)]["response_time"].mean()
        else:
            return df[(df.response.notnull()) & (df.correction == False)]["response_time"].median()

def trial_count(df, trial_type="all", correction=False, disregard_correction=True):
    ''' Defines a function to return the number of trials, with the option of returning the number of trials for a particular trial type
    
    Args:
        df: dataframe containing session data
        trial_type: all, hits, misses, false_alarms, correct_rejections, cued, precued. Default output is total number of trials.
        correction: "True" or "False" to specify correction or random trials, respectively. Default is False (i.e. return only random trials).
        disregard_correction: Return trial count irrespective of whether they were correction trials. This will override the value of `correction` if True. Default is True
        
    Output:
        float representing the number of trials for the specified trial type.
    
    Example:
        no_of_hits = trial_count(df, hits)'''

    if not disregard_correction:
        df = df[df.correction == correction]

    if trial_type == "all":
        return df.outcome.count()
    if trial_type == "hits":
        return df[(df.outcome == "correct") & (df.response.notnull())].outcome.count()
    if trial_type == "misses":
        return df[(df.outcome == "incorrect") & (df.response.isnull())].outcome.count()
    if trial_type == "false_alarms":
        return df[(df.outcome == "correct") & (df.response.notnull())]
    if trial_type == "correct_rejections":
        return df[(df.outcome == "correct") & df(df.response.isnull())].outcome.count()
    if trial_type == "cued":
        return df[(df.outcome != "precued")].outcome.count()
    if trial_type == "precued":
        return df[(df.outcome == "precued")].outcome.count()
    else:
        raise ValueError(f"Invalid trial type {trial_type}")
        
