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

def trial_count(df, trial_type, correction):
    ''' Defines a function to return the number of trials, with the option of returning the number of trials for a particular trial type
    
    Args:
        df: dataframe containing session data
        trial_type: hits, misses, false_alarms, correct_rejections. Default output is total number of trials.
        correction_type: "True" or "False" to specify correction or random trials, respectively. Default is both.
        
    Output:
        float representing the number of trials for the specified trial type.
    
    Example:
        no_of_hits = trial_count(df, hits)'''
    
    if trial_type == "hits":
        if correction == True:
            return df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == True)].outcome.count()
        elif correction == False:
                return df[(df.outcome == "correct") & (df.response.notnull()) & (df.correction == False)].outcome.count()
        else:
            return df[(df.outcome == "correct") & (df.response.notnull())].outcome.count()
