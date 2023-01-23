#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of basic score functions that applies to a point prediction or a single prediction interval.

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import numpy as np

def interval_score(observation, lower, upper, interval_range, specify_range_out=False):
    """interval_score.

    Parameters
    ----------
    observation : array_like
        Vector of observations.
    lower : array_like
        Prediction for the lower quantile.
    upper : array_like
        Prediction for the upper quantile.
    interval_range : int
        Percentage covered by the interval. For instance, if lower and upper correspond to 0.05 and 0.95
        quantiles, interval_range is 90.

    Returns
    -------
    out : dict
        Dictionary containing vectors for the interval scores, but also the dispersion, underprediction and
        overprediction.

    Raises
    ------
    ValueError:
        If the observation, the lower and upper vectors are not the same length or if interval_range is not
        between 0 and 100
    """
    if len(lower) != len(upper) or len(lower) != len(observation):
        raise ValueError("vector shape mismatch")
    if interval_range > 100 or interval_range < 0:
        raise ValueError("interval range should be between 0 and 100")

    #make sure vector operation works
    obs,l,u = np.array(observation),np.array(lower),np.array(upper)

    alpha = 1-interval_range/100 #prediction probability outside the interval
    dispersion = u - l
    underprediction = (2/alpha) * (l-obs) * (obs < l)
    overprediction = (2/alpha) * (obs-u) * (obs > u)
    score = dispersion + underprediction + overprediction
    if not specify_range_out:
        out = {'interval_score': score,
               'dispersion': dispersion,
               'underprediction': underprediction,
               'overprediction': overprediction}
    else:
        out = {f'{interval_range}_interval_score': score,
               f'{interval_range}_dispersion': dispersion,
               f'{interval_range}_underprediction': underprediction,
               f'{interval_range}_overprediction': overprediction}
    return out

def coverage(observation,lower,upper):
    """coverage. Output the fraction of observations within lower and upper.

    Parameters
    ----------
    observation : array_like
        Vector of observations.
    lower : array_like
        Prediction for the lower quantile.
    upper : array_like
        Prediction for the upper quantile.

    Returns
    -------
    cov : float
        Fraction of observations within the lower and upper bound.


    Raises
    ------
    ValueError:
        If the observation, the lower and upper vectors are not the same length.
    """
    if len(lower) != len(upper) or len(lower) != len(observation):
        raise ValueError("vector shape mismatch")

    #make sure vector operation works
    obs,l,u = np.array(observation),np.array(lower),np.array(upper)

    return np.mean(np.logical_and(obs >= l, obs <= u))



if __name__ == '__main__':
    observation = np.array([0.5]*3)
    lower = np.array([0.4,0.5,0.6])
    upper = np.array([0.45,0.55,0.65])

    print(coverage(observation,lower,upper))

