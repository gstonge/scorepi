#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Score functions that are applied on a dataframe.

Author: Guillaume St-Onge <stongeg1@gmail.com>

NOTE: right now, dataframe must be filtered for a single location?.

"""

import numpy as np
import pandas as pd
from functools import reduce
from .score_functions import *
from .base_classes import *
from itertools import product


def all_timestamped_scores_from_df(observations, predictions,
                                   interval_ranges=[10,20,30,40,50,60,70,80,90,95,98], **kwargs):
    """all_timestamped_scores_from_df.

    Parameters
    ----------
    observations : Observations object
        Specialized dateframe for the observations across time.
    predictions : Predictions object
        Specialized dateframe for the predictions (quantile and point) across time.
    interval_ranges : list of int
        Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
        to the interval for the 0.05 and 0.95 quantiles.

    Returns
    -------
    df : DataFrame
        DataFrame containing the interval score for each interval range across time, but also the dispersion,
        underprediction and overprediction. Also contains the weighted_interval_score and absolute errors.

    Raises
    ------
    ValueError:
        If the independent columns do not match for observations and predictions.
        If the median is not calculated.
        If the point estimate is not included.
    """
    #verify that the independent variable columns (usually dates and location) matches
    # if not np.array_equal(observations.get_unique_x(), predictions.get_unique_x()):
        # raise ValueError("Values for the independent columns do not match")
    #median and point estimate must be calculated
    if len(predictions.get_quantile(0.5)) == 0:
        raise ValueError("The median must be calculated")
    if len(predictions.get_point()) == 0:
        raise ValueError("The point estimate must be included")

    median = predictions.get_quantile(0.5)
    point = predictions.get_point()
    obs = observations.get_value()
    point_absolute_error = np.abs(obs-point)
    median_absolute_error = np.abs(obs-median)
    median_absolute_error_underprediction = np.heaviside(median-obs,0) * median_absolute_error
    median_absolute_error_overprediction = np.heaviside(obs-median,0) * median_absolute_error

    #calculate wis
    wis = 0.5 * median_absolute_error
    df_list = []
    if interval_ranges:
        for interval_range in interval_ranges:
            q_low,q_upp = 0.5-interval_range/200,0.5+interval_range/200
            if np.any(predictions.get_quantile(q_low) > predictions.get_quantile(q_upp)):
                print(predictions.get_quantile(q_upp) - predictions.get_quantile(q_low))
                raise RuntimeError("something went wrong, upper quantile bigger than lower quantile")
            score = interval_score(obs,
                                   predictions.get_quantile(q_low),
                                   predictions.get_quantile(q_upp),
                                   interval_range,specify_range_out=True)
            alpha = 1-(q_upp-q_low)
            wis += 0.5 * alpha * score[f'{interval_range}_interval_score']
            score[observations.t_col] = list(observations.get_t())
            for col in observations.other_ind_cols:
                score[col] = list(observations[col])
            df_list.append(pd.DataFrame(score))
        wis /= (len(interval_ranges) + 1/2)
        df = reduce(lambda x, y: pd.merge(x, y, on = observations.ind_cols), df_list)
        df['wis'] = wis
    else:
        df = pd.DataFrame({col:list(observations[col]) for col in observations.other_ind_cols})
        df[observations.t_col] = list(observations.get_t())

    df['point_absolute_error'] = point_absolute_error
    df['median_absolute_error'] = median_absolute_error
    df['median_absolute_error_underprediction'] = median_absolute_error_underprediction
    df['median_absolute_error_overprediction'] = median_absolute_error_overprediction
    return df


def all_coverages_from_df(observations, predictions, interval_ranges=[10,20,30,40,50,60,70,80,90,95,98],
                          **kwargs):
    """all_interval_score_from_df.

    Parameters
    ----------
    observations : DataFrame object
        Dateframe for the observations across time.
    predictions : DataFrame object
        Dateframe for the predictions (intervals) across time.
    interval_ranges : list of int
        Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
        to the interval for the 0.05 and 0.95 quantiles.

    Returns
    -------
    out : dict
        Dictionary containing the coverage for all interval ranges.

    Raises
    ------
    ValueError:
        If the independent columns do not match for observations and predictions.
    """
    #verify that the independent variable columns (usually dates and location) matches
    # if not np.array_equal(observations.get_unique_x(), predictions.get_unique_x()):
        # raise ValueError("Values for the independent columns do not match")

    out = dict()
    for interval_range in interval_ranges:
        q_low,q_upp = 0.5-interval_range/200,0.5+interval_range/200
        cov = coverage(observations.get_value(),
                       predictions.get_quantile(q_low),
                       predictions.get_quantile(q_upp))
        out[f'{interval_range}_cov'] = cov
    return out

def all_scores_from_df(observations, predictions, interval_ranges=[10,20,30,40,50,60,70,80,90,95,98],
                       mismatched_allowed=False, **kwargs):
    """all_scores_from_df.

    Parameters
    ----------
    observations : DataFrame object
        Dateframe for the observations across time.
    predictions : DataFrame object
        Dateframe for the predictions (intervals) across time.
    interval_ranges : list of int
        Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
        to the interval for the 0.05 and 0.95 quantiles.
    t_col_observation : str
        Column label for the timestamp of observations.
    t_col_prediction : str
        Column label for the timestamp of predictions.
    mismatched_allowed : bool
        If true and the timestamp does not match between predictions and observations, apply the score
        functions to the filtered data where both match.



    See the underlying function calls for the **kwargs

    Returns
    -------
    d,df : tuple of dictionary and DataFrame
        The dictionary contains scores and data aggregated over all timestamps.
        The DataFrame contains the timestamped score.

    Raises
    ------
    ValueError:
        If the timestamp columns does not match for observations and predictions.
    """
    pred = predictions.copy()
    obs = observations.copy()

    if len(obs.other_ind_cols) == 0:
        #get the intersection of predictions and observations
        if mismatched_allowed:
            pred,obs = intersec(pred,obs)
        d,df = all_scores_core(obs, pred, interval_ranges, **kwargs)

    #get score independently for each other independent col of the predictions
    else:
        d_list = []
        df_list = []
        for x in product(*(_get_unique_values_iter(pred,col) for col in pred.other_ind_cols)):
            pred_ = pred.copy()
            obs_ = obs.copy()
            #filter predictions and observations
            for col,val in x:
                pred_ = pred_.filter(pred_[col] == val)
                if col in obs_.other_ind_cols:
                    obs_ = obs_.filter(obs_[col] == val)
            #get the intersection of predictions and observations
            if mismatched_allowed:
                pred_,obs_ = intersec(pred_,obs_)
            #calculate scores and identify them by independent col values
            d_,df_ = all_scores_core(obs_, pred_, interval_ranges, **kwargs)
            for col,val in x:
                d_[col] = val
                df_[col] = val
            d_list.append(d_)
            df_list.append(df_)
        #combine scores
        d = pd.DataFrame(d_list)
        df = pd.concat(df_list)

    return d,df

def intersec(predictions,observations):
        pred = predictions.copy()
        obs = observations.copy()
        
        # groups the predictions by pred.t_col,
        # keeping the date if there are more than or equal to 2 unique prediction types 
        # ("point" and "quantile")
        pred = pred[pred.groupby(pred.t_col).type.transform(lambda x: x.nunique()).ge(2)]

        for col in observations.ind_cols:
            pred = pred[pred[col].isin(obs[col])]
            obs = obs[obs[col].isin(pred[col])]
        pred = Predictions( pred, 
                            value_col=predictions.value_col, 
                            quantile_col=predictions.quantile_col, 
                            type_col=predictions.type_col, 
                            t_col=predictions.t_col,
                            other_ind_cols=predictions.other_ind_cols)
        obs = Observations( obs,
                            value_col=observations.value_col, 
                            t_col=observations.t_col, 
                            other_ind_cols=observations.other_ind_cols)
        return pred, obs

def _get_unique_values_iter(df,col):
    for val in df[col].unique():
        yield col,val


def all_scores_core(obs, pred, interval_ranges, **kwargs):
    #get all timestamped scores
    df = all_timestamped_scores_from_df(obs, pred, **kwargs)

    #get all aggregated scores
    d = all_coverages_from_df(obs,pred)

    #report number of timestamp that match between observations and predictions
    d["nb_t_match"] = df["wis"].count()

    #aggregate wis and absolute error
    wis_total = df["wis"].sum()
    wis_mean = df["wis"].mean()
    d['wis_total'] = wis_total
    d['wis_mean'] = wis_mean

    pae_total = df["point_absolute_error"].sum()
    pae_mean = df["point_absolute_error"].mean()
    d['point_absolute_error_total'] = pae_total
    d['point_absolute_error_mean'] = pae_mean

    #calculate percentage of wis due to dispersion,underprediction,overprediction
    #===============================================================================

    #interval range 0
    for part in ["underprediction", "overprediction"]:
        if wis_total > 0:
            d[f"0_{part}_wis_fraction"] = 0.5*df[f"median_absolute_error_{part}"].sum()\
                    / ((len(interval_ranges) + 1/2) * wis_total)
        else:
            d[f"0_{part}_wis_fraction"] = np.nan

    #other interval range
    for interval_range in interval_ranges:
        alpha = 1 - interval_range/100
        norm = (len(interval_ranges) + 1/2) / (0.5 * alpha)
        for part in ["dispersion", "underprediction", "overprediction"]:
            contribution = df[f"{interval_range}_{part}"].sum()
            if wis_total > 0:
                d[f"{interval_range}_{part}_wis_fraction"] = contribution / (norm*wis_total)
            else:
                d[f"{interval_range}_{part}_wis_fraction"] = np.nan

    #aggregate over intervals
    for part in ["dispersion", "underprediction", "overprediction"]:
        d[f"{part}_wis_fraction"] = sum([d[f"{interval_range}_{part}_wis_fraction"]\
                                         for interval_range in interval_ranges])
    #add missing part from 0 interval
    for part in ["underprediction", "overprediction"]:
        d[f"{part}_wis_fraction"] += d[f"0_{part}_wis_fraction"]

    return d,df