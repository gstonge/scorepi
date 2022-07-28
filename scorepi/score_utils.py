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
    if not np.array_equal(np.unique(observations.get_x(),axis=0),
                          np.unique(predictions.get_x(),axis=0)):
        raise ValueError("Values for the independent columns do not match")
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
    for interval_range in interval_ranges:
        q_low,q_upp = 0.5-interval_range/200,0.5+interval_range/200
        if np.any(predictions.get_quantile(q_low) >= predictions.get_quantile(q_upp)):
            print(predictions.get_quantile(q_low), predictions.get_quantile(q_upp))
            raise RuntimeError("something went wrong")
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
    df['point_absolute_error'] = median_absolute_error
    df['median_absolute_error'] = median_absolute_error
    df['median_absolute_error_underprediction'] = median_absolute_error_underprediction
    df['median_absolute_error_overprediction'] = median_absolute_error_overprediction
    return df


# def all_timestamped_scores_from_df(observations, predictions,
                                   # interval_ranges=[10,20,30,40,50,60,70,80,90,95,98],
                                   # loc_col_observation='location',loc_col_prediction='location',
                                   # t_col_observation='date', t_col_prediction='target_end_date',
                                   # value_col_observation='value', value_col_prediction='value',
                                   # quantile_col='quantile', type_col='type', **kwargs):
    # """all_timestamped_scores_from_df.

    # Parameters
    # ----------
    # observations : DataFrame object
        # Dateframe for the observations across time.
    # predictions : DataFrame object
        # Dateframe for the predictions (intervals) across time.
    # interval_ranges : list of int
        # Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
        # to the interval for the 0.05 and 0.95 quantiles.
    # t_col_observation : str
        # Column label for the timestamp of observations.
    # t_col_prediction : str
        # Column label for the timestamp of predictions.
    # value_col_observation : str
        # Column label for the observations' value.
    # value_col_prediction : str
        # Column label for the predictions' value.
    # quantile_col : str
        # Column label for the predictions' quantile.

    # Returns
    # -------
    # df : DataFrame
        # DataFrame containing the interval score for each interval range across time, but also the dispersion,
        # underprediction and overprediction. Also contains the weighted_interval_score.

    # Raises
    # ------
    # ValueError:
        # If the timestamp columns does not match for observations and predictions.
        # If the median is not calculated.
    # """
    # #verify that the t_col (usually dates) matches between observations and predictions
    # if not np.array_equal(observations[t_col_observation].drop_duplicates().to_numpy(),
                  # predictions[t_col_prediction].drop_duplicates().to_numpy()):
        # raise ValueError("Values for the timestamp do not match")
    # #median must be calculated
    # if len(predictions[np.isclose(predictions[quantile_col],0.5)]) == 0:
        # raise ValueError("The median must be calculated")

    # median = predictions[np.isclose(predictions[quantile_col],0.5)][value_col_prediction].to_numpy()
    # point = predictions[predictions[type_col] == 'point'][value_col_prediction].to_numpy()
    # obs = observations[value_col_observation].to_numpy()
    # point_absolute_error = np.abs(obs-point)
    # median_absolute_error = np.abs(obs-median)
    # median_absolute_error_underprediction = np.heaviside(median-obs,0) * median_absolute_error
    # median_absolute_error_overprediction = np.heaviside(obs-median,0) * median_absolute_error

    # #calculate wis
    # wis = 0.5 * median_absolute_error
    # df_list = []
    # for interval_range in interval_ranges:
        # q_low,q_upp = 0.5-interval_range/200,0.5+interval_range/200
        # pred_low = predictions[np.isclose(predictions[quantile_col],q_low)]
        # pred_upp = predictions[np.isclose(predictions[quantile_col],q_upp)]
        # score = interval_score(obs,
                               # pred_low[value_col_prediction].to_numpy(),
                               # pred_upp[value_col_prediction].to_numpy(),
                               # interval_range,specify_range_out=True)
        # alpha = 1-(q_upp-q_low)
        # wis += 0.5 * alpha * score[f'{interval_range}_interval_score']
        # score[t_col_observation] = list(observations[t_col_observation])
        # score[loc_col_observation] = list(observations[loc_col_observation])
        # df_list.append(pd.DataFrame(score))
    # wis /= (len(interval_ranges) + 1/2)
    # df = reduce(lambda x, y: pd.merge(x, y, on = [t_col_observation,loc_col_observation]), df_list)

    # df['wis'] = wis
    # df['point_absolute_error'] = median_absolute_error
    # df['median_absolute_error'] = median_absolute_error
    # df['median_absolute_error_underprediction'] = median_absolute_error_underprediction
    # df['median_absolute_error_overprediction'] = median_absolute_error_overprediction
    # return df

# def absolute_error_from_df(observations, predictions,
                                   # loc_col_observation='location',loc_col_prediction='location',
                                   # t_col_observation='date', t_col_prediction='target_end_date',
                                   # value_col_observation='value', value_col_prediction='value',
                                   # type_col='type', **kwargs):
    # """maximum_absolute_error_from_df.

    # Parameters
    # ----------

    # observations : DataFrame object
        # Dateframe for the observations across time.
    # predictions : DataFrame object
        # Dateframe for the predictions (intervals) across time.
    # t_col_observation : str
        # Column label for the timestamp of observations.
    # t_col_prediction : str
        # Column label for the timestamp of predictions.
    # value_col_observation : str
        # Column label for the observations' value.
    # value_col_prediction : str
        # Column label for the predictions' value.
    # type_col : str
        # Column label for the type of prediction.
    # """
    # if not np.array_equal(observations[t_col_observation].drop_duplicates().to_numpy(),
                  # predictions[t_col_prediction].drop_duplicates().to_numpy()):
        # raise ValueError("Values for the timestamp do not match")
    # pred = predictions[predictions[type_col] == 'point']
    # mae = maximum_absolute_error(observations[value_col_observation],pred[value_col_prediction])
    # df = pd.DataFrame({'mae': mae, t_col_observation: observations[t_col_observation],
                       # loc_col_observation: observations[loc_col_observation]})
    # return df


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
    if not np.array_equal(np.unique(observations.get_x(),axis=0),
                          np.unique(predictions.get_x(),axis=0)):
        raise ValueError("Values for the independent columns do not match")

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
    if mismatched_allowed:
        #get the intersection of datasets
        for col in observations.ind_cols:
            pred = predictions.filter(predictions[col].isin(observations[col]))
            obs = observations.filter(observations[col].isin(predictions[col]))

    else:
        pred = predictions
        obs = observations

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
        d[f"0_{part}_wis_fraction"] = 0.5*df[f"median_absolute_error_{part}"].sum()\
                / ((len(interval_ranges) + 1/2) * wis_total)

    #other interval range
    for interval_range in interval_ranges:
        alpha = 1 - interval_range/100
        norm = (len(interval_ranges) + 1/2) / (0.5 * alpha)
        for part in ["dispersion", "underprediction", "overprediction"]:
            contribution = df[f"{interval_range}_{part}"].sum()
            d[f"{interval_range}_{part}_wis_fraction"] = contribution / (norm*wis_total)

    #aggregate over intervals
    for part in ["dispersion", "underprediction", "overprediction"]:
        d[f"{part}_wis_fraction"] = sum([d[f"{interval_range}_{part}_wis_fraction"]\
                                         for interval_range in interval_ranges])
    #add missing part from 0 interval
    for part in ["underprediction", "overprediction"]:
        d[f"{part}_wis_fraction"] += d[f"0_{part}_wis_fraction"]

    return d,df

