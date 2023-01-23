#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to combine forecasts into an ensemble model.

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import numpy as np
import pandas as pd
from .base_classes import Predictions


def median_ensemble(predictions_list,**kwargs):
    """median_ensemble.

    Parameters
    ----------
    predictions_list : list of Predictions object
    """
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    #concatenate the predictions
    all_predictions = pd.concat(predictions_list)
    #get median for quantiles
    ensemble_predictions = all_predictions.groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median().reset_index()

    ensemble_predictions = Predictions(ensemble_predictions,value_col=value_col,quantile_col=quantile_col,
                                       type_col=type_col,t_col=t_col,other_ind_cols=other_ind_cols)

    return ensemble_predictions

def extreme_ensemble(predictions_list,**kwargs):
    """median_ensemble.

    Parameters
    ----------
    predictions_list : list of Predictions object
    """
    #we assume all predictions have the same columns
    value_col = predictions_list[0].value_col
    quantile_col = predictions_list[0].quantile_col
    type_col = predictions_list[0].type_col
    t_col = predictions_list[0].t_col
    other_ind_cols = predictions_list[0].other_ind_cols
    ind_cols = predictions_list[0].ind_cols

    #concatenate the predictions
    all_predictions = pd.concat(predictions_list)

    #get min/max for quantiles
    ensemble_predictions_low = all_predictions[all_predictions[quantile_col] < 0.5].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).min().reset_index()
    ensemble_predictions_upp = all_predictions[all_predictions[quantile_col] > 0.5].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).max().reset_index()
    ensemble_predictions_med = all_predictions[np.isclose(all_predictions[quantile_col],0.5)].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median().reset_index()
    ensemble_predictions_point = all_predictions[all_predictions[type_col] == 'point'].groupby(
        by=ind_cols + [type_col,quantile_col],dropna=False).median().reset_index()

    ensemble_predictions = pd.concat([ensemble_predictions_low,ensemble_predictions_med,
                                      ensemble_predictions_point, ensemble_predictions_upp])

    ensemble_predictions = Predictions(ensemble_predictions,value_col=value_col,quantile_col=quantile_col,
                                       type_col=type_col,t_col=t_col,other_ind_cols=other_ind_cols)

    return ensemble_predictions

