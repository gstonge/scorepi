
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the score utils module

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date
from scorepi import *
pd.set_option('display.max_columns', None)

#global date values
date1 = date.fromisoformat('2019-12-04')
date2 = date.fromisoformat('2019-12-11')
date3 = date.fromisoformat('2019-12-18')

class TestAllIntervalScores:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'date':[date1,date2,date1,date2],
                     'quantile':[0.5,0.5,0.8,0.8],
                     'value':[1,2,3,4]}
        observations = Observations(data_obs)
        predictions = Predictions(data_pred)
        with pytest.raises(ValueError):
            all_timestamped_scores_from_df(observations, predictions, interval_ranges=[])

    def test_raise_error_no_median(self):
        data_obs = {'date':[date1,date2], 'value':[1,2]}
        data_pred = {'date':[date1,date2],
                     'quantile':[0.8,0.8],
                     'value':[3,4]}
        observations = Observations(data_obs)
        predictions = Predictions(data_pred)
        with pytest.raises(ValueError):
            all_timestamped_scores_from_df(observations, predictions, interval_ranges=[])

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[1,1]}
        data_pred = {'location':['US','US']*3,
                     'date':[date1,date2]*3,
                     'quantile':[0.25,0.25,0.5,0.5,0.75,0.75],
                     'value':[0,0,2,2,2,2]}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_timestamped_scores_from_df(observations, predictions, interval_ranges=[50])
        scores = scores[['50_interval_score','50_dispersion','50_underprediction','50_overprediction','date',
                         'location','wis','median_absolute_error','median_absolute_error_underprediction',
                         'median_absolute_error_overprediction']]
        data_expected_scores = {'50_interval_score':[2.]*2,
                                '50_dispersion':[2.]*2,
                                '50_underprediction':[0.]*2,
                                '50_overprediction':[0.]*2,
                                'date':[date1,date2],
                                'location':['US','US'],
                                'wis':[(2/4+1/2)/1.5]*2,
                                'median_absolute_error':[1.]*2,
                                'median_absolute_error_underprediction':[1.]*2,
                                'median_absolute_error_overprediction':[0.]*2}
        expected_scores = pd.DataFrame(data_expected_scores)
        assert expected_scores.compare(scores).empty

    def test_score_multi_loc(self):
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[1.,1.]*2}
        data_pred = {'location':['AL','AL','MA','MA']*3,
                     'date':[date1,date2]*6,
                     'quantile':[0.25]*4 + [0.5]*4 +[0.75]*4,
                     'value':[0.]*4 + [2.]*4 + [2.]*4}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_timestamped_scores_from_df(observations, predictions, interval_ranges=[50])
        scores = scores[['50_interval_score','50_dispersion','50_underprediction','50_overprediction','date',
                         'location','wis','median_absolute_error','median_absolute_error_underprediction',
                         'median_absolute_error_overprediction']].sort_values(by=['date','location']).reset_index(drop=True)

        data_expected_scores = {'50_interval_score':[2.]*4,
                                '50_dispersion':[2.]*4,
                                '50_underprediction':[0.]*4,
                                '50_overprediction':[0.]*4,
                                'date':[date1,date2]*2,
                                'location':['AL','AL','MA','MA'],
                                'wis':[(2/4+1/2)/1.5]*4,
                                'median_absolute_error':[1.]*4,
                                'median_absolute_error_underprediction':[1.]*4,
                                'median_absolute_error_overprediction':[0.]*4}
        expected_scores = pd.DataFrame(data_expected_scores).sort_values(by=['date','location']).reset_index(drop=True)
        assert expected_scores.compare(scores).empty


class TestAbsoluteError:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'date':[date1,date2],
                     'type':['quantile','quantile'],
                     'quantile':[0.5,0.5],
                     'value':[1,2]}
        observations = Observations(data_obs)
        predictions = Predictions(data_pred)
        with pytest.raises(ValueError):
            all_timestamped_scores_from_df(observations, predictions, interval_ranges=[])

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[1,1]}
        data_pred = {'location':['US','US']*2,
                     'date':[date1,date2]*2,
                     'type':['point','point', 'quantile', 'quantile'],
                     'quantile':[None,None,0.5,0.5],
                     'value':[2,2,2,2]}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_timestamped_scores_from_df(observations, predictions, interval_ranges=[])
        scores = scores[['location','date','point_absolute_error','median_absolute_error']].sort_values(
            by=['date','location']).reset_index(drop=True)

        data_expected_scores = {'location':['US','US'],
                                'date':[date1,date2],
                                'point_absolute_error':[1]*2,
                                'median_absolute_error':[1]*2}
        expected_scores = pd.DataFrame(data_expected_scores).sort_values(by=['date','location']).reset_index(drop=True)

        assert expected_scores.compare(scores).empty

    def test_score_multi_loc(self):
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[1,1]*2}
        data_pred = {'location':['AL','AL','MA','MA']*2,
                     'date':[date1,date2]*4,
                     'type':['point']*4+['quantile']*4,
                     'quantile':[None,None]*2+[0.5,0.5]*2,
                     'value':[2]*8}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_timestamped_scores_from_df(observations, predictions, interval_ranges=[])
        scores = scores[['location','date','point_absolute_error','median_absolute_error']].sort_values(
            by=['date','location']).reset_index(drop=True)

        data_expected_scores = {'location':['AL','AL','MA','MA'],
                                'date':[date1,date2]*2,
                                'point_absolute_error':[1]*4,
                                'median_absolute_error':[1]*4}

        expected_scores = pd.DataFrame(data_expected_scores).sort_values(by=['date','location']).reset_index(drop=True)
        assert expected_scores.compare(scores).empty


class TestAllCoverage:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'date':[date1,date2,date1,date2],
                     'quantile':[0.2,0.2,0.8,0.8],
                     'value':[1,2,3,4]}
        observations = Observations(data_obs)
        predictions = Predictions(data_pred)
        with pytest.raises(ValueError):
           all_coverages_from_df(observations, predictions, interval_ranges=[40])

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[0.9,1]}
        data_pred = {'location':['US','US']*5,
                     'date':[date1,date2]*5,
                     'quantile':[0.05]*2 + [0.25]*2 + [0.5]*2 +[0.75]*2 + [0.95]*2,
                     'value':[0]*2 + [0.5]*2 + [0.9]*2 + [0.99]*2 + [2]*2}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_coverages_from_df(observations, predictions, interval_ranges=[50,90])
        expected_scores = {'50_cov':0.5,'90_cov':1}
        for key in scores:
            assert scores[key] == expected_scores[key]

    def test_score_multi_loc(self):
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[0.9,1]*2}
        data_pred = {'location':['AL','AL','MA','MA']*5,
                     'date':[date1,date2]*10,
                     'quantile':[0.05]*4 + [0.25]*4 + [0.5]*4 +[0.75]*4 + [0.95]*4,
                     'value':[0]*4 + [0.5]*4 + [0.9]*4 + [0.99]*4 + [2]*4}
        observations = Observations(data_obs, other_ind_cols=['location'])
        predictions = Predictions(data_pred, other_ind_cols=['location'])
        scores = all_coverages_from_df(observations, predictions, interval_ranges=[50,90])
        expected_scores = {'50_cov':0.5,'90_cov':1}
        for key in scores:
            assert scores[key] == expected_scores[key]

