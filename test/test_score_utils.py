
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

#global date values
date1 = date.fromisoformat('2019-12-04')
date2 = date.fromisoformat('2019-12-11')
date3 = date.fromisoformat('2019-12-18')

class TestAllIntervalScores:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'target_end_date':[date1,date2,date1,date2],
                     'quantile':[0.5,0.5,0.8,0.8],
                     'value':[1,2,3,4]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        with pytest.raises(ValueError):
            all_interval_scores_from_df(observations, predictions, interval_ranges=[])

    def test_raise_error_no_median(self):
        data_obs = {'date':[date1,date2], 'value':[1,2]}
        data_pred = {'target_end_date':[date1,date2],
                     'quantile':[0.8,0.8],
                     'value':[3,4]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        with pytest.raises(ValueError):
            all_interval_scores_from_df(observations, predictions, interval_ranges=[])

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[1,1]}
        data_pred = {'location':['US','US']*3,
                     'target_end_date':[date1,date2]*3,
                     'quantile':[0.25,0.25,0.5,0.5,0.75,0.75],
                     'value':[0,0,2,2,2,2]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = all_interval_scores_from_df(observations, predictions, interval_ranges=[50])
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
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[1,1]*2}
        data_pred = {'location':['AL','AL','MA','MA']*3,
                     'target_end_date':[date1,date2]*6,
                     'quantile':[0.25]*4 + [0.5]*4 +[0.75]*4,
                     'value':[0]*4 + [2]*4 + [2]*4}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = all_interval_scores_from_df(observations, predictions, interval_ranges=[50])
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
        expected_scores = pd.DataFrame(data_expected_scores)
        assert expected_scores.compare(scores).empty


class TestMaximumAbsoluteError:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'target_end_date':[date1,date2],
                     'type':['point','point'],
                     'value':[1,2]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        with pytest.raises(ValueError):
            maximum_absolute_error_from_df(observations, predictions)

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[1,1]}
        data_pred = {'location':['US','US'],
                     'target_end_date':[date1,date2],
                     'type':['point','point'],
                     'value':[2,2]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = maximum_absolute_error_from_df(observations, predictions)
        data_expected_scores = {'mae':[1]*2,
                                'date':[date1,date2],
                                'location':['US','US']
                               }
        expected_scores = pd.DataFrame(data_expected_scores)
        assert expected_scores.compare(scores).empty

    def test_score_multi_loc(self):
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[1,1]*2}
        data_pred = {'location':['AL','AL','MA','MA'],
                     'target_end_date':[date1,date2]*2,
                     'type':['point']*4,
                     'value':[2]*4}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = maximum_absolute_error_from_df(observations, predictions)
        data_expected_scores = {'mae':[1]*4,'date':[date1,date2]*2, 'location':['AL','AL','MA','MA']}
        expected_scores = pd.DataFrame(data_expected_scores)
        assert expected_scores.compare(scores).empty


class TestAllCoverage:
    def test_raise_error_t_col_mismatch(self):
        data_obs = {'date':[date1,date2,date3], 'value':[1,2,3]}
        data_pred = {'target_end_date':[date1,date2,date1,date2],
                     'quantile':[0.5,0.5,0.8,0.8],
                     'value':[1,2,3,4]}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        with pytest.raises(ValueError):
           all_coverages_from_df(observations, predictions, interval_ranges=[])

    def test_score_single_loc(self):
        data_obs = {'location':['US','US'], 'date':[date1,date2], 'value':[0.9,1]}
        data_pred = {'location':['US','US']*5,
                     'target_end_date':[date1,date2]*5,
                     'quantile':[0.05]*2 + [0.25]*2 + [0.5]*2 +[0.75]*2 + [0.95]*2,
                     'value':[0]*2 + [0.5]*2 + [0.9]*2 + [0.99]*2 + [2]*2}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = all_coverages_from_df(observations, predictions, interval_ranges=[50,90])
        expected_scores = {'50_cov':0.5,'90_cov':1}
        for key in scores:
            assert scores[key] == expected_scores[key]

    def test_score_multi_loc(self):
        data_obs = {'location':['AL','AL','MA','MA'],'date':[date1,date2]*2, 'value':[0.9,1]*2}
        data_pred = {'location':['AL','AL','MA','MA']*5,
                     'target_end_date':[date1,date2]*10,
                     'quantile':[0.05]*4 + [0.25]*4 + [0.5]*4 +[0.75]*4 + [0.95]*4,
                     'value':[0]*4 + [0.5]*4 + [0.9]*4 + [0.99]*4 + [2]*4}
        observations = pd.DataFrame(data_obs)
        predictions = pd.DataFrame(data_pred)
        scores = all_coverages_from_df(observations, predictions, interval_ranges=[50,90])
        expected_scores = {'50_cov':0.5,'90_cov':1}
        for key in scores:
            assert scores[key] == expected_scores[key]


