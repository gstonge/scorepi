#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of base classes used by the scoring methods in this package.

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import pandas as pd

class Observations(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,
                    t_col='date',value_col='value',loc_col='location'):
        super().__init__(data=data,index=None, columns=None, dtype=None, copy=None)
        self.t_col_label = t_col
        self.value_col_label = value_col
        self.loc_col_label = loc_col
        #for t_col and value_col, there must be a column with the appropriate name
        try:
            self.t_col = self[t_col]
            self.value_col = self[value_col]
        except KeyError:
            raise ValueError("Wrong name for t_col or value_col")


        #if loc_col is not defined point to None
        try:
            self.loc_col = self[loc_col]
        except KeyError:
            self.loc_col = None

        #sort values in the DataFrame based on time and locations
        self.sort_col = [t_col,loc_col] if self.loc_col else [t_col]
        self.sort_values(by=self.sort_col,inplace=True)


class Predictions(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,
                    t_col='date',value_col='value',loc_col='location',quantile_col='quantile',
                    type_col='type'):
        super().__init__(data=data,index=None, columns=None, dtype=None, copy=None)
        self.t_col_label = t_col
        self.value_col_label = value_col
        self.loc_col_label = loc_col
        self.quantile_col_label = quantile_col
        self.type_col_label = type_col
        #for t_col, value_col, quantile_col, and type_col there must be a column with the appropriate name
        try:
            self.t_col = self[t_col]
            self.value_col = self[value_col]
            self.quantile_col = self[quantile_col]
            self.type_col = self[type_col]
        except KeyError:
            raise ValueError("Wrong name for t_col, value_col, quantile_col or type_col")

        #if loc_col is not defined point to None
        try:
            self.loc_col = self[loc_col]
        except KeyError:
            self.loc_col = None

        #sort values in the DataFrame based on time and locations
        self.sort_col = [t_col,loc_col] if self.loc_col else [t_col]
        self.sort_values(by=self.sort_col,inplace=True)
