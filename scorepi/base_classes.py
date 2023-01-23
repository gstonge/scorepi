#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of base classes used by the scoring methods in this package.

alias for col attribute might not be best idea. keep only col labels, and create methods to access cols

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import pandas as pd
import numpy as np

class Observations(pd.DataFrame):
    _metadata = ['value_col','t_col','other_ind_cols', 'ind_cols']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,
                    value_col='value', t_col='date', other_ind_cols=[]):
        """
        Parameters
        ----------
        value_col :
            Column label for the observations' value.
        t_col :
            Column label for the timestamp of predictions.
        other_ind_cols :
            List of other column labels that serve as independent variable, e.g., location.
        """
        super().__init__(data=data,index=index,columns=columns,dtype=dtype,copy=copy)

        self.value_col = value_col
        self.t_col = t_col
        self.other_ind_cols = other_ind_cols
        #for t_col and value_col, there must be a column with the appropriate name
        try:
            self[value_col]
            self[t_col]
            if other_ind_cols:
                self[other_ind_cols]
        except KeyError:
            raise ValueError("Column name mismatch")

        #sort values in the DataFrame based on time and other independent columns
        self.ind_cols = [t_col] + other_ind_cols
        self.sort_values(by=self.ind_cols,inplace=True)
        self.reset_index(drop=True,inplace=True)

    def filter(self,bool_df):
        new_obs = self[bool_df].copy().reset_index(drop=True)
        return type(self)(new_obs,value_col=self.value_col,t_col=self.t_col,
                          other_ind_cols=self.other_ind_cols)

    def copy(self):
        new_obs = super().copy()
        return type(self)(new_obs,value_col=self.value_col,t_col=self.t_col,
                          other_ind_cols=self.other_ind_cols)

    def get_value(self):
        return self[self.value_col].to_numpy()

    def get_t(self):
        return self[self.t_col].to_numpy()

    def get_x(self):
        return self[self.ind_cols].to_numpy()

    def get_unique_x(self):
        #convert to str such that it can be compared
        return np.unique(np.array((self[self.ind_cols]).to_numpy(), dtype=str), axis=0)



class Predictions(pd.DataFrame):
    _metadata = ['value_col','quantile_col','type_col','t_col','other_ind_cols','ind_cols']

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None,
                 value_col='value', quantile_col='quantile', type_col='type',
                 t_col='date', other_ind_cols=[]):
        """
        Parameters
        ----------
        value_col : str
            Column label for the observations' value.
        quantile_col : str
            Column label for the predictions' quantile.
        type_col : str
            Column label for the type of predictions (usually, quantile or point).
        t_col : str
            Column label for the timestamp of predictions.
        other_ind_cols : List of str
            List of other column labels that serve as independent variable, e.g., location.
        """


        super().__init__(data=data,index=index,columns=columns,dtype=dtype,copy=copy)

        self.value_col = value_col
        self.quantile_col = quantile_col
        self.type_col = type_col
        self.t_col = t_col
        self.other_ind_cols = other_ind_cols
        #for t_col, value_col, quantile_col, and type_col there must be a column with the appropriate name
        try:
            self[value_col]
            self[quantile_col]
            self[t_col]
            if other_ind_cols:
                self[other_ind_cols]
        except KeyError:
            raise ValueError("Column name mismatch")

        try:
            self[type_col]
        except KeyError:
            #type col not defined, therefore we assume it is only quantiles.
            self[type_col] = 'quantile'


        #sort values in the DataFrame based on time and other independent columns
        self.ind_cols = [t_col] + other_ind_cols
        self.sort_values(by=self.ind_cols,inplace=True)
        self.reset_index(drop=True,inplace=True)

    def filter(self,bool_df):
        new_pred = self[bool_df].copy().reset_index(drop=True)
        return type(self)(new_pred,value_col=self.value_col,quantile_col=self.quantile_col,
                          type_col=self.type_col,t_col=self.t_col,other_ind_cols=self.other_ind_cols)

    def copy(self):
        new_pred = super().copy()
        return type(self)(new_pred,value_col=self.value_col,quantile_col=self.quantile_col,
                          type_col=self.type_col,t_col=self.t_col,other_ind_cols=self.other_ind_cols)

    def get_t(self):
        return self[self.t_col].to_numpy()

    def get_x(self):
        return self[self.ind_cols].to_numpy()

    def get_unique_x(self):
        #convert to str such that it can be compared
        return np.unique(np.array((self[self.ind_cols]).to_numpy(), dtype=str), axis=0)


    def get_point(self):
        point = self[self[self.type_col] == 'point'][self.value_col].to_numpy()
        #if no point estimate, return the median
        if len(point) == 0:
            point = self.get_quantile(0.5)
        return point


    def get_quantile(self,q):
        new_pred = self.filter(np.isclose(self[self.quantile_col],q))
        return new_pred[self.value_col].to_numpy()

