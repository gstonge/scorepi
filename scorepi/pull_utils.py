#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to pull covid surveillance data and model predictions.

Data is pulled from:

https://github.com/reichlab/covid19-forecast-hub

and

https://github.com/midas-network/covid19-scenario-modeling-hub

Author: Guillaume St-Onge <stongeg1@gmail.com>
"""

import requests
import numpy as np
import pandas as pd
from epiweeks import Week
from datetime import date, timedelta

def _url_checker(url):
    get = requests.get(url)
    if get.status_code != 200:
        raise requests.exceptions.RequestException(f"{url}: is Not reachable")

def pull_covid_forecast_hub_predictions(model,start_week,end_week):
    """pull_covid_forecast_hub_predictions. Load predictions of the model saved by the covid19 forecast hub.

    Parameters
    ----------
    model : str
        Model name on thhe
    start_week : Week object
        First epiweek of the range.
    end_week : Week object
        Last epiweek of the range.
    """

    week_list = [start_week]
    while week_list[-1] != end_week:
        week_list.append(week_list[-1]+1)
    pull_dates = [(week.startdate()+timedelta(days = 1)) for week in week_list]
    get_url = lambda date:f"https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-processed/{model}/{date}-{model}.csv"
    #check which files are accessible
    url_list = []
    for date in pull_dates:
        try:
            url = get_url(date.isoformat())
            _url_checker(url)
            url_list += [url]
        except requests.exceptions.RequestException:
            #some group push date is on sundays
            try:
                url = get_url((date+timedelta(days = -1)).isoformat())
                _url_checker(url)
                url_list += [url]
            except requests.exceptions.RequestException:
                print(f"Data for date {date.isoformat()} is unavailable")
    df_predictions = pd.concat([pd.read_csv(url,dtype={'location':str},
                                            parse_dates=['target_end_date','forecast_date']) for url in url_list])
    return df_predictions


def pull_scenario_modeling_hub_predictions(model,dates):
    """pull_scenario_modeling_hub_predictions. Load predictions of the model saved by the scenario modeling
    hub.

    Parameters
    ----------
    model : str
        Model name on thhe
    dates : list or string
        List of potential dates in the iso format, e.g., 'yyyy-mm-dd'
    """
    predictions = None
    if isinstance(dates,str):
        dates = [dates]
    for date in dates:
        url = f"https://raw.githubusercontent.com/midas-network/covid19-scenario-modeling-hub/master/data-processed/{model}/{date}-{model}"
        for ext in [".csv",".gz",".zip",".csv.zip",".csv.gz"]:
            try:
                predictions = pd.read_csv(url+ext,dtype={'location':str},parse_dates=['target_end_date'])
            except:
                pass
    if predictions is None:
        print(f"Data for model {model} and date {dates} unavailable")
    return predictions


def pull_surveillance_data(target='death',incidence=True):
    mapping = {'death':'Deaths', 'case':'Cases', 'hospitalization': 'Hospitalizations'}
    if incidence:
        s = 'Incident'
    else:
        s = 'Cumulative'
    url = f"https://media.githubusercontent.com/media/reichlab/covid19-forecast-hub/master/data-truth/truth-{s}%20{mapping[target]}.csv"
    return pd.read_csv(url, dtype={'location':str})

