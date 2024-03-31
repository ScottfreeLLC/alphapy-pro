"""
Package   : AlphaPy
Module    : metalabel
Created   : December 3, 2022

Copyright 2024 ScottFree Analytics LLC
Mark Conway & Robert D. Scott II

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


#
# Imports
#

import logging
import numpy as np
import pandas as pd
from tqdm import tqdm


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_daily_vol
#

def get_daily_vol(ds_close, p=60):
    r"""Calculate daily volatility for dynamic thresholds.

    Parameters
    ----------
    ds_close : pandas.Series
        Array of closing values.
    p : int
        The lookback period for computing volatility.

    Returns
    -------
    ds_vol : pandas.Series (float)
        The array of volatilities.

    """

    logger.info('Calculating daily volatility for dynamic thresholds')

    ds_vol = ds_close.index.searchsorted(ds_close.index - pd.Timedelta(days=1))
    ds_vol = ds_vol[ds_vol > 0]
    ds_vol = (pd.Series(ds_close.index[ds_vol - 1], index=ds_close.index[ds_close.shape[0] - ds_vol.shape[0]:]))
    # calculate daily returns
    ds_vol = ds_close.loc[ds_vol.index] / ds_close.loc[ds_vol.values].values - 1
    ds_vol = ds_vol.ewm(span=p, min_periods=p).std().dropna()
    return ds_vol

#
# Function get_daily_dollar_vol
#

def get_daily_dollar_vol(df, p=60):
    r"""Calculate daily dollar volume.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing the close and volume values.
    p : int
        The lookback period for computing daily dollar volume.

    Returns
    -------
    ds_dv : pandas.Series (float)
        The array of dollar volumes.

    """

    logger.info('Calculating daily dollar volume')

    fractal_daily = '1D'
    ds_price_avg = df['close'].groupby(pd.Grouper(freq=fractal_daily)).mean().dropna()
    ds_volume_sum = df['volume'].groupby(pd.Grouper(freq=fractal_daily)).sum()
    ds_volume_sum = ds_volume_sum[ds_volume_sum > 0]
    ds_dv = ds_price_avg * ds_volume_sum
    ds_dv = ds_dv.ewm(span=p, min_periods=p).mean()
    return ds_dv[-1]


#
# Function get_t_events
#

def get_t_events(ds_close, threshold=0.01):
    r"""Calculate daily volatility for dynamic thresholds.

    Parameters
    ----------
    ds_close : pandas.Series
        Array of closing values.
    threshold : float
        The minimum volatility value at which events are captured.

    Returns
    -------
    ds_dt : pandas.Series (datetime)
        The vector of datetimes when the events occurred.

    """

    logger.info('Applying Symmetric CUSUM filter')

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    ds_diff = np.log(ds_close).diff().dropna()

    # Get event time stamps for the entire series
    for i in tqdm(ds_diff.index[1:]):
        pos = float(s_pos + ds_diff.loc[i])
        neg = float(s_neg + ds_diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    ds_dt = pd.DatetimeIndex(t_events)
    return ds_dt


#
# Function add_vertical_barrier
#

def add_vertical_barrier(t_events, ds_close, num_days=1.0):
    r"""Calculate daily volatility for dynamic thresholds.

    Parameters
    ----------
    t_events : pandas.Series (datetime)
        Series of events based on the CUSUM Filter
    ds_close : pandas.Series
        Array of closing values.
    num_days : float
        The maximum number of (fractional) days that a trade can be active.

    Returns
    -------
    ds_vb : pandas.Series (datetime)
        The vector of timestamps for the vertical barriers.

    """

    logger.info('Getting Vertical Barriers')

    ds_vb = ds_close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    ds_vb = ds_vb[ds_vb < ds_close.shape[0]]
    ds_vb = pd.Series(ds_close.index[ds_vb], index=t_events[:ds_vb.shape[0]])
    return ds_vb


#
# Function apply_pt_sl_on_t1
#

def apply_pt_sl_on_t1(ds_close, df_events, pt_sl):
    r"""Apply Profit Targets and Stop Losses.

    Parameters
    ----------
    ds_close : pandas.Series
        Array of closing values.
    df_events : pandas.DataFrame
        The record of events
    pt_sl : list[2]
        The profit-taking and stop-loss percentage levels, with 0 disabling the respective level.

    Returns
    -------
    df_touch : pandas.DataFrame
        The dataframe of timestamps at which each barrier was touched

    """

    logger.info('Applying Profit Targets and Stop Losses')

    # Apply stop loss and profit taking, if either event occurs before t1 (end of event).

    df_touch = df_events[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * df_events['trgt']
    else:
        pt = pd.Series(index=df_events.index)

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * df_events['trgt']
    else:
        sl = pd.Series(index=df_events.index)

    for loc, t1 in df_events['t1'].fillna(ds_close.index[-1]).items():
        # path prices
        df0 = ds_close[loc:t1]
        # path returns
        df0 = (df0 / ds_close[loc] - 1) * df_events.at[loc, 'side']
        # earliest stop loss
        df_touch.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        # earliest profit taking
        df_touch.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()

    return df_touch


#
# Function get_events
#

def get_events(ds_close, ds_dt, pt_sl, ds_vol, min_ret, ds_vb=False, ds_side=None):
    r"""Get the dataframe of target events.

    Parameters
    ----------
    ds_close : pandas.Series
        Array of closing values.
    ds_dt : pandas.Series (datetime)
        Series of events based on the CUSUM Filter
    pt_sl : list[2]
        The profit-taking and stop-loss percentage levels, with 0 disabling the respective level.
    ds_vol : pandas.Series (float)
        The array of volatilities used in conjunction with ``pt_sl``.
    min_ret : float
        The minimum target return required for running a triple barrier search.
    ds_vb : pandas.Series (datetime)
        The vector of timestamps for the vertical barriers.
    ds_side : pandas.Series (datetime)
        Side of the bet (long/short) as decided by the primary model

    Returns
    -------
    df_events : pandas.DataFrame
        The record of Triple Barrier events
        - df_events.index is event's starttime
        - df_events.t1 is the event's endtime
        - df_events.trgt is the event's target
        - df_events.side implies the algorithm's position side

    """

    logger.info('Getting the Triple Barrier Method Events')

    # Get the target based on volatility.

    ds_vol = ds_vol.loc[ds_vol.index.intersection(ds_dt)]
    ds_vol = ds_vol[ds_vol > min_ret]

    # Get the vertical barrier with maximum holding period.

    if ds_vb is False:
        ds_vb = pd.Series(pd.NaT, index=ds_dt)

    # Form the events dataframe, applying the stop loss on the vertical barrier.

    if ds_side is None:
        ds_side_ = pd.Series(1., index=ds_vol.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        ds_side_ = ds_side.loc[ds_vol.index]
        pt_sl_ = pt_sl[:2]

    df_events = pd.concat({'t1': ds_vb, 'trgt': ds_vol, 'side': ds_side_}, axis=1)
    df_events = df_events.dropna(subset=['side'])

    # Apply the Triple Barrier.

    df_tbm = apply_pt_sl_on_t1(ds_close, df_events, pt_sl_)

    df_events['t1'] = df_tbm.dropna(how='all').min(axis=1)  # pd.min ignores nan

    if ds_side is None:
        df_events = df_events.drop('side', axis=1)

    return df_events


#
# Function barrier_touched
#

def barrier_touched(df_meta):
    r"""Determine whether the barriers have been touched.

    Parameters
    ----------
    df_meta : pandas.DataFrame
        The meta-labeled events

    Returns
    -------
    df_meta : pandas.DataFrame
        The dataframe containing TBM returns, target, and labels

    """

    store = []
    for i in np.arange(len(df_meta)):
        date_time = df_meta.index[i]
        ret = df_meta.loc[date_time, 'ret']
        target = df_meta.loc[date_time, 'trgt']

        if ret > 0.0 and ret > target:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and ret < -target:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    df_meta['bin'] = store
    return df_meta


#
# Function get_bins
#

def get_bins(df_events, ds_close):
    r"""Get the dataframe of meta-label events.

    Parameters
    ----------

    df_events : pandas.DataFrame
        The record of Triple Barrier events
        - df_events.index is event's starttime
        - df_events.t1 is the event's endtime
        - df_events.trgt is the event's target
        - df_events.side implies the algorithm's position side
            Case 1: Side Not in Events: bin in (-1, 1) <- label by price action
            Case 2: Side In Events    : bin in ( 0, 1) <- label by P&L (meta-labeling)

    ds_close : pandas.Series
        Array of closing values.

    Returns
    -------

    df_meta : pandas.DataFrame
        The meta-labeled events

    """

    # Align prices with their respective events.

    df_events_ = df_events.dropna(subset=['t1'])
    prices = df_events_.index.union(df_events_['t1'].values)
    prices = prices.drop_duplicates()
    prices = ds_close.reindex(prices, method='bfill')

    # Create the output dataframe.
    df_meta = pd.DataFrame(index=df_events_.index)

    # Calculate the log returns, else the results will be skewed for short positions.

    df_meta['ret'] = np.log(prices.loc[df_events_['t1'].values].values) - np.log(prices.loc[df_events_.index])
    df_meta['trgt'] = df_events_['trgt']

    # Meta Labeling: Events that are correct will have positive returns.

    if 'side' in df_events_:
        df_meta['ret'] = df_meta['ret'] * df_events_['side']

    # Meta Labeling: Apply label 0 when the vertical barrier is reached.
    df_meta = barrier_touched(df_meta)

    # Meta Labeling: Label incorrect events with label 0.

    if 'side' in df_events_:
        df_meta.loc[df_meta['ret'] <= 0, 'bin'] = 0

    # Transform the log returns back to normal returns.
    df_meta['ret'] = np.exp(df_meta['ret']) - 1

    # Add the side to the output. This is used when a meta-label model must be fit.

    tb_cols = df_events.columns
    if 'side' in tb_cols:
        df_meta['side'] = df_events['side']

    return df_meta
