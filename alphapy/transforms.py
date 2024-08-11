"""
Package   : AlphaPy
Module    : transforms
Created   : March 14, 2020

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

import itertools
import logging
import numpy as np
import pandas as pd

from alphapy.calendrical import biz_day_month
from alphapy.calendrical import biz_day_week
from alphapy.calendrical import get_rdate
from alphapy.globals import NULLTEXT
from alphapy.globals import BSEP, USEP
from alphapy.variables import vexec


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function adx
#

def adx(df, p=14):
    r"""Calculate the Average Directional Index (ADX).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with all columns required for calculation. If you
        are applying ADX through ``vapply``, then these columns are
        calculated automatically.
    p : int
        The period over which to calculate the ADX.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    The Average Directional Movement Index (ADX) was invented by J. Welles
    Wilder in 1978 [WIKI_ADX]_.  Its value reflects the strength of trend in any
    given instrument.

    .. [WIKI_ADX] https://en.wikipedia.org/wiki/Average_directional_movement_index

    """
    c1 = 'diplus'
    df = vexec(df, c1)
    c2 = 'diminus'
    df = vexec(df, c2)
    # calculations
    dip = df[c1]
    dim = df[c2]
    didiff = abs(dip - dim)
    disum = dip + dim
    new_column = 100 * didiff.ewm(span=p).mean() / disum
    return new_column


#
# Function bbands
#

def bbands(df, c='close', p=20, sd=2.0, low_band=True):
    r"""Calculate the Bollinger Bands.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Simple Moving Average.
    sd : float
        The number of standard deviations.
    low_band : bool, optional
        If set to True, then calculate the lower band, else the upper band.

    Returns
    -------
    bband : pandas.Series
        The series for the selected Bollinger Band.
    """

    sma = ma(df, c, p)
    if low_band:
        bband = sma - sd * df[c].rolling(p).std(ddof=0)
    else:
        bband = sma + sd * df[c].rolling(p).std(ddof=0)
    return bband


#
# Function bblower
#

def bblower(df, c='close', p=20, sd=1.5):
    r"""Calculate the lower Bollinger Band.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Simple Moving Average.
    sd : float
        The number of standard deviations.

    Returns
    -------
    lower_band : pandas.Series
        The series containing the lower Bollinger Band.
    """

    lower_band = bbands(df, c, p, sd)
    return lower_band


#
# Function bbupper
#

def bbupper(df, c='close', p=20, sd=1.5):
    r"""Calculate the upper Bollinger Band.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Simple Moving Average.
    sd : float
        The number of standard deviations.

    Returns
    -------
    upper_band : pandas.Series
        The series containing the upper Bollinger Band.
    """

    upper_band = bbands(df, c, p, sd, low_band=False)
    return upper_band


#
# Function bizday
#

def bizday(df, c):
    r"""Extract business day of month and week.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    date_features : pandas.DataFrame
        The dataframe containing the date features.
    """

    date_features = pd.DataFrame()
    try:
        date_features = dateparts(df[c])
        rdate = date_features.apply(get_rdate, axis=1)
        bdm = pd.Series(rdate.apply(biz_day_month), name='bizdaymonth')
        bdw = pd.Series(rdate.apply(biz_day_week), name='bizdayweek')
        frames = [date_features, bdm, bdw]
        date_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract business date information from the series")
    return date_features


#
# Function c2max
#

def c2max(df, c1, c2):
    r"""Take the maximum value between two columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``df``.
    c2 : str
        Name of the second column in the dataframe ``df``.

    Returns
    -------
    max_val : float
        The maximum value of the two columns.

    """
    max_val = max(df[c1], df[c2])
    return max_val


#
# Function c2min
#

def c2min(df, c1, c2):
    r"""Take the minimum value between two columns in a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``df``.
    c2 : str
        Name of the second column in the dataframe ``df``.

    Returns
    -------
    min_val : float
        The minimum value of the two columns.

    """
    min_val = min(df[c1], df[c2])
    return min_val


#
# Function dateparts
#

def dateparts(df, c):
    r"""Extract date into its components: year, month, day, dayofweek.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    date_features : pandas.DataFrame
        The dataframe containing the date features.
    """

    ds_dt = pd.to_datetime(df[c])
    date_features = pd.DataFrame()
    try:
        fyear = pd.Series(ds_dt.dt.year, name='year').astype(int)
        fmonth = pd.Series(ds_dt.dt.month, name='month').astype(int)
        fday = pd.Series(ds_dt.dt.day, name='day').astype(int)
        fdow = pd.Series(ds_dt.dt.dayofweek, name='dayofweek').astype(int)
        frames = [fyear, fmonth, fday, fdow]
        date_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract date information from the series")
    return date_features


#
# Function diff
#

def diff(df, c, n=1):
    r"""Calculate the n-th order difference for the given variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    n : int
        The number of times that the values are differenced.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.
    """
    new_column = np.diff(df[c], n)
    return new_column


#
# Function diminus
#

def diminus(df, p=14):
    r"""Calculate the Minus Directional Indicator (-DI).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the -DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of a downtrend. When the -DI is sloping downward,
    it is a signal that the downtrend is getting stronger* [IP_NDI]_.

    .. [IP_NDI] http://www.investopedia.com/terms/n/negativedirectionalindicator.asp

    """
    tr = 'truerange'
    df = vexec(df, tr)
    atr = USEP.join(['atr', str(p)])
    df = vexec(df, atr)
    dmm = 'dmminus'
    df[dmm] = dminus(df)
    new_column = 100 * dminus(df).ewm(span=p).mean() / df[atr]
    return new_column


#
# Function diplus
#

def diplus(df, p=14):
    r"""Calculate the Plus Directional Indicator (+DI).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which to calculate the +DI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A component of the average directional index (ADX) that is used to
    measure the presence of an uptrend. When the +DI is sloping upward,
    it is a signal that the uptrend is getting stronger* [IP_PDI]_.

    .. [IP_PDI] http://www.investopedia.com/terms/p/positivedirectionalindicator.asp

    """
    tr = 'truerange'
    df = vexec(df, tr)
    atr = USEP.join(['atr', str(p)])
    df = vexec(df, atr)
    dmp = 'dmplus'
    df = vexec(df, dmp)
    new_column = 100 * df[dmp].ewm(span=p).mean() / df[atr]
    return new_column


#
# Function dminus
#

def dminus(df):
    r"""Calculate the Minus Directional Movement (-DM).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with high and low columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is negative (minus) when the prior low minus
    the current low is greater than the current high minus the prior high.
    This so-called Minus Directional Movement (-DM) equals the prior low
    minus the current low, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    """
    c1 = 'downmove'
    df[c1] = -net(df, 'low')
    c2 = 'upmove'
    df[c2] = net(df, 'high')
    new_column = df.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function dmplus
#

def dmplus(df):
    r"""Calculate the Plus Directional Movement (+DM).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with high and low columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Directional movement is positive (plus) when the current high minus
    the prior high is greater than the prior low minus the current low.
    This so-called Plus Directional Movement (+DM) then equals the current
    high minus the prior high, provided it is positive. A negative value
    would simply be entered as zero* [SC_ADX]_.

    .. [SC_ADX] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

    """
    c1 = 'upmove'
    df[c1] = net(df, 'high')
    c2 = 'downmove'
    df[c2] = -net(df, 'low')
    new_column = df.apply(gtval0, axis=1, args=[c1, c2])
    return new_column


#
# Function ema
#

def ema(df, c, p=20):
    r"""Calculate the Exponential Moving Average (EMA) on a rolling basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average (EMA).

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *An exponential moving average (EMA) is a type of moving average
    that is similar to a simple moving average, except that more weight
    is given to the latest data* [IP_EMA]_.

    .. [IP_EMA] http://www.investopedia.com/terms/e/ema.asp

    """
    new_column = df[c].ewm(span=p).mean()
    return new_column


#
# Function gap
#

def gap(df):
    r"""Calculate the gap percentage between the current open and
    the previous close.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with open and close columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    .. [IP_GAP] http://www.investopedia.com/terms/g/gap.asp

    """
    c1 = ''.join(['close', '[1]'])
    df = vexec(df, c1)
    new_column = 100 * pchange2(df, 'open', c1)
    return new_column


#
# Function gapbadown
#

def gapbadown(df):
    r"""Determine whether or not there has been a breakaway gap down.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with open and low columns.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    .. [IP_BAGAP] http://www.investopedia.com/terms/b/breakawaygap.asp

    """
    new_column = df['open'] < df['low'].shift(1)
    return new_column


#
# Function gapbaup
#

def gapbaup(df):
    r"""Determine whether or not there has been a breakaway gap up.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with open and high columns.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A breakaway gap represents a gap in the movement of a stock price
    supported by levels of high volume* [IP_BAGAP]_.

    """
    new_column = df['open'] > df['high'].shift(1)
    return new_column


#
# Function gapdown
#

def gapdown(df):
    r"""Determine whether or not there has been a gap down.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with open and close columns.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = df['open'] < df['close'].shift(1)
    return new_column


#
# Function gapup
#

def gapup(df):
    r"""Determine whether or not there has been a gap up.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with open and close columns.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *A gap is a break between prices on a chart that occurs when the
    price of a stock makes a sharp move up or down with no trading
    occurring in between* [IP_GAP]_.

    """
    new_column = df['open'] > df['close'].shift(1)
    return new_column


#
# Function gtval
#

def gtval(df, c1, c2):
    r"""Determine whether or not the first column of a dataframe
    is greater than the second.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``df``.
    c2 : str
        Name of the second column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c1] > df[c2]
    return new_column


#
# Function gtval0
#

def gtval0(df, c1, c2):
    r"""For positive values in the first column of the dataframe
    that are greater than the second column, get the value in
    the first column, otherwise return zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``df``.
    c2 : str
        Name of the second column in the dataframe ``df``.

    Returns
    -------
    new_val : float
        A positive value or zero.

    """
    if df[c1] > df[c2] and df[c1] > 0:
        new_val = df[c1]
    else:
        new_val = 0
    return new_val


#
# Function haclose
#

def haclose(df):
    r"""Calculate the Heikin-Ashi Close.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with OHLC columns.

    Returns
    -------
    haclose_ds : pandas.Series
        The series containing the Heikin-Ashi Close.

    """
    haclose_ds = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    return haclose_ds


#
# Function hahigh
#

def hahigh(df):
    r"""Calculate the Heikin-Ashi High.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with OHLC columns.

    Returns
    -------
    hahigh_ds : pandas.Series
        The series containing the Heikin-Ashi High.

    """
    hahigh_ds = pd.DataFrame([df['high'], haopen(df), haclose(df)]).max(axis=0)
    return hahigh_ds


#
# Function halow
#

def halow(df):
    r"""Calculate the Heikin-Ashi Low.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with OHLC columns.

    Returns
    -------
    halow_ds : pandas.Series
        The series containing the Heikin-Ashi Low.

    """
    halow_ds = pd.DataFrame([df['low'], haopen(df), haclose(df)]).min(axis=0)
    return halow_ds


#
# Function haopen
#

def haopen(df):
    r"""Calculate the Heikin-Ashi Open.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with OHLC columns.

    Returns
    -------
    haopen : pandas.Series
        The series containing the Heikin-Ashi Open.

    """
    s1 = haclose(df)
    s2 = (df['open'] + df['close']) / 2.0
    dfha = pd.concat([s1, s2], axis=1)
    dfha.columns = ['haclose', 'haopen']
    for i in range(1, len(dfha)):
        dfha.iloc[i]['haopen'] = (dfha.iloc[i-1]['haopen'] + dfha.iloc[i-1]['haclose']) / 2.0
    return dfha['haopen']


#
# Function higher
#

def higher(df, c, o=1):
    r"""Determine whether or not a series value is higher than
    the value ``o`` periods back.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] > df[c].shift(o)
    return new_column


#
# Function highest
#

def highest(df, c, p=20):
    r"""Calculate the highest value on a rolling basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the rolling maximum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = df[c].rolling(p).max()
    return new_column


#
# Function hlrange
#

def hlrange(df, p=1):
    r"""Calculate the Range, the difference between High and Low.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns ``high`` and ``low``.
    p : int
        The period over which the range is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = highest(df, 'high', p) - lowest(df, 'low', p)
    return new_column


#
# Function keltner
#

def keltner(df, c='close', p=20, atrs=1.5, channel='midline'):
    r"""Calculate the Keltner Channels.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    kc : pandas.Series
        The series containing the Keltner Channel.
    """

    ds_ema = ema(df, c, p)
    atr = USEP.join(['atr', str(p)])
    df = vexec(df, atr)
    if channel == 'lower':
        kc = ds_ema - df[atr].apply(lambda x: atrs * x)
    elif channel == 'upper':
        kc = ds_ema + df[atr].apply(lambda x: atrs * x)
    else:
        kc = ds_ema
    return kc


#
# Function keltnerlb
#

def keltnerlb(df, c='close', p=20, atrs=1.5):
    r"""Calculate the lower Keltner Channel.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    kclb : pandas.Series
        The series containing the lower Keltner Channel.
    """

    kclb = keltner(df, c, p, atrs, channel='lower')
    return kclb


#
# Function keltnerml
#

def keltnerml(df, c='close', p=20, atrs=1.5):
    r"""Calculate the midline Keltner Channel.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    kcml : pandas.Series
        The series containing the midline Keltner Channel.
    """

    kcml = keltner(df, c, p, atrs)
    return kcml


#
# Function keltnerub
#

def keltnerub(df, c='close', p=20, atrs=1.5):
    r"""Calculate the upper Keltner Channel.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    kcub : pandas.Series
        The series containing the upper Keltner Channel.
    """

    kcub = keltner(df, c, p, atrs, channel='upper')
    return kcub


#
# Function lower
#

def lower(df, c, o=1):
    r"""Determine whether or not a series value is lower than
    the value ``o`` periods back.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] < df[c].shift(o)
    return new_column


#
# Function lowest
#

def lowest(df, c, p=20):
    r"""Calculate the lowest value on a rolling basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the rolling minimum.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    return df[c].rolling(p).min()


#
# Function ma
#

def ma(df, c='close', p=20):
    r"""Calculate the mean on a rolling basis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the rolling mean.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *In statistics, a moving average (rolling average or running average)
    is a calculation to analyze data points by creating series of averages
    of different subsets of the full data set* [WIKI_MA]_.

    .. [WIKI_MA] https://en.wikipedia.org/wiki/Moving_average

    """
    new_column = df[c].rolling(p).mean()
    return new_column


#
# Function maabove
#

def maabove(df, c, p=50):
    r"""Determine those values of the dataframe that are above the
    moving average.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] > ma(df, c, p)
    return new_column


#
# Function mabelow
#

def mabelow(df, c, p=50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period of the moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] < ma(df, c, p)
    return new_column


#
# Function maratio
#

def maratio(df, c, p1=1, p2=10):
    r"""Calculate the ratio of two moving averages.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p1 : int
        The period of the first moving average.
    p2 : int
        The period of the second moving average.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = ma(df, c, p1) / ma(df, c, p2)
    return new_column


#
# Function negval0
#

def negval0(df, c):
    r"""Get the negative value, otherwise zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (float)
        Negative value or zero.

    """
    new_column = df[c].apply(lambda x: -x if x < 0 else 0)
    return new_column


#
# Function negvals
#

def negvals(df, c):
    r"""Find the negative values in the series.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] < 0
    return new_column


#
# Function net
#

def net(df, c='close', o=1):
    r"""Calculate the net change of a given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Net change is the difference between the closing price of a security
    on the day's trading and the previous day's closing price. Net change
    can be positive or negative and is quoted in terms of dollars* [IP_NET]_.

    .. [IP_NET] http://www.investopedia.com/terms/n/netchange.asp

    """
    new_column = df[c] - df[c].shift(o)
    return new_column


#
# Function netreturn
#

def netreturn(df, c, o=1):
    r"""Calculate the net return, or Return On Invesment (ROI)

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    o : int, optional
        Offset value for shifting the series.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *ROI measures the amount of return on an investment relative to the
    original cost. To calculate ROI, the benefit (or return) of an
    investment is divided by the cost of the investment, and the result
    is expressed as a percentage or a ratio* [IP_ROI]_.

    .. [IP_ROI] http://www.investopedia.com/terms/r/returnoninvestment.asp

    """
    new_column = 100 * pchange1(df, c, o)
    return new_column


#
# Function pchange
#

def pchange(df, c, o=1):
    r"""Calculate the percentage change within the same variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    o : int
        Offset to the previous value.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = df[c] / df[c].shift(o) - 1.0
    return new_column


#
# Function pchange2
#

def pchange2(df, c1, c2):
    r"""Calculate the percentage change between two variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``df``.
    c2 : str
        Name of the second column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    new_column = df[c1] / df[c2] - 1.0
    return new_column


#
# Function pivothigh
#

def pivothigh(df, c, p=20):
    r"""Find the pivot high values in the series.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        Period of the series.

    Returns
    -------
    ds_pivot_highs : pd.Series (int)
        The array of pivot high values.

    """

    def get_pivot_high(ds, max_len):
        ds_len = min(len(ds), max_len)
        pivot = 1
        if ds_len == 1:
            return pivot
        else:
            ds_pivot = ds.iloc[-ds_len:].reset_index(drop=True)
            value_high = ds_pivot.iloc[-1]
            for i in reversed(range(ds_len-1)):
                if value_high > ds_pivot[i]:
                    pivot += 1
                else:
                    break
            return pivot

    ds = df[c]
    ds_pivot_highs = ds.expanding().apply(get_pivot_high, args=(p,))
    return ds_pivot_highs


#
# Function pivotlow
#

def pivotlow(df, c, p=20):
    r"""Find the pivot low values in the series.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        Period of the series.

    Returns
    -------
    ds_pivot_lows : pd.Series (int)
        The array of pivot low values.

    """

    def get_pivot_low(ds, max_len):
        ds_len = min(len(ds), max_len)
        pivot = 1
        if ds_len == 1:
            return pivot
        else:
            ds_pivot = ds.iloc[-ds_len:].reset_index(drop=True)
            value_low = ds.iloc[-1]
            for i in reversed(range(ds_len-1)):
                if value_low < ds_pivot[i]:
                    pivot += 1
                else:
                    break
            return pivot

    ds = df[c]
    ds_pivot_lows = ds.expanding().apply(get_pivot_low, args=(p,))
    return ds_pivot_lows


#
# Function posvals
#

def posvals(df, c):
    r"""Find the positive values in the series.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    """
    new_column = df[c] > 0
    return new_column


#
# Function posval0
#

def posval0(df, c):
    r"""Get the positive value, otherwise zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    new_column : pandas.Series (float)
        Positive value or zero.

    """
    new_column = df[c].apply(lambda x: x if x > 0 else 0)
    return new_column


#
# Function rindex
#

def rindex(df, ci, ch, cl, p=1):
    r"""Calculate the *range index* spanning a given period ``p``.

    The **range index** is a number between 0 and 100 that
    relates the value of the index column ``ci`` to the
    high column ``ch`` and the low column ``cl``. For example,
    if the low value of the range is 10 and the high value
    is 20, then the range index for a value of 15 would be 50%.
    The range index for 18 would be 80%.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the columns ``ci``, ``ch``, and ``cl``.
    ci : str
        Name of the index column in the dataframe ``df``.
    ch : str
        Name of the high column in the dataframe ``df``.
    cl : str
        Name of the low column in the dataframe ``df``.
    p : int
        The period over which the range index of column ``ci``
        is calculated.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    """
    o = p-1 if df[ci].name == 'open' else 0
    hh = highest(df, ch, p)
    ll = lowest(df, cl, p)
    fn = df[ci].shift(o) - ll
    fd = hh - ll
    new_column = 100 * fn / fd
    return new_column


#
# Function rsi
#

def rsi(df, p=14):
    r"""Calculate the Relative Strength Index (RSI).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``net``.
    p : int
        The period over which to calculate the RSI.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Developed by J. Welles Wilder, the Relative Strength Index (RSI) is a momentum
    oscillator that measures the speed and change of price movements* [SC_RSI]_.

    .. [SC_RSI] http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi

    """
    cdiff = 'net'
    df = vexec(df, cdiff)
    df['pval'] = posval0(df, cdiff)
    df['mval'] = negval0(df, cdiff)
    upcs = ma(df, 'pval', p)
    dpcs = ma(df, 'mval', p)
    new_column = 100 - (100 / (1 + (upcs / dpcs)))
    return new_column


#
# Function rtotal
#

def rtotal(df, c, w):
    r"""Calculate the running total.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    w : int
        The rolling period.

    Returns
    -------
    running_total : int
        The final running total.

    Example
    -------

    >>> rtotal(df, c, 20))

    """
    ds = df[c]
    running_total = ds.rolling(window=w).apply(np.count_nonzero)
    return running_total.fillna(0).astype(int)


#
# Function runs
#

def runs(df, c, w):
    r"""Calculate the total number of runs.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    w : int
        The rolling period.

    Returns
    -------
    runs_value : int
        The total number of runs.

    Example
    -------

    >>> runs(df, c, 20)

    """
    ds = df[c]
    runs_value = ds.rolling(window=w).apply(lambda x: len(list(itertools.groupby(x))))
    return runs_value.fillna(0).astype(int)


#
# Function runstest
#

def runstest(df, c, wfuncs, w):
    r"""Perform a runs test on binary series.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    wfuncs : list
        The set of runs test functions to apply to the column:

        ``'all'``:
            Run all of the functions below.
        ``'rtotal'``:
            The running total over the ``window`` period.
        ``'runs'``:
            Total number of runs in ``window``.
        ``'streak'``:
            The length of the latest streak.
        ``'zscore'``:
            The Z-Score over the ``window`` period.
    w : int
        The rolling period.

    Returns
    -------
    new_features : pandas.DataFrame
        The dataframe containing the runs test features.

    References
    ----------
    For more information about runs tests for detecting non-randomness,
    refer to [RUNS]_.

    .. [RUNS] http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm

    """

    all_funcs = {'runs'   : runs,
                 'streak' : streak,
                 'rtotal' : rtotal,
                 'zscore' : zscore}
    # use all functions
    if 'all' in wfuncs:
        wfuncs = list(all_funcs.keys())
    # apply each of the runs functions
    new_features = pd.DataFrame()
    for wf in wfuncs:
        if wf in all_funcs:
            new_feature = all_funcs[wf](df, c, w)
            new_feature.fillna(0, inplace=True)
            new_column_name = USEP.join([wf, c])
            new_feature = new_feature.rename(new_column_name)
            frames = [new_features, new_feature]
            new_features = pd.concat(frames, axis=1)
        else:
            logger.info("Runs Function %s not found", w)
    return new_features


#
# Function split2letters
#

def split2letters(df, c):
    r"""Separate text into distinct characters.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``df``.

    Returns
    -------
    new_feature : pandas.Series
        The array containing the new feature.

    Example
    -------
    The value 'abc' becomes 'a b c'.

    """
    fc = df[c]
    new_feature = None
    dtype = fc.dtypes
    if dtype == 'object':
        fc.fillna(NULLTEXT, inplace=True)
        maxlen = fc.astype(str).str.len().max()
        if maxlen > 1:
            new_feature = fc.apply(lambda x: BSEP.join(list(x)))
    return new_feature


#
# Function streak
#

def streak(df, c, w):
    r"""Determine the length of the latest streak.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    w : int
        The rolling period.

    Returns
    -------
    latest_streak : int
        The length of the latest streak.

    Example
    -------

    >>> streak(df, c, 20)

    """
    ds = df[c]
    latest_streak = ds.rolling(window=w).apply(lambda x: [len(list(g)) for k, g in itertools.groupby(x)][-1])
    return latest_streak.fillna(0).astype(int)


#
# Function tdseqbuy
#

def tdseqbuy(df, c='close', high='high', low='low'):
    r"""Calculate Tom DeMark's Sequential Buy indicator.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the columns ``c``, ``high``, and ``low``.
    c : str, optional
        Name of the column in the dataframe ``df`` representing the close prices.
    high : str, optional
        Name of the column in the dataframe ``df`` representing the high prices.
    low : str, optional
        Name of the column in the dataframe ``df`` representing the low prices.

    Returns
    -------
    tdbuy : pandas.Series
        The array containing the Sequential Buy count.

    References
    ----------
    *Tom DeMark's Sequential indicator is used to identify a potential reversal
    of the current trend by comparing the closing price to previous closing
    prices over a fixed period* [WIKI_TDSEQ]_.

    .. [WIKI_TDSEQ] https://en.wikipedia.org/wiki/Tom_DeMark_Indicators#TD_Sequential

    """

    # Initialize columns
    tdbuy = pd.Series(0, index=df.index)

    # Calculate the TD Setup
    for i in range(4, len(df)):
        # Buy Setup: Close less than the close 4 bars earlier for 9 consecutive bars
        if df[c].iloc[i] < df[c].iloc[i - 4]:
            tdbuy.iloc[i] = tdbuy.iloc[i - 1] + 1 if tdbuy.iloc[i - 1] < 9 else 0
        else:
            tdbuy.iloc[i] = 0

    return tdbuy


#
# Function tdseqsell
#

def tdseqsell(df, c='close', high='high', low='low'):
    r"""Calculate Tom DeMark's Sequential Sell indicator.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the columns ``c``, ``high``, and ``low``.
    c : str, optional
        Name of the column in the dataframe ``df`` representing the close prices.
    high : str, optional
        Name of the column in the dataframe ``df`` representing the high prices.
    low : str, optional
        Name of the column in the dataframe ``df`` representing the low prices.

    Returns
    -------
    tdsell : pandas.Series
        The array containing the Sequential Sell count.

    References
    ----------
    *Tom DeMark's Sequential indicator is used to identify a potential reversal
    of the current trend by comparing the closing price to previous closing
    prices over a fixed period* [WIKI_TDSEQ]_.

    .. [WIKI_TDSEQ] https://en.wikipedia.org/wiki/Tom_DeMark_Indicators#TD_Sequential

    """

    # Initialize columns
    tdsell = pd.Series(0, index=df.index)

    # Calculate the TD Setup
    for i in range(4, len(df)):
        # Sell Setup: Close greater than the close 4 bars earlier for 9 consecutive bars
        if df[c].iloc[i] > df[c].iloc[i - 4]:
            tdsell.iloc[i] = tdsell.iloc[i - 1] + 1 if tdsell.iloc[i - 1] < 9 else 0
        else:
            tdsell.iloc[i] = 0

    return tdsell


#
# Function texplode
#

def texplode(df, c):
    r"""Get dummy values for a text column.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the text column in the dataframe ``df``.

    Returns
    -------
    dummies : pandas.DataFrame
        The dataframe containing the dummy variables.

    Example
    -------

    This function is useful for columns that appear to
    have separate character codes but are consolidated
    into a single column. Here, the column ``c`` is
    transformed into five dummy variables.

    === === === === === ===
     c  0_a 1_x 1_b 2_x 2_z
    === === === === === ===
    abz   1   0   1   0   1
    abz   1   0   1   0   1
    axx   1   1   0   1   0
    abz   1   0   1   0   1
    axz   1   1   0   0   1
    === === === === === ===

    """
    fc = df[c]
    maxlen = fc.astype(str).str.len().max()
    fc.fillna(maxlen * BSEP, inplace=True)
    fpad = str().join(['{:', BSEP, '>', str(maxlen), '}'])
    fcpad = fc.apply(fpad.format)
    fcex = fcpad.apply(lambda x: pd.Series(list(x)))
    dummies = pd.get_dummies(fcex)
    return dummies


#
# Function timeparts
#

def timeparts(df, c):
    r"""Extract time into its components: hour, minute, second.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.

    Returns
    -------
    time_features : pandas.DataFrame
        The dataframe containing the time features.
    """

    ds_dt = pd.to_datetime(df[c].astype(str), format='%H:%M:%S')
    time_features = pd.DataFrame()
    try:
        fhour = pd.Series(ds_dt.dt.hour, name='hour').astype(int)
        fminute = pd.Series(ds_dt.dt.minute, name='minute').astype(int)
        fsecond = pd.Series(ds_dt.dt.second, name='second').astype(int)
        frames = [fhour, fminute, fsecond]
        time_features = pd.concat(frames, axis=1)
    except:
        logger.info("Could not extract time information from the series")
    return time_features


#
# Function truehigh
#

def truehigh(df):
    r"""Calculate the *True High* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with high and low columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's high, or the previous close, whichever is higher* [TS_TR]_.

    .. [TS_TR] http://help.tradestation.com/09_01/tradestationhelp/charting_definitions/true_range.htm

    """
    l1 = ''.join(['low', '[1]'])
    df = vexec(df, l1)
    new_column = df.apply(c2max, axis=1, args=[l1, 'high'])
    return new_column


#
# Function truelow
#

def truelow(df):
    r"""Calculate the *True Low* value.

    Parameters
    ----------
    f : pandas.DataFrame
        Dataframe with high and low columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *Today's low, or the previous close, whichever is lower* [TS_TR]_.

    """
    h1 = ''.join(['high', '[1]'])
    df = vexec(df, h1)
    new_column = df.apply(c2min, axis=1, args=[h1, 'low'])
    return new_column


#
# Function truerange
#

def truerange(df):
    r"""Calculate the *True Range* value.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with with high and low columns.

    Returns
    -------
    new_column : pandas.Series (float)
        The array containing the new feature.

    References
    ----------
    *True High - True Low* [TS_TR]_.

    """
    new_column = truehigh(df) - truelow(df)
    return new_column


#
# Function ttmsqueeze
#

def ttmsqueeze(df, c='close', p=20):
    r"""Calculate the TTM Squeeze momentum oscillator.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.

    Returns
    -------
    ttmosc : float
        The value of the TTM Squeeze Indicator.
    """

    # calculate Donchian midline    
    hh = highest(df, 'high')
    ll = lowest(df, 'low')
    midp = (hh + ll) / 2.0
    # calculate the Simple Moving Average
    sma = ma(df)
    # calculate the delta between the midline and SMA
    delta = (df[c] - (midp + sma) / 2.0)
    # linear regression
    fit_y = np.array(range(0, p))
    ttmosc = delta.rolling(window = p).apply(lambda x:
                            np.polyfit(fit_y, x, 1)[0] * (p-1) +
                            np.polyfit(fit_y, x, 1)[1], raw=True)
    return ttmosc


#
# Function ttmsqueezelong
#

def ttmsqueezelong(df, c='close', p=20, sd=2.0, atrs=1.5):
    r"""Signal a TTM Squeeze Long Entry.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    sd : float
        The number of standard deviations.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    squeezelong : bool
        True if there is a TTM Squeeze Long Entry.
    """

    squeeze_off = ttmsqueezeoff(df, c, p, sd, atrs)
    ttm_squeeze = ttmsqueeze(df, c, p)
    squeeze1 = squeeze_off.shift(1).fillna(False)
    squeeze2 = squeeze_off.shift(2).fillna(False)
    long_cond1 = np.logical_and(squeeze1, ~squeeze2)
    long_cond2 = np.greater(ttm_squeeze, 0)
    squeezelong = np.logical_and(long_cond1, long_cond2)
    return squeezelong


#
# Function ttmsqueezeoff
#

def ttmsqueezeoff(df, c='close', p=20, sd=2.0, atrs=1.5):
    r"""Determine the TTM Squeeze Off condition.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    sd : float
        The number of standard deviations.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    squeezeoff : bool
        The status of the TTM Squeeze Off Indicator.
    """

    kclb = keltner(df, c, p, atrs, channel='lower')
    kcub = keltner(df, c, p, atrs, channel='upper')
    bblb = bbands(df, c, p, sd)
    bbub = bbands(df, c, p, sd, low_band=False)
    squeezeoff = np.logical_and(np.greater(bbub, kcub), np.less(bblb, kclb))
    return squeezeoff


#
# Function ttmsqueezeon
#

def ttmsqueezeon(df, c='close', p=20, sd=2.0, atrs=1.5):
    r"""Determine the TTM Squeeze On condition.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    sd : float
        The number of standard deviations.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    squeezeon : bool
        The status of the TTM Squeeze On Indicator.
    """

    kclb = keltner(df, c, p, atrs, channel='lower')
    kcub = keltner(df, c, p, atrs, channel='upper')
    bblb = bbands(df, c, p, sd)
    bbub = bbands(df, c, p, sd, low_band=False)
    squeezeon = np.logical_and(np.less(bbub, kcub), np.greater(bblb, kclb))
    return squeezeon


#
# Function ttmsqueezeshort
#

def ttmsqueezeshort(df, c='close', p=20, sd=2.0, atrs=1.5):
    r"""Signal a TTM Squeeze Short Entry.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    p : int
        The period over which to calculate the Exponential Moving Average.
    sd : float
        The number of standard deviations.
    atrs : float
        The multiple of Average True Range.

    Returns
    -------
    squeezeshort : bool
        True if there is a TTM Squeeze Short Entry.
    """

    squeeze_off = ttmsqueezeoff(df, c, p, sd, atrs)
    ttm_squeeze = ttmsqueeze(df, c, p)

    squeeze1 = squeeze_off.shift(1).fillna(False)
    squeeze2 = squeeze_off.shift(2).fillna(False)
    short_cond1 = np.logical_and(squeeze1, ~squeeze2)
    short_cond2 = np.less(ttm_squeeze, 0)
    squeezeshort = np.logical_and(short_cond1, short_cond2)
    return squeezeshort


#
# Function vwap
#

def vwap(df, c='close', granularity='day', anchor_dates=None):
    """
    Adjusted VWAP calculation using Unix timestamps for compatibility with np.digitize.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    granularity : str
        The calendrical period over which to calculate VWAP.
    anchor_dates : list
        The set of dates over which to calculate VWAP.

    Returns
    -------
    vwap_value : float
        The calculated Volume-Weighted Average Price (VWAP).
    """

    # Ensure the index is in datetime format
    df.index = pd.to_datetime(df.index)

    if not anchor_dates:
        # Generate anchor dates automatically based on granularity
        if granularity == 'day':
            anchor_dates = pd.Series(df.index.normalize()).unique()
        elif granularity == 'week':
            anchor_dates = (df.index - pd.to_timedelta(df.index.dayofweek, unit='d')).normalize().unique()
        elif granularity == 'month':
            anchor_dates = df.index.to_period('M').to_timestamp().normalize().unique()
        elif granularity == 'quarter':
            anchor_dates = df.index.to_period('Q').to_timestamp().normalize().unique()
    else:
        anchor_dates = pd.to_datetime(anchor_dates).normalize()
    
    # Convert datetime index and anchor_dates to Unix timestamps for np.digitize
    denominator = 10**9
    unix_index = df.index.astype(np.int64) // denominator
    unix_anchor_dates = anchor_dates.astype(np.int64) // denominator

    # Assign periods based on Unix timestamps
    df['period'] = np.digitize(unix_index, bins=unix_anchor_dates, right=False)

    # Calculate VWAP
    df['dollar_volume'] = df[c] * df['volume']
    grouped = df.groupby('period')
    df['cumulative_dollar_volume'] = grouped['dollar_volume'].cumsum()
    df['cumulative_volume'] = grouped['volume'].cumsum()
    df['vwap'] = df['cumulative_dollar_volume'] / df['cumulative_volume']

    # Clean up by dropping temporary columns and keeping the original datetime index
    vwap_value = df[['vwap']].copy()
    df.drop(['period', 'dollar_volume', 'cumulative_dollar_volume', 'cumulative_volume', 'vwap'],
            axis=1, inplace=True)

    return vwap_value


#
# Function xmadown
#

def xmadown(df, c='close', pfast=20, pslow=50):
    r"""Determine those values of the dataframe that cross below the
    moving average.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``df``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    .. [WIKI_XMA] https://en.wikipedia.org/wiki/Moving_average_crossover

    """
    sma = ma(df, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(df, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma < lma) & (sma_prev > lma_prev)
    return new_column


#
# Function xmaup
#

def xmaup(df, c='close', pfast=20, pslow=50):
    r"""Determine those values of the dataframe that are below the
    moving average.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str, optional
        Name of the column in the dataframe ``df``.
    pfast : int, optional
        The period of the fast moving average.
    pslow : int, optional
        The period of the slow moving average.

    Returns
    -------
    new_column : pandas.Series (bool)
        The array containing the new feature.

    References
    ----------
    *In the statistics of time series, and in particular the analysis
    of financial time series for stock trading purposes, a moving-average
    crossover occurs when, on plotting two moving averages each based
    on different degrees of smoothing, the traces of these moving averages
    cross* [WIKI_XMA]_.

    """
    sma = ma(df, c, pfast)
    sma_prev = sma.shift(1)
    lma = ma(df, c, pslow)
    lma_prev = lma.shift(1)
    new_column = (sma > lma) & (sma_prev < lma_prev)
    return new_column


#
# Function zscore
#

def zscore(df, c, w):
    r"""Calculate the Z-Score.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the column ``c``.
    c : str
        Name of the column in the dataframe ``df``.
    w : int
        The rolling period.

    Returns
    -------
    zscore : float
        The value of the Z-Score.

    References
    ----------
    To calculate the Z-Score, you can find more information here [ZSCORE]_.

    .. [ZSCORE] https://en.wikipedia.org/wiki/Standard_score

    Example
    -------

    >>> zscore(f, c, 20)

    """
    ds = df[c]
    r = ds.rolling(window=w)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    zscore = (ds - m) / s
    return zscore
