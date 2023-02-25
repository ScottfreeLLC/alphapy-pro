"""
Package   : AlphaPy
Module    : nlp
Created   : January 14, 2023

Copyright 2023 ScottFree Analytics LLC
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
import math

from alphapy.globals import BSEP
from alphapy.transforms import ma
from alphapy.transforms import net
from alphapy.transforms import pivot_high
from alphapy.transforms import pivot_low


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function encode_pivot
#

def encode_pivot(row, c1, c2):
    r"""Encode the strongest pivot value, H or L.

    Parameters
    ----------
    row : pandas.DataFrame
        Row of the dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    pivot_str : str
        The encoded pivot string.

    """

    if row[c1] > row[c2]:
        pivot_str = 'H' + str(int(row[c1]))
    elif row[c2] > row[c1]:
        pivot_str = 'L' + str(int(row[c2]))
    else:
        pivot_str = 'T0'
    return pivot_str


#
# Function encode_net
#

def encode_net(row, c1, c2):
    r"""Encode the net change value, P or N.

    Parameters
    ----------
    row : pandas.DataFrame
        Row of the dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    net_str : str
        The encoded net change string.

    """
    if row[c1] > 0:
        try:
            value = min(row[c1] // row[c2], 2)
            value = 0 if math.isnan(value) else value
        except:
            value = 0
        cat = 'P'
    elif row[c1] < 0:
        try:
            value = min(abs((row[c1] // row[c2]) + 1), 2)
            value = 0 if math.isnan(value) else value
        except:
            value = 0
        cat = 'N'
    else:
        value = 0
        cat = 'Z'
    net_str = cat + str(int(value))
    return net_str


#
# Function encode_range
#

def encode_range(row, c1, c2):
    r"""Encode the range value.

    Parameters
    ----------
    row : pandas.DataFrame
        Row of the dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    range_str : str
        The encoded range string.

    """
    try:
        value = min(row[c1] // row[c2], 2)
        value = 0 if math.isnan(value) else value
    except:
        value = 0
    range_str = 'R' + str(int(value))
    return range_str


#
# Function encode_volume
#

def encode_volume(row, c1, c2):
    r"""Encode the strongest pivot value, H or L.

    Parameters
    ----------
    row : pandas.DataFrame
        Row of the dataframe containing the two columns ``c1`` and ``c2``.
    c1 : str
        Name of the first column in the dataframe ``f``.
    c2 : str
        Name of the second column in the dataframe ``f``.

    Returns
    -------
    volume_str : str
        The encoded volume string.

    """
    try:
        value = min(row[c1] // row[c2], 2)
        value = 0 if math.isnan(value) else value
    except:
        value = 0
    volume_str = 'V' + str(int(value))
    return volume_str


#
# Function encode_price
#

def encode_price(df_price, p=20, intraday=None):
    r"""Encode the price data into NLP sequences.

    Parameters
    ----------
    df_price : pandas.DataFrame
        Dataframe with all columns required for calculation.
    p : int
        Maximum period for recording extremes
    intraday : tuple
        Intraday flag

    Returns
    -------
    encoded_str : str
        The string containing the encoded sequences.

    Notes
    -----
    High or Low   : [1..p]
    Plus or Minus : [0, 1, 2]
    Range         : [0, 1, 2]
    Volume        : [0, 1, 2]

    """

    # Encode the pivot value.

    df_price['pivot_high'] = pivot_high(df_price, 'high', p)
    df_price['pivot_low'] = pivot_low(df_price, 'low', p)
    df_price['pivot_str'] = df_price.apply(encode_pivot, args=['pivot_high', 'pivot_low'], axis=1)

    # Encode the net price.

    df_price['net'] = net(df_price)
    df_price['net_abs'] = df_price['net'].abs()
    df_price['net_ma'] = ma(df_price, 'net_abs', p)
    df_price['net_str'] = df_price.apply(encode_net, args=['net', 'net_ma'], axis=1)

    # Encode the range.

    df_price['range'] = df_price['high'] - df_price['low']
    df_price['range_ma'] = ma(df_price, 'range', p)
    df_price['range_str'] = df_price.apply(encode_range, args=['range', 'range_ma'], axis=1)

    # Encode the volume.

    df_price['volume_ma'] = ma(df_price, 'volume', p)
    df_price['volume_str'] = df_price.apply(encode_volume, args=['volume', 'volume_ma'], axis=1)

    # Create and return the encoded string.

    df_price['encoded_str'] = df_price['pivot_str'] + df_price['net_str'] + df_price['range_str'] + df_price['volume_str']
    encoded_str = df_price['encoded_str'].str.cat(sep=BSEP)

    if intraday:
        encoded_str = ' ' + intraday[0] + ' ' + encoded_str + ' ' + intraday[1] + ' '
    return encoded_str
