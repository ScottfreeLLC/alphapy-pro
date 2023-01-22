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

from alphapy.globals import SSEP
from alphapy.transforms import higher
from alphapy.transforms import lower
from alphapy.transforms import ma
from alphapy.transforms import net
from alphapy.transforms import streak
from alphapy.transforms import truerange


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_pivot
#

def get_pivot(row, c1, c2):
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
        pivot_str = 'H' + str(row[c1])
    elif row[c1] < row[c2]:
        pivot_str = 'L' + str(row[c2])
    else:
        pivot_str = 'T0'
    return pivot_str


#
# Function encode_pivot
#

def encode_pivot(df_price, c='close', p=20):
    r"""Encode the price data into NLP sequences.

    Parameters
    ----------
    df_price : pandas.DataFrame
        Dataframe with all columns required for calculation.
    p : int
        Maximum period for recording extremes

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

    # Calculate the high pivots.

    col_name = 'higher'
    df_price[col_name] = higher(df_price, 'high')
    df_price['pivot_high'] = streak(df_price, col_name, p)

    # Calculate the low pivots.

    col_name = 'lower'
    df_price[col_name] = lower(df_price, 'low')
    df_price['pivot_low'] = streak(df_price, col_name, p)

    # Encode the pivot value.
    df_price['pivot_str'] = df_price.apply(encode_pivot, args=['pivot_high', 'pivot_low'], axis=1)

    # Encode the net price.

    df_price['net'] = net(df_price)
    df_price['net_str'] = df_price.apply(encode_net, args=['net'], axis=1)

    # Encode the range.

    df_price['atr'] = ma(truerange(df_price), c, p)
    df_price['range'] = df_price['high'] - df_price['low']
    df_price['range_str'] = df_price.apply(encode_range, args=['atr', 'range'], axis=1)

    # Encode the volume.

    df_price['atr_volume'] = ma(truerange(df_price), 'volume', p)
    df_price['volume_str'] = df_price.apply(encode_volume, args=['atr_volume'], axis=1)

    # Create the encoded string.

    df_price['encoded_str'] = df_price['pivot_str'] + df_price['net_str'] + df_price['range_str'] + df_price['volume_str']
    encoded_values = df_price['encoded_str'].values.tolist()
    encoded_str = SSEP.join(encoded_values)

    return encoded_str
