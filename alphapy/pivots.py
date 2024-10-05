"""
Package   : AlphaPy
Module    : pivots.py
Created   : September 14, 2024

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
import pandas as pd
import plotly.graph_objects as go

from alphapy.globals import PivotType


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function pivothigh
#

def pivothigh(df_pivots, start, end, strength):
    """
    Identify a pivot high in the given range with the required strength.

    Parameters
    ----------
    df_pivots : pandas.DataFrame
        DataFrame containing pivot points and their properties.
    start : int
        Start index for checking pivots.
    end : int
        End index for checking pivots.
    strength : int
        Minimum strength required for pivots.

    Returns
    -------
    int
        Index of the identified pivot high, or -1 if not found.
    """

    # Find the highest pivot point with the required strength
    for i in range(start, end + 1):
        if df_pivots.iloc[i]['pivot_type'] == PivotType.PivotHigh.value and df_pivots.iloc[i]['pivot_strength'] >= strength:
            return i
    return -1


#
# Function pivotlow
#

def pivotlow(df_pivots, start, end, strength):
    """
    Identify a pivot low in the given range with the required strength.

    Parameters
    ----------
    df_pivots : pandas.DataFrame
        DataFrame containing pivot points and their properties.
    start : int
        Start index for checking pivots.
    end : int
        End index for checking pivots.
    strength : int
        Minimum strength required for pivots.

    Returns
    -------
    int
        Index of the identified pivot low, or -1 if not found.
    """

    # Find the lowest pivot point with the required strength
    for i in range(start, end + 1):
        if df_pivots.iloc[i]['pivot_type'] == PivotType.PivotLow.value and df_pivots.iloc[i]['pivot_strength'] >= strength:
            return i
    return -1


#
# Function pivotmap
#

def pivotmap(df, window=100, min_strength=5):
    """
    Find the pivots in a given window and calculate their strength.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data with 'date', 'high', 'low', and 'close' columns.
    window : int
        The number of bars to consider in the window.
    min_strength : int
        The minimum pivot strength in the window.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the identified pivots and their properties (date, bar_index, pivot_type, pivot_strength, close).
    """

    # Initialize the pivot map list
    pm = []
    df_pivot = pd.DataFrame()

    # Ensure there is enough data to form a window
    if window >= 3 and min_strength >= 1 and len(df) >= window:
        # Loop through each bar in the window from end to start
        for ib in range(len(df) - window, len(df)):
            high_strength = 1
            low_strength = 1

            # Check for Pivot High
            for offset in range(1, min(window, ib + 1)):
                if df['high'].iloc[ib] > df['high'].iloc[ib - offset]:
                    high_strength += 1
                else:
                    break  # stop counting when a higher high is not found

            # Check for Pivot Low
            for offset in range(1, min(window, ib + 1)):
                if df['low'].iloc[ib] < df['low'].iloc[ib - offset]:
                    low_strength += 1
                else:
                    break  # stop counting when a lower low is not found

            # Determine pivot type and add to the pivot map
            if high_strength >= min_strength:
                pm.append([df['date'].iloc[ib],
                           len(df) - 1 - ib,
                           PivotType.PivotHigh,
                           high_strength,
                           df['close'].iloc[ib]])
            elif low_strength >= min_strength:
                pm.append([df['date'].iloc[ib],
                           len(df) - 1 - ib,
                           PivotType.PivotLow,
                           low_strength,
                           df['close'].iloc[ib]])

    # Convert pivot map to DataFrame
    df_pivot = pd.DataFrame(pm, columns=['date', 'bar_index', 'pivot_type', 'pivot_strength', 'close'])
    return df_pivot


#
# Function plot_pivot_map
#

def plot_pivot_map(df, df_pivots):
    """
    Visualize the price data with pivot points using a candlestick chart in Plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the price data with 'open', 'high', 'low', and 'close' columns.
    df_pivots : pandas.DataFrame
        DataFrame containing pivot points with 'bar_index', 'pivot_type', and 'pivot_strength'.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure showing the candlestick chart with pivot points.
    """

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name='Candlestick')])

    # Add the pivot points
    for _, row in df_pivots.iterrows():
        if row['pivot_type'] == PivotType.PivotHigh.value:
            fig.add_trace(go.Scatter(x=[row['bar_index']], y=[df['high'].iloc[row['bar_index']]],
                                     mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='PivotHigh'))
        elif row['pivot_type'] == PivotType.PivotLow.value:
            fig.add_trace(go.Scatter(x=[row['bar_index']], y=[df['low'].iloc[row['bar_index']]],
                                     mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='PivotLow'))

    # Update layout
    fig.update_layout(
        title="Candlestick Chart with Pivot Points",
        xaxis_title="Bar Index",
        yaxis_title="Price",
        legend_title="Legend",
    )

    # Show the plot
    fig.show()
