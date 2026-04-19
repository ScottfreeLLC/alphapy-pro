"""
Package   : AlphaPy
Module    : pivots
Created   : March 14, 2024

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


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function pivothigh
#

def pivothigh(df, window_length, minimum_strength=5):
    """
    Identify PivotHighs in the last n rows of the data using the high column.

    Parameters:
    - df: pandas DataFrame containing 'high' prices.
    - window_length: The number of most recent rows to include in the pivot detection.
    - minimum_strength: The minimum strength of the high pivot.

    Returns:
    - List of dictionaries containing pivot high points sorted by the highest price.
    """

    # Get the last n rows for the detection window
    df_window = df.tail(window_length).reset_index(drop=True)
    pivots = []

    # Vectorized access to 'high' prices for performance
    highs = df_window['high'].values
    n = len(highs)

    # Loop through each row, avoiding the boundary points
    for i in range(1, n - 1):
        current_high = highs[i]
        current_strength = 0

        # Check neighbors within the valid range for pivot strength
        left_values = highs[i - current_strength - 1::-1]
        right_values = highs[i + 1:i + window_length - i]

        for left, right in zip(left_values, right_values):
            if left >= current_high or right >= current_high:
                break
            current_strength += 1

        # Only consider valid pivots with sufficient strength
        if current_strength >= minimum_strength:
            pivot_dict = {
                'pivot_index': i,
                'date': df_window['date'].iloc[i],
                'price': current_high,
                'strength': current_strength
            }
            pivots.append(pivot_dict)

    # Sort pivots by the highest prices (descending)
    return sorted(pivots, key=lambda x: x['price'], reverse=True)


#
# Function pivotlow
#

def pivotlow(df, window_length, minimum_strength=5):
    """
    Identify PivotLows in the last n rows of the data using the low column.

    Parameters:
    - df: pandas DataFrame containing 'low' prices.
    - window_length: The number of most recent rows to include in the pivot detection.
    - minimum_strength: The minimum strength of the low pivot.

    Returns:
    - List of dictionaries containing pivot low points sorted by the lowest price.
    """

    # Get the last n rows for the detection window
    df_window = df.tail(window_length).reset_index(drop=True)
    pivots = []

    # Vectorized access to 'low' prices for performance
    lows = df_window['low'].values
    n = len(lows)

    # Loop through each row, avoiding the boundary points
    for i in range(1, n - 1):
        current_low = lows[i]
        current_strength = 0

        # Check neighbors within the valid range for pivot strength
        left_values = lows[i - current_strength - 1::-1]
        right_values = lows[i + 1:i + window_length - i]

        for left, right in zip(left_values, right_values):
            if left <= current_low or right <= current_low:
                break
            current_strength += 1

        # Only consider valid pivots with sufficient strength
        if current_strength >= minimum_strength:
            pivot_dict = {
                'pivot_index': i,
                'date': df_window['date'].iloc[i],
                'price': current_low,
                'strength': current_strength
            }
            pivots.append(pivot_dict)

    # Sort pivots by the lowest prices (ascending)
    return sorted(pivots, key=lambda x: x['price'])


#
# Function gartley_bullish
#

def gartley_bullish(df, window_length):
    """
    Identify a Bullish Gartley pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Bullish Gartley
    #     X (strongest low)
    #     A (strongest high)
    #     B (second strongest low)
    #     C (second strongest high)
    #     D (between X and B)

    x_bullish = low_pivots[0]['pivot_index']
    x_low = low_pivots[0]['price']
    a_bullish = high_pivots[0]['pivot_index']
    a_high = high_pivots[0]['price']
    b_bullish = low_pivots[1]['pivot_index']
    b_low = low_pivots[1]['price']
    c_bullish = high_pivots[1]['pivot_index']
    c_high = high_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bullish Gartley sequence
    if x_bullish < a_bullish < b_bullish < c_bullish:
        xa_distance_bullish = a_high - x_low
        ab_retracement_bullish = (a_high - b_low) / xa_distance_bullish
        bc_retracement_bullish = (c_high - b_low) / (a_high - b_low)
        cd_extension_bullish = (c_high - d_close) / (c_high - b_low)
        logger.info(f"Bullish Gartley")
        logger.info(f"AB Retracement: {ab_retracement_bullish}")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        logger.info(f"CD Extension: {cd_extension_bullish}")
        ab_condition = 0.618 <= ab_retracement_bullish <= 0.786
        bc_condition = 0.382 <= bc_retracement_bullish <= 0.886
        cd_condition = x_low < d_close < b_low
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Gartley Bullish Pattern"
        else:
            result['result'] = "Gartley Bullish Sequence"
        result['X'] = low_pivots[0]
        result['A'] = high_pivots[0]
        result['B'] = low_pivots[1]
        result['C'] = high_pivots[1]
        result['D'] = d_close

    return result


#
# Function gartley_bearish
#

def gartley_bearish(df, window_length):
    """
    Identify a Bearish Gartley pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result
    
    # Bearish Gartley logic:
    #     X (strongest high)
    #     A (strongest low)
    #     B (second strongest high)
    #     C (second strongest low)
    #     D (between X and B)

    x_bearish = high_pivots[0]['pivot_index']
    x_high = high_pivots[0]['price']
    a_bearish = low_pivots[0]['pivot_index']
    a_low = low_pivots[0]['price']
    b_bearish = high_pivots[1]['pivot_index']
    b_high = high_pivots[1]['price']
    c_bearish = low_pivots[1]['pivot_index']
    c_low = low_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bearish Gartley sequence
    if x_bearish < a_bearish < b_bearish < c_bearish:
        xa_distance_bearish = x_high - a_low
        ab_retracement_bearish = (b_high - a_low) / xa_distance_bearish
        bc_retracement_bearish = (b_high - c_low) / (b_high - a_low)
        cd_extension_bearish = (d_close - c_low) / (b_high - c_low)
        logger.info(f"Bearish Gartley")
        logger.info(f"AB Retracement: {ab_retracement_bearish}")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        logger.info(f"CD Extension: {cd_extension_bearish}")
        ab_condition = 0.618 <= ab_retracement_bearish <= 0.786
        bc_condition = 0.382 <= bc_retracement_bearish <= 0.886
        cd_condition = b_high < d_close < x_high
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Gartley Bearish Pattern"
        else:
            result['result'] = "Gartley Bearish Sequence"
        result['X'] = high_pivots[0]
        result['A'] = low_pivots[0]
        result['B'] = high_pivots[1]
        result['C'] = low_pivots[1]
        result['D'] = d_close

    return result


#
# Function abcd_bullish
#

def abcd_bullish(df, window_length):
    """
    Identify a Bullish ABCD pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Bullish ABCD
    #     A (strongest high)
    #     B (strongest low)
    #     C (second strongest high)
    #     D (lower than B)

    # Calculate pivot points using pivotHigh and pivotLow functions
    n_high_pivots = 2
    high_pivots = pivothigh(df, window_length)
    n_low_pivots = 1
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < n_high_pivots or len(low_pivots) < n_low_pivots:
        logger.info(f"Minimum number of high pivots must be {n_high_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Minimum number of low pivots must be {n_low_pivots}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    a_bullish = high_pivots[0]['pivot_index']
    a_high = high_pivots[0]['price']
    b_bullish = low_pivots[0]['pivot_index']
    b_low = low_pivots[0]['price']
    c_bullish = high_pivots[1]['pivot_index']
    c_high = high_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bullish ABCD sequence
    if a_bullish < b_bullish < c_bullish:
        ab_distance_bullish = a_high - b_low
        bc_retracement_bullish = (c_high - b_low) / ab_distance_bullish
        logger.info(f"Bullish ABCD")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        bc_condition = 0.618 <= bc_retracement_bullish <= 0.786
        cd_condition = d_close < b_low
        if bc_condition and cd_condition:
            result['result'] = "ABCD Bullish Pattern"
        else:
            result['result'] = "ABCD Bullish Sequence"
        result['A'] = high_pivots[0]
        result['B'] = low_pivots[0]
        result['C'] = high_pivots[1]
        result['D'] = d_close

    return result


#
# Function abcd_bearish
#

def abcd_bearish(df, window_length):
    """
    Identify a Bearish ABCD pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)
    
    # Bearish ABCD:
    #     A (strongest low)
    #     B (strongest high)
    #     C (second strongest low)
    #     D (higher than B)

    # Calculate pivot points using pivotHigh and pivotLow functions
    n_high_pivots = 1
    high_pivots = pivothigh(df, window_length)
    n_low_pivots = 2
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < n_high_pivots or len(low_pivots) < n_low_pivots:
        logger.info(f"Minimum number of high pivots must be {n_high_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Minimum number of low pivots must be {n_low_pivots}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    a_bearish = low_pivots[0]['pivot_index']
    a_low = low_pivots[0]['price']
    b_bearish = high_pivots[0]['pivot_index']
    b_high = high_pivots[0]['price']
    c_bearish = low_pivots[1]['pivot_index']
    c_low = low_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bearish ABCD sequence
    if a_bearish < b_bearish < c_bearish:
        ab_distance_bearish = b_high - a_low
        bc_retracement_bearish = (b_high - c_low) / ab_distance_bearish
        logger.info(f"Bearish ABCD")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        bc_condition = 0.618 <= bc_retracement_bearish <= 0.786
        cd_condition = b_high < d_close
        if bc_condition and cd_condition:
            result['result'] = "ABCD Bearish Pattern"
        else:
            result['result'] = "ABCD Bearish Sequence"
        result['A'] = low_pivots[0]
        result['B'] = high_pivots[0]
        result['C'] = low_pivots[1]
        result['D'] = d_close

    return result


#
# Function drive3_bullish
#

def drive3_bullish(df, window_length):
    """
    Identify a Bullish Three-Drive pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Bullish Three-Drive
    #     A (strongest high)
    #     B (second strongest low)
    #     C (second strongest high)
    #     D (strongest low)
    #     E (third strongest high)
    #     F (lower than D)

    # Calculate pivot points using pivotHigh and pivotLow functions
    n_high_pivots = 3
    high_pivots = pivothigh(df, window_length)
    n_low_pivots = 2
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < n_high_pivots or len(low_pivots) < n_low_pivots:
        logger.info(f"Minimum number of high pivots must be {n_high_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Minimum number of low pivots must be {n_low_pivots}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    a_bullish = high_pivots[0]['pivot_index']
    a_high = high_pivots[0]['price']
    b_bullish = low_pivots[1]['pivot_index']
    b_low = low_pivots[1]['price']
    c_bullish = high_pivots[1]['pivot_index']
    c_high = high_pivots[1]['price']
    d_bullish = low_pivots[0]['pivot_index']
    d_low = low_pivots[0]['price']
    e_bullish = high_pivots[2]['pivot_index']
    e_high = high_pivots[2]['price']
    f_close = df['close'].iloc[window_length-1]
    
    # Check Bullish Three-Drive sequence
    if a_bullish < b_bullish < c_bullish < d_bullish < e_bullish:
        ab_distance_bullish = a_high - b_low
        bc_retracement_bullish = (c_high - b_low) / ab_distance_bullish
        cd_extension_bullish = (c_high - d_low) / (c_high - b_low)
        de_retracement_bullish = (e_high - d_low) / (c_high - d_low)
        logger.info(f"Bullish Three-Drive")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        logger.info(f"CD Extension: {cd_extension_bullish}")
        logger.info(f"DE Retracement: {de_retracement_bullish}")
        bc_condition = 0.618 <= bc_retracement_bullish <= 0.786
        cd_condition = 1.272 <= cd_extension_bullish <= 1.618
        de_condition = 0.618 <= de_retracement_bullish <= 0.786
        ef_condition = f_close < d_low
        if bc_condition and cd_condition and de_condition and ef_condition:
            result['result'] = "Three-Drive Bullish Pattern"
        else:
            result['result'] = "Three-Drive Bullish Sequence"
        result['A'] = high_pivots[0]
        result['B'] = low_pivots[1]
        result['C'] = high_pivots[1]
        result['D'] = low_pivots[0]
        result['E'] = high_pivots[2]
        result['F'] = f_close

    return result


#
# Function drive3_bearish
#

def drive3_bearish(df, window_length):
    """
    Identify a Bearish Three-Drive pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)
    
    # Bearish Three-Drive:
    #     A (strongest low)
    #     B (second strongest high)
    #     C (second strongest low)
    #     D (strongest high)
    #     E (third strongest low)
    #     F (higher than D)

    # Calculate pivot points using pivotHigh and pivotLow functions
    n_high_pivots = 2
    high_pivots = pivothigh(df, window_length)
    n_low_pivots = 3
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < n_high_pivots or len(low_pivots) < n_low_pivots:
        logger.info(f"Minimum number of high pivots must be {n_high_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Minimum number of low pivots must be {n_low_pivots}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    a_bearish = low_pivots[0]['pivot_index']
    a_low = low_pivots[0]['price']
    b_bearish = high_pivots[1]['pivot_index']
    b_high = high_pivots[1]['price']
    c_bearish = low_pivots[1]['pivot_index']
    c_low = low_pivots[1]['price']
    d_bearish = high_pivots[0]['pivot_index']
    d_high = high_pivots[0]['price']
    e_bearish = low_pivots[2]['pivot_index']
    e_low = low_pivots[2]['price']
    f_close = df['close'].iloc[window_length-1]
    
    # Check Bearish Three-Drive sequence
    if a_bearish < b_bearish < c_bearish < d_bearish < e_bearish:
        ab_distance_bearish = b_high - a_low
        bc_retracement_bearish = (b_high - c_low) / ab_distance_bearish
        cd_extension_bearish = (d_high - c_low) / (b_high - c_low)
        de_retracement_bearish = (d_high - e_low) / (d_high - c_low)
        logger.info(f"Bearish Three-Drive")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        logger.info(f"CD Extension: {cd_extension_bearish}")
        logger.info(f"DE Retracement: {de_retracement_bearish}")
        bc_condition = 0.618 <= bc_retracement_bearish <= 0.786
        cd_condition = 1.272 <= cd_extension_bearish <= 1.618
        de_condition = 0.618 <= de_retracement_bearish <= 0.786
        ef_condition = d_high < f_close
        if bc_condition and cd_condition and de_condition and ef_condition:
            result['result'] = "Three-Drive Bearish Pattern"
        else:
            result['result'] = "Three-Drive Bearish Sequence"
        result['A'] = low_pivots[0]
        result['B'] = high_pivots[1]
        result['C'] = low_pivots[1]
        result['D'] = high_pivots[0]
        result['E'] = low_pivots[2]
        result['F'] = f_close

    return result


#
# Function wolfe_bullish
#

def wolfe_bullish(df, window_length):
    """
    Identify a Bullish Wolfe Wave pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Bullish Wolfe Wave
    #     X (second strongest low)
    #     A (strongest high)
    #     B (strongest low)
    #     C (second strongest high)
    #     D (lower than B)

    x_bullish = low_pivots[1]['pivot_index']
    x_low = low_pivots[1]['price']
    a_bullish = high_pivots[0]['pivot_index']
    a_high = high_pivots[0]['price']
    b_bullish = low_pivots[0]['pivot_index']
    b_low = low_pivots[0]['price']
    c_bullish = high_pivots[1]['pivot_index']
    c_high = high_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bullish Wolfe Wave sequence
    if x_bullish < a_bullish < b_bullish < c_bullish:
        xa_distance_bullish = a_high - x_low
        ab_retracement_bullish = (a_high - b_low) / xa_distance_bullish
        bc_retracement_bullish = (c_high - b_low) / (a_high - b_low)
        logger.info(f"Bullish Wolfe Wave")
        logger.info(f"AB Retracement: {ab_retracement_bullish}")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        ab_condition = 0.786 <= ab_retracement_bullish <= 1.272
        bc_condition = 0.786 <= bc_retracement_bullish <= 1.272
        cd_condition = d_close < b_low
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Wolfe Wave Bullish Pattern"
        else:
            result['result'] = "Wolfe Wave Bullish Sequence"
        result['X'] = low_pivots[1]
        result['A'] = high_pivots[0]
        result['B'] = low_pivots[0]
        result['C'] = high_pivots[1]
        result['D'] = d_close

    return result


#
# Function wolfe_bearish
#

def wolfe_bearish(df, window_length):
    """
    Identify a Bearish Wolfe Wave pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result
    
    # Bearish Wolfe Wave logic:
    #     X (second strongest high)
    #     A (strongest low)
    #     B (strongest high)
    #     C (second strongest low)
    #     D (higher than B)

    x_bearish = high_pivots[1]['pivot_index']
    x_high = high_pivots[1]['price']
    a_bearish = low_pivots[0]['pivot_index']
    a_low = low_pivots[0]['price']
    b_bearish = high_pivots[0]['pivot_index']
    b_high = high_pivots[0]['price']
    c_bearish = low_pivots[1]['pivot_index']
    c_low = low_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bearish Wolfe Wave sequence
    if x_bearish < a_bearish < b_bearish < c_bearish:
        xa_distance_bearish = x_high - a_low
        ab_retracement_bearish = (b_high - a_low) / xa_distance_bearish
        bc_retracement_bearish = (b_high - c_low) / (b_high - a_low)
        logger.info(f"Bearish Wolfe Wave")
        logger.info(f"AB Retracement: {ab_retracement_bearish}")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        ab_condition = 0.786 <= ab_retracement_bearish <= 1.272
        bc_condition = 0.786 <= bc_retracement_bearish <= 1.272
        cd_condition = b_high < d_close
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Wolfe Wave Bearish Pattern"
        else:
            result['result'] = "Wolfe Wave Bearish Sequence"
        result['X'] = high_pivots[1]
        result['A'] = low_pivots[0]
        result['B'] = high_pivots[0]
        result['C'] = low_pivots[1]
        result['D'] = d_close

    return result


#
# Function expansion_bullish
#

def expansion_bullish(df, window_length):
    """
    Identify a Bullish Expansion pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Bullish Expansion
    #     X (second strongest low)
    #     A (second strongest high)
    #     B (strongest low)
    #     C (strongest high)
    #     D (lower than B)

    x_bullish = low_pivots[1]['pivot_index']
    x_low = low_pivots[1]['price']
    a_bullish = high_pivots[1]['pivot_index']
    a_high = high_pivots[1]['price']
    b_bullish = low_pivots[0]['pivot_index']
    b_low = low_pivots[0]['price']
    c_bullish = high_pivots[0]['pivot_index']
    c_high = high_pivots[0]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bullish Expansion sequence
    if x_bullish < a_bullish < b_bullish < c_bullish:
        xa_distance_bullish = a_high - x_low
        ab_retracement_bullish = (a_high - b_low) / xa_distance_bullish
        bc_retracement_bullish = (c_high - b_low) / (a_high - b_low)
        logger.info(f"Bullish Expansion")
        logger.info(f"AB Retracement: {ab_retracement_bullish}")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        ab_condition = 1.272 <= ab_retracement_bullish <= 1.618
        bc_condition = 1.272 <= bc_retracement_bullish <= 1.618
        cd_condition = d_close < b_low
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Expansion Bullish Pattern"
        else:
            result['result'] = "Expansion Bullish Sequence"
        result['X'] = low_pivots[1]
        result['A'] = high_pivots[1]
        result['B'] = low_pivots[0]
        result['C'] = high_pivots[0]
        result['D'] = d_close

    return result


#
# Function expansion_bearish
#

def expansion_bearish(df, window_length):
    """
    Identify a Bearish Expansion pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result
    
    # Bearish Expansion logic:
    #     X (second strongest high)
    #     A (second strongest low)
    #     B (strongest high)
    #     C (strongest low)
    #     D (higher than B)

    x_bearish = high_pivots[1]['pivot_index']
    x_high = high_pivots[1]['price']
    a_bearish = low_pivots[1]['pivot_index']
    a_low = low_pivots[1]['price']
    b_bearish = high_pivots[0]['pivot_index']
    b_high = high_pivots[0]['price']
    c_bearish = low_pivots[0]['pivot_index']
    c_low = low_pivots[0]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bearish Expansion sequence
    if x_bearish < a_bearish < b_bearish < c_bearish:
        xa_distance_bearish = x_high - a_low
        ab_retracement_bearish = (b_high - a_low) / xa_distance_bearish
        bc_retracement_bearish = (b_high - c_low) / (b_high - a_low)
        logger.info(f"Bearish Expansion")
        logger.info(f"AB Retracement: {ab_retracement_bearish}")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        ab_condition = 1.272 <= ab_retracement_bearish <= 1.618
        bc_condition = 1.272 <= bc_retracement_bearish <= 1.618
        cd_condition = b_high < d_close
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Expansion Bearish Pattern"
        else:
            result['result'] = "Expansion Bearish Sequence"
        result['X'] = high_pivots[1]
        result['A'] = low_pivots[1]
        result['B'] = high_pivots[0]
        result['C'] = low_pivots[0]
        result['D'] = d_close

    return result


#
# Function squeeze_bullish
#

def squeeze_bullish(df, window_length):
    """
    Identify a Bullish Squeeze pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Bullish Squeeze
    #     X (strongest low)
    #     A (strongest high)
    #     B (second strongest low)
    #     C (second strongest high)
    #     D (between B and C)

    x_bullish = low_pivots[0]['pivot_index']
    x_low = low_pivots[0]['price']
    a_bullish = high_pivots[0]['pivot_index']
    a_high = high_pivots[0]['price']
    b_bullish = low_pivots[1]['pivot_index']
    b_low = low_pivots[1]['price']
    c_bullish = high_pivots[1]['pivot_index']
    c_high = high_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bullish Expansion sequence
    if x_bullish < a_bullish < b_bullish < c_bullish:
        xa_distance_bullish = a_high - x_low
        ab_retracement_bullish = (a_high - b_low) / xa_distance_bullish
        bc_retracement_bullish = (c_high - b_low) / (a_high - b_low)
        cd_retracement_bullish = (c_high - d_close) / (c_high - b_low)
        logger.info(f"Bullish Expansion")
        logger.info(f"AB Retracement: {ab_retracement_bullish}")
        logger.info(f"BC Retracement: {bc_retracement_bullish}")
        logger.info(f"CD Retracement: {cd_retracement_bullish}")
        ab_condition = 0.618 <= ab_retracement_bullish <= 0.786
        bc_condition = 0.618 <= bc_retracement_bullish <= 0.786
        cd_condition = b_low < d_close < c_high
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Squeeze Bullish Pattern"
        else:
            result['result'] = "Squeeze Bullish Sequence"
        result['X'] = low_pivots[0]
        result['A'] = high_pivots[0]
        result['B'] = low_pivots[1]
        result['C'] = high_pivots[1]
        result['D'] = d_close

    return result


#
# Function squeeze_bearish
#

def squeeze_bearish(df, window_length):
    """
    Identify a Bearish Squeeze pattern using the high and low pivot points.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result
    
    # Bearish Squeeze logic:
    #     X (strongest high)
    #     A (strongest low)
    #     B (second strongest high)
    #     C (second strongest low)
    #     D (between B and C)

    x_bearish = high_pivots[0]['pivot_index']
    x_high = high_pivots[0]['price']
    a_bearish = low_pivots[0]['pivot_index']
    a_low = low_pivots[0]['price']
    b_bearish = high_pivots[1]['pivot_index']
    b_high = high_pivots[1]['price']
    c_bearish = low_pivots[1]['pivot_index']
    c_low = low_pivots[1]['price']
    d_close = df['close'].iloc[window_length-1]
    
    # Check Bearish Squeeze sequence
    if x_bearish < a_bearish < b_bearish < c_bearish:
        xa_distance_bearish = x_high - a_low
        ab_retracement_bearish = (b_high - a_low) / xa_distance_bearish
        bc_retracement_bearish = (b_high - c_low) / (b_high - a_low)
        cd_retracement_bearish = (d_close - c_low) / (b_high - c_low)
        logger.info(f"Bearish Squeeze")
        logger.info(f"AB Retracement: {ab_retracement_bearish}")
        logger.info(f"BC Retracement: {bc_retracement_bearish}")
        logger.info(f"CD Retracement: {cd_retracement_bearish}")
        ab_condition = 0.618 <= ab_retracement_bearish <= 0.786
        bc_condition = 0.618 <= bc_retracement_bearish <= 0.786
        cd_condition = c_low < d_close < b_high
        if ab_condition and bc_condition and cd_condition:
            result['result'] = "Squeeze Bearish Pattern"
        else:
            result['result'] = "Squeeze Bearish Sequence"
        result['X'] = high_pivots[0]
        result['A'] = low_pivots[0]
        result['B'] = high_pivots[1]
        result['C'] = low_pivots[1]
        result['D'] = d_close

    return result


#
# Function rectangle_neutral
#

def rectangle_neutral(df, window_length):
    """
    Identify a Neutral Rectangle pattern using the high and low pivot points.
    A rectangle pattern has two high pivots at similar levels and two low pivots at similar levels,
    forming a horizontal bounding box.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate ATR for the threshold
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = df['tr'].rolling(window=14).mean().iloc[-1]

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Rectangle Pattern (most recent pivots first in the sorted lists)
    # Get the two most recent high pivots (sorting by date/index, not price)
    high_pivots_by_date = sorted(high_pivots, key=lambda x: x['pivot_index'], reverse=True)[:2]
    low_pivots_by_date = sorted(low_pivots, key=lambda x: x['pivot_index'], reverse=True)[:2]
    
    if len(high_pivots_by_date) < 2 or len(low_pivots_by_date) < 2:
        return result

    # Check if the high pivots are within 1 ATR of each other
    high1 = high_pivots_by_date[0]['price']
    high2 = high_pivots_by_date[1]['price']
    low1 = low_pivots_by_date[0]['price']
    low2 = low_pivots_by_date[1]['price']
    
    d_close = df['close'].iloc[-1]

    # Rectangle conditions:
    # 1. Two high pivots within 1 ATR
    # 2. Two low pivots within 1 ATR
    # 3. Current close is between the most recent high and low pivots
    high_condition = abs(high1 - high2) <= atr
    low_condition = abs(low1 - low2) <= atr
    close_condition = low1 < d_close < high1

    if high_condition and low_condition and close_condition:
        result['result'] = "Rectangle Neutral Pattern"
        result['A'] = high_pivots_by_date[1]  # Earlier high pivot
        result['B'] = low_pivots_by_date[1]   # Earlier low pivot
        result['C'] = high_pivots_by_date[0]  # Recent high pivot
        result['D'] = low_pivots_by_date[0]   # Recent low pivot
        result['atr'] = atr
        result['high_diff'] = abs(high1 - high2)
        result['low_diff'] = abs(low1 - low2)
        logger.info(f"Rectangle Pattern: High diff={result['high_diff']:.2f}, Low diff={result['low_diff']:.2f}, ATR={atr:.2f}")

    return result


#
# Function wedge_neutral
#

def wedge_neutral(df, window_length):
    """
    Identify a Neutral Wedge pattern using the high and low pivot points.
    A wedge pattern has the most recent high pivot lower than the previous high pivot
    and the most recent low pivot lower than the previous low pivot.
    """

    # Initialize the results
    result = {}

    # Take the last n rows of the DataFrame
    df = df.tail(window_length)

    # Calculate ATR for the threshold
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = df['tr'].rolling(window=14).mean().iloc[-1]

    # Calculate pivot points using pivotHigh and pivotLow functions
    minimum_pivots = 2
    high_pivots = pivothigh(df, window_length)
    low_pivots = pivotlow(df, window_length)
    if len(high_pivots) < minimum_pivots or len(low_pivots) < minimum_pivots:
        logger.info(f"Minimum number of high or low pivots must be {minimum_pivots}")
        logger.info(f"High Pivots: {len(high_pivots)}")
        logger.info(f"Low  Pivots: {len(low_pivots)}")
        return result

    # Wedge Pattern (most recent pivots first in the sorted lists)
    # Get the two most recent high pivots (sorting by date/index, not price)
    high_pivots_by_date = sorted(high_pivots, key=lambda x: x['pivot_index'], reverse=True)[:2]
    low_pivots_by_date = sorted(low_pivots, key=lambda x: x['pivot_index'], reverse=True)[:2]
    
    if len(high_pivots_by_date) < 2 or len(low_pivots_by_date) < 2:
        return result

    # Most recent pivots
    recent_high = high_pivots_by_date[0]['price']
    previous_high = high_pivots_by_date[1]['price']
    recent_low = low_pivots_by_date[0]['price']
    previous_low = low_pivots_by_date[1]['price']
    
    d_close = df['close'].iloc[-1]

    # Wedge conditions:
    # 1. Recent high is lower than previous high by more than 1 ATR
    # 2. Recent low is lower than previous low by more than 1 ATR
    # 3. Current close is between the most recent high and low pivots
    high_condition = (previous_high - recent_high) > atr
    low_condition = (previous_low - recent_low) > atr
    close_condition = recent_low < d_close < recent_high

    if high_condition and low_condition and close_condition:
        result['result'] = "Wedge Neutral Pattern"
        result['A'] = high_pivots_by_date[1]  # Earlier high pivot
        result['B'] = low_pivots_by_date[1]   # Earlier low pivot
        result['C'] = high_pivots_by_date[0]  # Recent high pivot (lower)
        result['D'] = low_pivots_by_date[0]   # Recent low pivot (lower)
        result['E'] = d_close
        result['atr'] = atr
        result['high_drop'] = previous_high - recent_high
        result['low_drop'] = previous_low - recent_low
        logger.info(f"Wedge Pattern: High drop={result['high_drop']:.2f}, Low drop={result['low_drop']:.2f}, ATR={atr:.2f}")

    return result


#
# Function plot_pattern
#

def plot_pattern(df, pattern, title="Candlestick Chart with Pattern"):
    """
    Generate a Plotly candlestick chart with an overlay for any pattern.

    Parameters:
    - df: A pandas DataFrame with columns: ['date', 'open', 'high', 'low', 'close'].
    - pattern: A dictionary containing pattern points with 'price' and 'date' values.
        - 4 Point Pattern: A, B, C, D
        - 5 Point Pattern: X, A, B, C, D
        - 6 Point Pattern: A, B, C, D, E, F
    - title: (optional) Title for the chart.
    """

    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].iloc[-1]

    # Normalize scalar values (e.g., 'D') to dictionaries with 'date' and 'price'
    for key, value in pattern.items():
        if not isinstance(value, dict):
            pattern[key] = {'date': last_date, 'price': value}

    # Sort pattern points by their 'date' field to ensure chronological order
    sorted_pattern = sorted(pattern.values(), key=lambda x: x['date'])

    # Extract dates and prices from the sorted pattern
    pattern_dates = [point['date'] for point in sorted_pattern]
    pattern_prices = [point['price'] for point in sorted_pattern]

    # Create the candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlestick'
    )])

    # Add the pattern overlay to the chart
    fig.add_trace(go.Scatter(
        x=pattern_dates,
        y=pattern_prices,
        mode='lines+markers',
        name='Pattern Overlay',
        line=dict(color='orange', width=2),
        marker=dict(size=10)
    ))

    # Customize the chart layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )

    # Display the chart
    fig.show()
