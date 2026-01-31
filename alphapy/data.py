################################################################################
#
# Package   : AlphaPy
# Module    : data
# Created   : July 11, 2013
#
# Copyright 2019 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# Imports
#

from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import read_frame
from alphapy.globals import datasets
from alphapy.globals import ModelType
from alphapy.globals import Partition
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.globals import SSEP
from alphapy.globals import WILDCARD
from alphapy.space import Space
from alphapy.data_sources import AlpacaDataSource, PolygonDataSource

from datetime import datetime
from io import BytesIO
import logging
import numpy as np
import polars as pl
import re
import requests
from sklearn.preprocessing import LabelEncoder
import sys
import yfinance as yf


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_data
#

def get_data(model, partition):
    r"""Get data for the given partition.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.
    partition : alphapy.Partition
        Reference to the dataset.

    Returns
    -------
    df_X : polars.DataFrame
        The feature set.
    df_y : polars.DataFrame
        The array of target values, if available.

    """

    logger.info("Loading Data")

    # Extract the model data

    directory = model.specs['directory']
    live_results = model.specs['live_results']
    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    features = model.specs['features']
    model_type = model.specs['model_type']
    separator = model.specs['separator']
    target = model.specs['target']

    # Initialize X and y

    df_X = pl.DataFrame()
    df_y = pl.DataFrame()

    # Read in the file

    filename = datasets[partition]
    data_dir = SSEP.join([directory, 'data'])
    df = read_frame(data_dir, filename, extension, separator)
    if df.is_empty():
        input_dir = SSEP.join([run_dir, 'input'])
        df = read_frame(input_dir, filename, extension, separator)

    # Get features and target

    if not df.is_empty():
        if target in df.columns:
            logger.info("Found target %s in data frame", target)
            # check if target column has NaN values
            nan_count = df[target].null_count()
            logger.info("Found %d records with NaN target values", nan_count)
            # drop NA targets
            if not live_results or partition == Partition.train:
                df = df.drop_nulls(subset=[target])
                if nan_count > 0:
                    logger.info("Dropped %d records with NaN target values", nan_count)
            # assign the target column to y
            df_y = df.select(target)
            # encode label only for classification or system
            if model_type == ModelType.classification or model_type == ModelType.system:
                y = LabelEncoder().fit_transform(df_y[target].to_numpy())
                df_y = pl.DataFrame({target: y})
            # drop the target from the original frame
            df = df.drop(target)
        else:
            logger.info("Target %s not found in %s", target, partition)
        # Extract features
        if features == WILDCARD:
            df_X = df
        else:
            df_X = df.select(features)

    # Labels are returned usually only for training data
    return df_X, df_y


#
# Function shuffle_data
#

def shuffle_data(model):
    r"""Randomly shuffle the training data.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the shuffled data.

    """

    # Extract model parameters.

    seed = model.specs['seed']
    shuffle = model.specs['shuffle']

    # Extract model data.

    X_train = model.X_train

    # Shuffle data

    if shuffle:
        logger.info("Shuffling Training Data")
        np.random.seed(seed)
        np.random.shuffle(X_train)
        model.X_train = X_train
    else:
        logger.info("Skipping Shuffling")

    return model


#
# Function convert_data
#

def convert_data(df: pl.DataFrame, intraday_data: bool) -> pl.DataFrame:
    r"""Convert the market data frame to canonical format.

    Parameters
    ----------
    df : polars.DataFrame
        The intraday dataframe.
    intraday_data : bool
        Flag set to True if the frame contains intraday data.

    Returns
    -------
    df : polars.DataFrame
        The canonical dataframe with date/time columns.

    """
    # Ensure datetime column exists
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have a 'datetime' column")

    # Add date column
    df = df.with_columns(
        pl.col("datetime").dt.date().alias("date")
    )

    # Add date parts
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.weekday().alias("dayofweek"),
        pl.col("date").dt.ordinal_day().alias("dayofyear"),
        pl.col("date").dt.week().alias("weekofyear"),
        pl.col("date").dt.quarter().alias("quarter"),
    ])

    if intraday_data:
        # Add time column
        df = df.with_columns(
            pl.col("datetime").dt.time().alias("time")
        )

        # Add time parts
        df = df.with_columns([
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.minute().alias("minute"),
            pl.col("datetime").dt.second().alias("second"),
        ])

        # Group by date for intraday calculations
        df = df.with_columns([
            pl.col("datetime").cum_count().over("date").alias("barnumber"),
        ])

        # Calculate bar percentage within day
        df = df.with_columns([
            (pl.col("barnumber") * 100.0 / pl.col("barnumber").max().over("date")).alias("barpct"),
        ])

        # Progressive intraday columns
        df = df.with_columns([
            pl.col("open").first().over("date").alias("opend"),
            pl.col("high").cum_max().over("date").alias("highd"),
            pl.col("low").cum_min().over("date").alias("lowd"),
            pl.col("close").last().over("date").alias("closed"),
        ])

        # Mark end of day
        df = df.with_columns([
            (pl.col("barnumber") == pl.col("barnumber").max().over("date")).alias("endofday"),
        ])

        # Drop time column after extracting parts
        df = df.drop("time")

    # Drop date column after extracting parts
    df = df.drop("date")

    # Ensure numerical columns are float
    cols_float = ["open", "high", "low", "close", "volume"]
    for col in cols_float:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    # Forward-fill prices and volume
    for col in cols_float:
        if col in df.columns:
            df = df.with_columns(pl.col(col).forward_fill())

    # Sort by datetime
    df = df.sort("datetime")

    return df


#
# Function convert_offset
#

def convert_offset(alias, mappings):
    r"""Convert offset alias to timespan.

    Parameters
    ----------
    alias : str
        Pandas offset alias.
    mappings : dict
        Mapping of offset alias time frame.

    Returns
    -------
    number : int
        The number of periods of the data to be retrieved.
    timespan : str
        The period of the data feed to be retrieved.

    """

    # Separate number and term

    match = re.match(r"(\d+)?(\w+)", alias)
    if match:
        number, term = match.groups()
        number = int(number) if number else 1

        # Convert term to timespan
        if term in mappings:
            timespan = mappings[term]
            return (number, timespan)
        else:
            raise ValueError(f"Unknown offset alias: {alias}")
    else:
        raise ValueError(f"Invalid offset alias: {alias}")


#
# Function get_polygon_data
#

def get_polygon_data(source, alphapy_specs, symbol, intraday_data, data_fractal,
                     from_date, to_date, lookback_period):
    r"""Get Polygon daily and intraday data using consolidated PolygonDataSource.

    Parameters
    ----------
    source : str
        The data feed.
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : polars.DataFrame
        The dataframe containing the market data.

    """
    # Use the consolidated PolygonDataSource
    api_key = alphapy_specs['sources']['polygon']['api_key']
    polygon = PolygonDataSource(api_key=api_key)

    # Parse dates
    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date, "%Y-%m-%d")

    # Get bars
    results = polygon.get_bars(
        symbols=symbol.upper(),
        timeframe=data_fractal,
        lookback=lookback_period,
        start_date=start_dt,
        end_date=end_dt,
    )

    if symbol.upper() in results:
        return results[symbol.upper()]

    return pl.DataFrame()


#
# Function get_yahoo_data
#

def get_yahoo_data(source, alphapy_specs, symbol, intraday_data, data_fractal,
                   from_date, to_date, lookback_period):
    r"""Get Yahoo daily and intraday data.

    Parameters
    ----------
    source : str
        The data feed.
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : polars.DataFrame
        The dataframe containing the market data.

    """

    data_fractal = data_fractal.lower()
    yahoo_fractals = {'min': 'm',
                      'h': 'h',
                      'd': 'd',
                      'w': 'wk',
                      'm': 'mo'}
    pandas_offsets = yahoo_fractals.keys()
    fractal = [offset for offset in pandas_offsets if offset in data_fractal]

    if fractal:
        fvalue = fractal[0]
        yahoo_fractal = data_fractal.replace(fvalue, yahoo_fractals[fvalue])
        # intraday limit is 60 days
        ignore_tz = True if intraday_data else False

        # yfinance returns pandas, convert to polars
        import pandas as pd
        pdf = yf.download(symbol, start=from_date, end=to_date, interval=yahoo_fractal,
                          ignore_tz=ignore_tz, threads=False)
        if pdf.empty:
            logger.info("Could not get data for: %s", symbol)
            return pl.DataFrame()

        # Convert pandas to polars
        pdf = pdf.reset_index()
        pdf.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in pdf.columns]

        # Rename index column to datetime
        if 'date' in pdf.columns:
            pdf = pdf.rename(columns={'date': 'datetime'})

        df = pl.from_pandas(pdf)
        return df
    else:
        logger.error("Valid Pandas Offsets for Yahoo Data are: %s", pandas_offsets)
        return pl.DataFrame()


#
# Function get_eodhd_data
#

def get_eodhd_data(source, alphapy_specs, symbol, intraday_data, data_fractal,
                   from_date, to_date, lookback_period):
    r"""Get EODHD daily and intraday data.

    Parameters
    ----------
    source : str
        The data feed.
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    symbol : str
        A valid stock symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : polars.DataFrame
        The dataframe containing the intraday data.

    """
    import pytz

    symbol = symbol.upper()

    mappings = {
        "min": "m",
        "T": "m",
        "H": "h",
        "D": "d",
        "W": "w",
        "M": "m",
    }
    n_periods, period = convert_offset(data_fractal, mappings)

    api_key = alphapy_specs['sources']['eodhd']['api_key']
    api_key_str = 'api_token=' + api_key
    format_str = 'fmt=csv'

    if intraday_data:
        url_base = f'https://eodhd.com/api/intraday/{symbol}?'
        tz_eastern = pytz.timezone('US/Eastern')
        from_obj = datetime.strptime(from_date, "%Y-%m-%d")
        from_obj_est = tz_eastern.localize(from_obj)
        from_unix_time = int(from_obj_est.timestamp())
        from_str = 'from=' + str(from_unix_time)
        to_obj = datetime.strptime(to_date, "%Y-%m-%d")
        to_obj_est = tz_eastern.localize(to_obj)
        to_unix_time = int(to_obj_est.timestamp())
        to_str = 'to=' + str(to_unix_time)
        interval_str = 'interval=' + str(n_periods) + period
        url_str = '&'.join([api_key_str, from_str, to_str, interval_str, format_str])
    else:
        url_base = f'https://eodhd.com/api/eod/{symbol}?'
        from_str = 'from=' + from_date
        to_str = 'to=' + to_date
        period_str = 'period=' + period
        url_str = '&'.join([api_key_str, from_str, to_str, period_str, format_str])

    url = url_base + url_str
    response = requests.get(url).content

    # Read CSV with polars
    df = pl.read_csv(BytesIO(response))
    df = df.rename({col: col.lower() for col in df.columns})
    cols_ohlcv = ['open', 'high', 'low', 'close', 'volume']

    if intraday_data:
        col_dt = 'datetime'
        df = df.with_columns(
            pl.col(col_dt).str.to_datetime().alias(col_dt)
        )
    else:
        col_dt = 'date'
        df = df.with_columns(
            pl.col(col_dt).str.to_datetime().alias("datetime")
        )
        df = df.drop(col_dt)

    # Select relevant columns
    select_cols = ["datetime"] + [c for c in cols_ohlcv if c in df.columns]
    df = df.select(select_cols)

    return df


#
# Function get_alpaca_data
#

def get_alpaca_data(source, alphapy_specs, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get Alpaca daily and intraday data.

    Parameters
    ----------
    source : str
        The data feed.
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    symbol : str
        A valid stock or crypto symbol.
    intraday_data : bool
        If True, then get intraday data.
    data_fractal : str
        Pandas offset alias.
    from_date : str
        Starting date for symbol retrieval.
    to_date : str
        Ending date for symbol retrieval.
    lookback_period : int
        The number of periods of data to retrieve.

    Returns
    -------
    df : polars.DataFrame
        The dataframe containing the market data.

    """
    # Get credentials (optional for crypto)
    alpaca_config = alphapy_specs.get('sources', {}).get('alpaca', {})
    api_key = alpaca_config.get('api_key')
    api_secret = alpaca_config.get('api_secret')

    alpaca = AlpacaDataSource(api_key=api_key, api_secret=api_secret)

    # Parse dates
    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.strptime(to_date, "%Y-%m-%d")

    # Get bars
    results = alpaca.get_bars(
        symbols=symbol.upper(),
        timeframe=data_fractal,
        lookback=lookback_period,
        start_date=start_dt,
        end_date=end_dt,
    )

    if symbol.upper() in results:
        return results[symbol.upper()]

    return pl.DataFrame()


#
# Data Dispatch Tables
#

data_dispatch_table = {
    'alpaca': get_alpaca_data,
    'eodhd': get_eodhd_data,
    'polygon': get_polygon_data,
    'yahoo': get_yahoo_data,
}


#
# Function assign_global_data
#

def assign_global_data(df, symbol, gspace, fractal):
    r"""Create global pointer to dataframe.

    Parameters
    ----------
    df : polars.DataFrame
        The dataframe for the given symbol.
    symbol : str
        Pandas offset alias.
    gspace : alphapy.Space
        AlphaPy data taxonomy data source and subject.
    fractal : str
        Pandas offset alias.

    Returns
    -------
    df : polars.DataFrame
        The dataframe for the given symbol.

    """
    try:
        space = Space(gspace.subject, gspace.source, fractal)
        _ = Frame(symbol.lower(), space, df)
    except:
        logger.error("Could not allocate Frame for: %s", symbol.upper())
    return df


#
# Function standardize_data
#

def standardize_data(symbol, gspace, df, fractal, intraday_data):
    r"""Standardize market data.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    gspace : alphapy.Space
        AlphaPy data taxonomy data source and subject.
    df : polars.DataFrame
        The raw output dataframe from the market datafeed.
    fractal : str
        Pandas offset alias.
    intraday_data : bool
        If True, then get intraday data.

    Returns
    -------
    df : polars.DataFrame
        The standardized output dataframe for the market data.

    """

    # convert data to canonical form
    df = convert_data(df, intraday_data)
    # create global pointer to dataframe
    df = assign_global_data(df, symbol, gspace, fractal)
    # return dataframe
    return df


#
# Function resample_ohlcv
#

def resample_ohlcv(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Resample OHLCV data to a different timeframe.

    Parameters
    ----------
    df : polars.DataFrame
        DataFrame with datetime and OHLCV columns
    timeframe : str
        Target timeframe (e.g., "1h", "1d", "1D")

    Returns
    -------
    df : polars.DataFrame
        Resampled DataFrame
    """
    # Normalize timeframe to lowercase for Polars (e.g., "1D" -> "1d")
    timeframe = timeframe.lower()
    return df.group_by_dynamic(
        "datetime",
        every=timeframe,
    ).agg([
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    ]).drop_nulls()


#
# Function get_market_data
#

def get_market_data(alphapy_specs, model, market_specs, group,
                    lookback_period, intraday_data=False, local_dir=''):
    r"""Get data from an external feed.

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    model : alphapy.Model
        The model object describing the data.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.
    group : alphapy.Group
        The group of symbols.
    lookback_period : int
        The number of periods of data to retrieve.
    intraday_data : bool
        If True, then get intraday data.
    local_dir : str
        Local data directory, if needed.
    """

    # Unpack market specifications

    data_fractal = market_specs['data_fractal']
    feature_fractals = market_specs['fractals']
    from_date = market_specs['data_start_date']
    to_date = market_specs['data_end_date']
    start_time = datetime.strptime(market_specs['data_start_time'], '%H:%M').time()
    end_time = datetime.strptime(market_specs['data_end_time'], '%H:%M').time()

    # Unpack model specifications

    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack group elements

    gspace = group.space
    gsubject = gspace.subject
    gsource = gspace.source

    # Determine the feed source

    if intraday_data:
        logger.info("Source [%s] Intraday Data [%s] for %d days",
                    gsource, data_fractal, lookback_period)
    else:
        logger.info("Source [%s] Daily Data [%s] for %d days",
                    gsource, data_fractal, lookback_period)

    # Get the data from the specified data feed

    remove_list = []

    for symbol in group.members:
        logger.info("Getting %s data from %s to %s",
                    symbol.upper(), from_date, to_date)
        # Locate the data source
        if gsource == 'data':
            # locally stored intraday or daily data
            dspace = Space(gsubject, gsource, data_fractal)
            fname = frame_name(symbol, dspace)
            df = read_frame(local_dir, fname, extension, separator)
        elif gsource in data_dispatch_table.keys():
            df = data_dispatch_table[gsource](gsource,
                                              alphapy_specs,
                                              symbol,
                                              intraday_data,
                                              data_fractal,
                                              from_date,
                                              to_date,
                                              lookback_period)
        else:
            raise ValueError("Unsupported Data Source: %s", gsource)

        # Now that we have content, standardize the data
        if not df.is_empty():
            logger.info("Rows: %d [%s]", len(df), data_fractal)

            # Ensure column names are lowercase
            df = df.rename({col: col.lower() for col in df.columns})

            # Find datetime column
            dt_cols = ['datetime', 'date']
            dt_column = [x for x in df.columns if x in dt_cols]

            if dt_column:
                # Rename to datetime if needed
                if dt_column[0] == 'date':
                    df = df.with_columns(
                        pl.col("date").cast(pl.Datetime).alias("datetime")
                    ).drop("date")
            else:
                raise ValueError("DataFrame must have a datetime or date column")

            # Remove duplicates
            df = df.unique(subset=["datetime"], keep="first")

            # Filter by time range for intraday
            if intraday_data and start_time and end_time:
                df = df.filter(
                    (pl.col("datetime").dt.time() >= start_time) &
                    (pl.col("datetime").dt.time() <= end_time)
                )

            # Filter by date range - cast to handle timezone/precision differences
            df = df.filter(
                (pl.col("datetime").dt.date() >= pl.lit(from_date).str.to_date()) &
                (pl.col("datetime").dt.date() <= pl.lit(to_date).str.to_date())
            )

            # Register the dataframe
            df = standardize_data(symbol, gspace, df, data_fractal, intraday_data)

            # Resample to other fractals
            for ff in feature_fractals:
                if ff != data_fractal:
                    df_rs = resample_ohlcv(df, ff)
                    logger.info("Rows: %d [%s] resampled", len(df_rs), ff)
                    # Check if this is intraday
                    intraday_fractal = any(substring in ff for substring in PD_INTRADAY_OFFSETS)
                    df_rs = standardize_data(symbol, gspace, df_rs, ff, intraday_fractal)
        else:
            logger.info("No DataFrame for %s", symbol.upper())
            remove_list.append(symbol)

    # Remove any group members not found

    if remove_list:
        group.remove(remove_list)

    return
