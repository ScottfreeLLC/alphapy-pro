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
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.globals import SSEP
from alphapy.globals import SamplingMethod
from alphapy.globals import WILDCARD
from alphapy.space import Space
from alphapy.transforms import dateparts
from alphapy.transforms import timeparts

from datetime import datetime
from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
import logging
import numpy as np
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
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
    df_X : pandas.DataFrame
        The feature set.
    df_y : pandas.DataFrame
        The array of target values, if available.

    """

    logger.info("Loading Data")

    # Extract the model data

    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    features = model.specs['features']
    model_type = model.specs['model_type']
    separator = model.specs['separator']
    target = model.specs['target']
    allow_na_targets = model.specs['allow_na_targets']

    # Initialize X and y

    df_X = pd.DataFrame()
    df_y = pd.DataFrame()

    # Read in the file

    filename = datasets[partition]
    input_dir = SSEP.join([run_dir, 'input'])
    df = read_frame(input_dir, filename, extension, separator)

    # Get features and target

    if not df.empty:
        if target in df.columns:
            logger.info("Found target %s in data frame", target)
            # check if target column has NaN values
            nan_count = df[target].isnull().sum()
            if nan_count > 0 and not allow_na_targets:
                logger.info("Found %d records with NaN target values", nan_count)
                logger.info("Labels (y) for %s will not be used", partition)
            else:
                logger.info("Labels (y) found for %s", partition)
                # drop NA targets
                df = df.dropna(subset=[target])
                if nan_count > 0:
                    logger.info("Dropped %d records with NaN target values", nan_count)
                # assign the target column to y
                df_y = df[target]
                # encode label only for classification
                if model_type == ModelType.classification:
                    y = LabelEncoder().fit_transform(df_y)
                    df_y = pd.DataFrame(y, columns=[target])
            # drop the target from the original frame
            df = df.drop([target], axis=1)
        else:
            logger.info("Target %s not found in %s", target, partition)
        # Extract features
        if features == WILDCARD:
            df_X = df
        else:
            df_X = df[features]

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
# Function sample_data
#

def sample_data(model):
    r"""Sample the training data.

    Sampling is configured in the ``model.yml`` file (data:sampling:method)
    You can learn more about resampling techniques here [IMB]_.

    Parameters
    ----------
    model : alphapy.Model
        The model object describing the data.

    Returns
    -------
    model : alphapy.Model
        The model object with the sampled data.

    """

    logger.info("Sampling Data")

    # Extract model parameters.

    sampling_method = model.specs['sampling_method']
    sampling_ratio = model.specs['sampling_ratio']
    target = model.specs['target']

    # Extract model data.

    X_train = model.X_train
    y_train = model.y_train

    # Calculate the sampling ratio if one is not provided.

    _, uc = np.unique(y_train, return_counts=True)
    current_ratio = uc[1] / uc[0]
    logger.info("Sampling Ratio for target %s [%r]: %.2f => %.2f",
                target, uc[1], current_ratio, sampling_ratio)

    # Choose the sampling method.

    if sampling_method == SamplingMethod.under_random:
        sampler = RandomUnderSampler(sampling_strategy=sampling_ratio)
    elif sampling_method == SamplingMethod.under_tomek:
        sampler = TomekLinks()
    elif sampling_method == SamplingMethod.under_cluster:
        sampler = ClusterCentroids()
    elif sampling_method == SamplingMethod.under_nearmiss:
        sampler = NearMiss(version=1)
    elif sampling_method == SamplingMethod.under_ncr:
        sampler = NeighbourhoodCleaningRule()
    elif sampling_method == SamplingMethod.over_random:
        sampler = RandomOverSampler(sampling_strategy=sampling_ratio)
    elif sampling_method == SamplingMethod.over_smote:
        sampler = SMOTE()
    elif sampling_method == SamplingMethod.over_smoteb:
        sampler = SMOTE(sampling_strategy=sampling_ratio, kind='borderline1')
    elif sampling_method == SamplingMethod.over_smotesv:
        sampler = SMOTE(sampling_strategy=sampling_ratio, kind='svm')
    elif sampling_method == SamplingMethod.overunder_smote_tomek:
        sampler = SMOTETomek(sampling_strategy=sampling_ratio)
    elif sampling_method == SamplingMethod.overunder_smote_enn:
        sampler = SMOTEENN(sampling_strategy=sampling_ratio)
    elif sampling_method == SamplingMethod.ensemble_easy:
        sampler = EasyEnsembleClassifier()
    else:
        raise ValueError("Unknown Sampling Method %s" % sampling_method)

    # Get the newly sampled features.

    X, y = sampler.fit_resample(X_train, y_train)

    logger.info("Original Samples : %d", X_train.shape[0])
    logger.info("New Samples      : %d", X.shape[0])

    # Store the new features in the model.

    model.X_train = X
    model.y_train = y

    return model


#
# Function convert_data
#

def convert_data(df, intraday_data):
    r"""Convert the market data frame to canonical format.

    Parameters
    ----------
    df : pandas.DataFrame
        The intraday dataframe.
    intraday_data : bool
        Flag set to True if the frame contains intraday data.

    Returns
    -------
    df : pandas.DataFrame
        The canonical dataframe with date/time index.

    """

    # Create the date and time columns

    df['date'] = df.index
    df['date'] = pd.to_datetime(df['date']).dt.date
    if intraday_data:
        df['time'] = df.index
        df['time'] = pd.to_datetime(df['time']).dt.time

    # Add datetime columns

    # daily data
    df = pd.concat([df, dateparts(df, 'date')], axis=1)

    # intraday data
    if intraday_data:
        # Group by date first
        date_group = df.groupby('date')
        # Number the intraday bars
        df['barnumber'] = date_group.cumcount().astype(int)
        df['barpct'] = date_group['barnumber'].apply(lambda x: 100.0 * x / x.count())
        # Add progressive intraday columns
        df['opend'] = date_group['open'].transform('first')
        df['highd'] = date_group['high'].cummax()
        df['lowd'] = date_group['low'].cummin()
        df['closed'] = date_group['close'].transform('last')
        # Mark the end of the trading day
        df['endofday'] = False
        df.loc[date_group.tail(1).index, 'endofday'] = True
        # get time fields
        df = pd.concat([df, timeparts(df, 'time')], axis=1)

    # Drop date and time fields after extracting parts

    del df['date']
    if intraday_data:
        del df['time']

    # Make the numerical columns floating point

    cols_float = ['open', 'high', 'low', 'close', 'volume']
    df[cols_float] = df[cols_float].astype(float)

    # Forward-Fill prices and volume
    df.loc[:, cols_float] = df.loc[:, cols_float].ffill()

    # Order the frame by increasing date if necessary
    df = df.sort_index()
    return df


#
# Function get_google_intraday_data
#

def get_google_intraday_data(symbol, lookback_period, fractal):
    r"""Get Google Finance intraday data.

    We get intraday data from the Google Finance API, even though
    it is not officially supported. You can retrieve a maximum of
    50 days of history, so you may want to build your own database
    for more extensive backtesting.

    Parameters
    ----------
    symbol : str
        A valid stock symbol.
    lookback_period : int
        The number of days of intraday data to retrieve, capped at 50.
    fractal : str
        The intraday frequency, e.g., "5m" for 5-minute data.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe containing the intraday data.

    """

    # Google requires upper-case symbol, otherwise not found
    symbol = symbol.upper()
    # Initialize data frame
    df = pd.DataFrame()
    # Convert fractal to interval
    interval = 60 * int(re.findall('\d+', fractal)[0])
    # Google has a 50-day limit
    max_days = 50
    if lookback_period > max_days:
        lookback_period = max_days
    # Set Google data constants
    toffset = 7
    line_length = 6
    # Make the request to Google
    base_url = 'https://finance.google.com/finance/getprices?q={}&i={}&p={}d&f=d,o,h,l,c,v'
    url = base_url.format(symbol, interval, lookback_period)
    response = requests.get(url)
    # Process the response
    text = response.text.split('\n')
    records = []
    for line in text[toffset:]:
        items = line.split(',')
        if len(items) == line_length:
            dt_item = items[0]
            close_item = items[1]
            high_item = items[2]
            low_item = items[3]
            open_item = items[4]
            volume_item = items[5]
            if dt_item[0] == 'a':
                day_item = float(dt_item[1:])
                offset = 0
            else:
                offset = float(dt_item)
            dt = datetime.fromtimestamp(day_item + (interval * offset))
            dt = pd.to_datetime(dt)
            dt_date = dt.strftime('%Y-%m-%d')
            dt_time = dt.strftime('%H:%M:%S')
            record = (dt_date, dt_time, open_item, high_item, low_item, close_item, volume_item)
            records.append(record)
    # Create data frame
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame.from_records(records, columns=cols)
    # Return the dataframe
    return df


#
# Function get_google_data
#

def get_google_data(source, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get data from Google.

    Parameters
    ----------
    source : str
        The data feed.
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
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    df = pd.DataFrame()
    if intraday_data:
        # use internal function
        # df = get_google_intraday_data(symbol, lookback_period, data_fractal)
        logger.info("Google Finance API for intraday data no longer available")
    else:
        # Google Finance API no longer available
        logger.info("Google Finance API for daily data no longer available")
    return df


#
# Function get_iex_data
#

def get_iex_data(source, symbol, intraday_data, data_fractal,
                 from_date, to_date, lookback_period):
    r"""Get data from IEX.

    Parameters
    ----------
    source : str
        The data feed.
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
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    symbol = symbol.upper()
    df = pd.DataFrame()

    if intraday_data:
        # use iexfinance function to get intraday data for each date
        df = pd.DataFrame()
        for d in pd.date_range(from_date, to_date):
            dstr = d.strftime('%Y-%m-%d')
            logger.info("%s Data for %s", symbol, dstr)
            try:
                df1 = get_historical_intraday(symbol, d, output_format="pandas")
                df1_len = len(df1)
                if df1_len > 0:
                    logger.info("%s: %d rows", symbol, df1_len)
                    df = df.append(df1)
                else:
                    logger.info("%s: No Trading Data for %s", symbol, dstr)
            except:
                iex_error = "*** IEX Intraday Data Error (check Quota) ***"
                logger.error(iex_error)
                sys.exit(iex_error)
    else:
        # use iexfinance function for historical daily data
        try:
            df = get_historical_data(symbol, from_date, to_date, output_format="pandas")
        except:
            iex_error = "*** IEX Daily Data Error (check Quota) ***"
            logger.error(iex_error)
            sys.exit(iex_error)
    return df


#
# Function get_pandas_data
#

def get_pandas_data(source, symbol, intraday_data, data_fractal,
                    from_date, to_date, lookback_period):
    r"""Get Pandas Web Reader data.

    Parameters
    ----------
    source : str
        The data feed.
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
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    # Call the Pandas Web data reader.

    try:
        df = web.DataReader(symbol, source, from_date, to_date)
    except:
        df = pd.DataFrame()
        logger.info("Could not retrieve %s data with pandas-datareader", symbol.upper())

    return df


#
# Function get_yahoo_data
#

def get_yahoo_data(source, symbol, intraday_data, data_fractal,
                   from_date, to_date, lookback_period):
    r"""Get Yahoo daily and intraday data.

    Parameters
    ----------
    source : str
        The data feed.
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
    df : pandas.DataFrame
        The dataframe containing the market data.

    """

    df = pd.DataFrame()
    data_fractal = data_fractal.lower()
    yahoo_fractals = {'min' : 'm',
                      'h'   : 'h',
                      'd'   : 'd',
                      'w'   : 'wk',
                      'm'   : 'mo'}
    pandas_offsets = yahoo_fractals.keys()
    fractal = [offset for offset in pandas_offsets if offset in data_fractal]
    if fractal:
        fvalue = fractal[0]
        yahoo_fractal = data_fractal.replace(fvalue, yahoo_fractals[fvalue])
        # intraday limit is 60 days
        ignore_tz = True if intraday_data else False
        df = yf.download(symbol, start=from_date, end=to_date, interval=yahoo_fractal,
                         ignore_tz=ignore_tz, threads=False)
        if df.empty:
            logger.info("Could not get data for: %s", symbol)
        else:
            df.index = df.index.tz_localize(None)
    else:
        logger.error("Valid Pandas Offsets for Yahoo Data are: %s", pandas_offsets)
    return df


#
# Data Dispatch Tables
#

data_dispatch_table = {'google' : get_google_data,
                       'iex'    : get_iex_data,
                       'pandas' : get_pandas_data,
                       'yahoo'  : get_yahoo_data}


#
# Function assign_global_data
#

def assign_global_data(df, symbol, gspace, fractal):
    r"""Create global pointer to dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe for the given symbol.
    symbol : str
        Pandas offset alias.
    gspace : alphapy.Space
        AlphaPy data taxonomy data source and subject.
    fractal : str
        Pandas offset alias.

    Returns
    -------
    df : pandas.DataFrame
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
    r"""Get data from an external feed.

    Parameters
    ----------
    symbol : str
        Pandas offset alias.
    gspace : alphapy.Space
        AlphaPy data taxonomy data source and subject.
    df : pandas.DataFrame
        The raw output dataframe from the market datafeed.
    fractal : str
        Pandas offset alias.
    intraday_data : bool
        If True, then get intraday data.

    Returns
    -------
    df : pandas.DataFrame
        The standardized output dataframe for the market data.

    """

    # convert data to canonical form
    df = convert_data(df, intraday_data)
    # create global pointer to dataframe
    df = assign_global_data(df, symbol, gspace, fractal)
    # return dataframe
    return df


#
# Function get_market_data
#

def get_market_data(model, market_specs, group, lookback_period,
                    intraday_data=False, local_dir=''):
    r"""Get data from an external feed.

    Parameters
    ----------
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

    # Unpack model specifications

    extension = model.specs['extension']
    separator = model.specs['separator']

    # Unpack group elements

    gspace = group.space
    gsubject = gspace.subject
    gsource = gspace.source

    # Determine the feed source

    if intraday_data:
        # intraday data (date and time)
        logger.info("Source [%s] Intraday Data [%s] for %d days",
                    gsource, data_fractal, lookback_period)
    else:
        # daily data or higher (date only)
        logger.info("Source [%s] Daily Data [%s] for %d days",
                    gsource, data_fractal, lookback_period)

    # Get the data from the specified data feed

    df = pd.DataFrame()
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
                                              symbol,
                                              intraday_data,
                                              data_fractal,
                                              from_date,
                                              to_date,
                                              lookback_period)
        else:
            logger.error("Unsupported Data Source: %s", gsource)
        # Now that we have content, standardize the data
        if not df.empty:
            df = df.copy()
            logger.info("Rows: %d [%s]", len(df), data_fractal)
            # reset the index to find the correct datetime column
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            # find date or datetime column
            dt_cols = ['datetime', 'date']
            dt_index = None
            if df.index.name:
                df.index.name = df.index.name.lower()
                dt_index = [x for x in dt_cols if df.index.name == x]
            else:
                dt_column = [x for x in df.columns if x in dt_cols]
            # Set the dataframe's index to the relevant column
            if not dt_index:
                if dt_column:
                    df.set_index(pd.DatetimeIndex(pd.to_datetime(df[dt_column[0]])),
                                                  drop=True, inplace=True)
                else:
                    raise ValueError("Dataframe must have a datetime or date column")
            # drop any remaining date or index columns
            df.drop(columns=dt_cols, inplace=True, errors='ignore')
            df.drop(columns=['index'], inplace=True, errors='ignore')
            # scope dataframe in date range
            df = df.loc[pd.to_datetime(from_date) : pd.to_datetime(to_date)]
            # register the dataframe in the global namespace
            df = standardize_data(symbol, gspace, df, data_fractal, intraday_data)
            # resample data and drop any NA values
            for ff in feature_fractals:
                if ff != data_fractal:
                    df_rs = df.resample(ff).agg({'open'   : 'first',
                                                 'high'   : 'max',
                                                 'low'    : 'min',
                                                 'close'  : 'last',
                                                 'volume' : 'sum'})
                    df_rs.dropna(axis=0, how='any', inplace=True)
                    logger.info("Rows: %d [%s] resampled", len(df_rs), ff)
                    # standardize resampled data
                    intraday_fractal = any(substring in ff for substring in PD_INTRADAY_OFFSETS)
                    df_rs = standardize_data(symbol, gspace, df_rs, ff, intraday_fractal)
        else:
            logger.info("No DataFrame for %s", symbol.upper())
            remove_list.append(symbol)

    # Remove any group members not found

    if remove_list:
        group.remove(remove_list)

    return
