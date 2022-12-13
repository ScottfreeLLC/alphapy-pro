"""
Package   : AlphaPy
Module    : mflow_main
Created   : July 11, 2013

Copyright 2022 ScottFree Analytics LLC
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
# Suppress Warnings
#

import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


#
# Imports
#

import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
import sys
import yaml

from alphapy.alphapy_main import get_alphapy_config
from alphapy.alphapy_main import main_pipeline
from alphapy.data import get_market_data
from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import write_frame
from alphapy.globals import LOFF, ROFF, SSEP, USEP, BarType
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.group import Group
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolio
from alphapy.space import Space
from alphapy.system import run_system
from alphapy.system import System
from alphapy.transforms import netreturn
from alphapy.utilities import subtract_days, valid_date
from alphapy.variables import vapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_market_config
#

def get_market_config(directory='.'):
    r"""Read the configuration file for MarketFlow.

    Parameters
    ----------
    directory : str
        The location of the configuration file.

    Returns
    -------
    cfg : dict
        The original configuration specification.
    specs : dict
        The parameters for controlling MarketFlow.

    """

    logger.info('*'*80)
    logger.info("MarketFlow Configuration")

    # Read the configuration file

    full_path = SSEP.join([directory, 'config', 'market.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Store configuration parameters in dictionary

    specs = {}

    #
    # Section: trading
    #

    logger.info("Getting Trading Parameters")
    try:
        specs['trading'] = cfg['trading']
    except:
        raise ValueError("No Trading Parameters Found")

    #
    # Section: data
    #

    specs['data_source'] = cfg['data']['data_source']

    # Fractals must conform to the pandas offset format

    fractal = cfg['data']['data_fractal']
    try:
        data_fractal_td = pd.to_timedelta(fractal)
    except:
        raise ValueError("Fractal [%s] is an invalid pandas offset" % fractal)
    specs['data_fractal'] = fractal

    data_history = cfg['data']['data_history']
    if not data_history:
        data_history = 0

    start_date = cfg['data']['data_start_date']
    end_date = cfg['data']['data_end_date']

    if not start_date or not end_date:
        data_history_dt = pd.to_timedelta(data_history, unit='d')
        today_date_dt = pd.to_datetime('today')
        if start_date:
            start_date_dt = pd.to_datetime(start_date)
            if data_history > 0:
                end_date_dt = start_date_dt + data_history_dt
                end_date_dt = today_date_dt if end_date_dt > today_date_dt else end_date_dt
            else:
                end_date_dt = today_date_dt
        elif data_history > 0:
            if end_date:
                end_date_dt = pd.to_datetime(end_date)
                start_date_dt = end_date_dt - data_history_dt
            else:
                end_date_dt = today_date_dt
                start_date_dt = end_date_dt - data_history_dt
        else:
            raise ValueError("Either parameter data_start_date or data_history is required")
        start_date = start_date_dt.strftime('%Y-%m-%d')
        end_date = end_date_dt.strftime('%Y-%m-%d')

    specs['data_history'] = data_history
    specs['data_start_date'] = start_date
    specs['data_end_date'] = end_date
    specs['predict_history'] = cfg['data']['predict_history']
    specs['subject'] = cfg['data']['subject']
    specs['target_group'] = cfg['data']['target_group']
    specs['cohort_group'] = cfg['data']['cohort_group']

    #
    # Section: Bar Type, Features and Fractals
    #

    logger.info("Getting Bar Type")

    try:
        specs['bar_type'] = BarType[cfg['bar_type']]
    except:
        logger.info("No valid bar type was specified. Default: time")
        specs['bar_type'] = BarType.time

    logger.info("Getting Features")
    specs['features'] = cfg['features']

    logger.info("Getting Fractals")

    fractals = list(specs['features'].keys())
    if len(fractals) > 1 and specs['bar_type'] != BarType.time:
        raise ValueError("Multiple Fractals valid only on time bars")
    
    fractal_dict = {}
    for frac in fractals:
        try:
            td = pd.to_timedelta(frac)
            fractal_dict[td] = frac
        except:
            raise ValueError("Fractal [%s] is an invalid pandas offset" % frac)
    
    # sort by ascending fractal
    fractals_sorted = dict(sorted(fractal_dict.items()))
    # store features sorted by fractal
    feature_fractals = list(fractals_sorted.values())
    # first (lowest) feature fractal must be >= data fractal
    if list(fractals_sorted)[0] < data_fractal_td:
        raise ValueError("Lowest feature fractal [%s] must >= data fractal [%s]" %
                         (feature_fractals[0], specs['data_fractal']))
    # assign to market specifications
    specs['fractals'] = feature_fractals

    #
    # Section: functions
    #

    logger.info("Getting Variable Functions")
    try:
        specs['functions'] = cfg['functions']
    except:
        logger.info("No Variable Functions Found")
        specs['functions'] = {}

    #
    # Log the market parameters
    #

    logger.info('MARKET PARAMETERS:')
    logger.info('bar_type         = %s', specs['bar_type'])
    logger.info('data_source      = %s', specs['data_source'])
    logger.info('data_end_date    = %s', specs['data_end_date'])
    logger.info('data_fractal     = %s', specs['data_fractal'])
    logger.info('data_start_date  = %s', specs['data_start_date'])
    logger.info('data_history     = %d', specs['data_history'])
    logger.info('features         = %s', specs['features'])
    logger.info('fractals         = %s', specs['fractals'])
    logger.info('predict_history  = %d', specs['predict_history'])
    logger.info('subject          = %s', specs['subject'])
    logger.info('target_group     = %s', specs['target_group'])
    logger.info('cohort_group     = %s', specs['cohort_group'])
    logger.info('trading          = %s', specs['trading'])

    # Market Specifications
    return cfg, specs


#
# Function set_model_targets
#

def set_model_targets(model, meta_model, dfs, fractals, forecast_period, predict_history):
    r"""Set the model return targets.

    First, the target value is lagged for the ``forecast_period``. Each frame
    is split along the ``predict_date`` from the ``analysis``, and finally
    the train and test files are generated.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    meta_model : bool
        True if meta model.
    dfs : list
        The list of pandas dataframes to analyze.
    fractals : list
        List of Pandas offset aliases.
    forecast_period : int
        The period for forecasting the target of the analysis.
    predict_history : int
        The number of periods required for lookback calculations.

    """

    # Unpack model data

    test_file = model.test_file
    train_file = model.train_file

    # Unpack model specifications

    directory = model.specs['directory']
    extension = model.specs['extension']
    predict_date = model.specs['predict_date']
    predict_mode = model.specs['predict_mode']
    separator = model.specs['separator']
    target = model.specs['target']
    train_date = model.specs['train_date']

    # Calculate split date

    logger.info("Analysis Dates")
    split_date = subtract_days(predict_date, predict_history)

    # Create dataframes

    if predict_mode:
        # create predict frame
        logger.info("Split Date for Prediction Mode: %s", split_date)
        predict_frame = pd.DataFrame()
    else:
        # create train and test frames
        logger.info("Split Date for Training Mode: %s", predict_date)
        train_frame = pd.DataFrame()
        test_frame = pd.DataFrame()

    #
    # For the metamodel, we are creating a target variable based on
    # whether the trade was successful. If the trade is profitable,
    # then the target is 1 else 0.
    #
    # For long signals, the ROI must be greater than 0.
    # For short signals, the ROI must be less than 0.
    #

    for df in dfs:
        # subset each individual frame and add to the master frame
        symbol = df['symbol'].iloc[0]
        first_date = df.index[0]
        last_date = df.index[-1]
        logger.info("Analyzing %s from %s to %s", symbol.upper(), first_date, last_date)
        if not df.empty:
            # find patterns in dataframe
            rows_old = df.shape[0]
            rows_new = df[target].sum()
            logger.info("%d Patterns Found in %d Rows", rows_new, rows_old)
            if meta_model:
                # shift ROI column back by the number of forecast periods
                roi_col = USEP.join(['roi', str(forecast_period), fractals[0]])
                roi_shift = df[roi_col].shift(-forecast_period)
                df[target] = np.greater(roi_shift, 0.0)
            else:
                # shift target column back by the number of forecast periods
                df[target] = df[target].shift(-forecast_period)
            # get frame subsets
            if predict_mode:
                new_predict = df.loc[(df.index >= split_date) & (df.index <= last_date)].copy()
                if len(new_predict) > 0:
                    predict_frame = predict_frame.append(new_predict)
                else:
                    logger.info("%s Prediction Frame has zero rows. Check prediction date.", symbol.upper())
            else:
                # split data into train and test
                new_train = df.loc[(df.index >= train_date) & (df.index < predict_date)].copy()
                if not new_train.empty:
                    # check if target column has NaN values
                    nan_count = new_train[target].isnull().sum()
                    if nan_count > 0:
                        logger.info("%s has %d train records with a NaN target.", symbol.upper(), nan_count)
                    # drop records with NaN values in target column
                    new_train = new_train.dropna(subset=[target])
                    train_frame = train_frame.append(new_train)
                    # get test frame
                    new_test = df.loc[(df.index >= predict_date) & (df.index <= last_date)]
                    if not new_test.empty:
                        # check if target column has NaN values
                        nan_count = new_test[target].isnull().sum()
                        forecast_check = forecast_period - 1
                        if nan_count != forecast_check:
                            logger.info("%s has %d test records with a NaN target.", symbol.upper(), nan_count)
                        # drop records with NaN values in target column
                        new_test = new_test.dropna(subset=[target])
                        # append selected records to the test frame
                        test_frame = test_frame.append(new_test)
                    else:
                        logger.info("%s Testing Frame has zero rows. Check prediction date.", symbol.upper())
                else:
                    logger.info("%s Training Frame has zero rows. Check data source.", symbol.upper())
        else:
            logger.info("%s Dataframe is empty", symbol.upper())

    # Convert column names from special characters

    def new_col_name(col_name):
        start = col_name.find(LOFF)
        end = col_name.find(ROFF)
        lag_string = col_name[start:end+1]
        lag_value = lag_string[1:-1]
        if lag_value:
            new_name = ''.join([col_name.replace(lag_string, ''), '_lag', lag_value])
        else:
            new_name = col_name
        return new_name

    new_columns = [new_col_name(x) for x in train_frame.columns]
    train_frame.columns = new_columns
    if not test_frame.empty:
        test_frame.columns = new_columns

    # Write out the frames for input into the AlphaPy pipeline

    directory = SSEP.join([directory, 'input'])
    if predict_mode:
        # write out the predict frame
        test_frame.sort_index(inplace=True)
        write_frame(test_frame, directory, test_file, extension, separator,
                    index=True, index_label='date')
    else:
        # write out the train and test frames
        train_frame.sort_index(inplace=True)
        write_frame(train_frame, directory, train_file, extension, separator,
                    index=True, index_label='date')
        test_frame.sort_index(inplace=True)
        write_frame(test_frame, directory, test_file, extension, separator,
                    index=True, index_label='date')
    return


#
# Function get_cohort_returns
#

def get_cohort_returns(dfs, group, fractal):
    r"""Calculate returns for the cohorts.

    Parameters
    ----------
    dfs : list
        The list of pandas dataframes to apply the cohort returns.
    group : alphapy.Group
        The cohort group for calculating returns.
    fractal : str
        Pandas offset alias.

    """

    logger.info("Calculating Cohort Returns")

    # Get cohort group information

    gspace = group.space
    gsubject = gspace.subject
    gsource = gspace.source
    symbols = [item.lower() for item in group.members]

    #
    # For each frame, calculate the difference in returns
    #

    col_roi = USEP.join(['roi', '1', fractal])
    for symbol in symbols:
        fspace = Space(gsubject, gsource, fractal)
        fname = frame_name(symbol.lower(), fspace)
        if fname in Frame.frames:
            df_cohort = Frame.frames[fname].df
            if not df_cohort.empty:
                roi_cohort = netreturn(df_cohort, 'close')
                col_roi_symbol = USEP.join([col_roi, symbol])
                for df in dfs:
                    df[col_roi_symbol] = df[col_roi] - roi_cohort
            else:
                logger.info("Empty Dataframe for %s [%s]", symbol, fractal)
        else:
            logger.info("Dataframe Not Found for %s [%s]", symbol, fractal)               
    return


#
# Function market_pipeline
#

def market_pipeline(alphapy_specs, model, market_specs):
    r"""Market Flow Pipeline

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for controlling the AlphaPy pipeline.
    model : alphapy.Model
        The model object for AlphaPy.
    market_specs : dict
        The specifications for controlling the MarketFlow pipeline.

    Returns
    -------
    model : alphapy.Model
        The final results are stored in the model object.

    Notes
    -----
    (1) Define a group.
    (2) Get the market data.
    (3) Apply system features.
    (4) Create an analysis.
    (5) Run the analysis, which calls AlphaPy.

    """

    logger.info("Running Market Flow Pipeline")
    
    # Get AlphaPy specifications

    data_dir = alphapy_specs['data_dir']

    # Get model specifications

    predict_mode = model.specs['predict_mode']
    target = model.specs['target']

    # Get market specifications

    data_fractal = market_specs['data_fractal']
    data_history = market_specs['data_history']
    data_source = market_specs['data_source']
    fractals = market_specs['fractals']
    trade_fractal = fractals[0]
    functions = market_specs['functions']
    predict_history = market_specs['predict_history']
    subject = market_specs['subject']
    target_group = market_specs['target_group']
    cohort_group = market_specs['cohort_group']    

    # Set the target group and space

    group = Group.groups[target_group]
    group.space = Space(subject, data_source, trade_fractal)
    logger.info("Group Space: %s", group.space)
    logger.info("All Symbols: %s", group.members)

    # Set the cohort group and space

    group_cohort = Group.groups[cohort_group]
    group_cohort.space = Space(subject, data_source, trade_fractal)
    logger.info("Cohort Group Space: %s", group_cohort.space)
    logger.info("Cohort Symbols: %s", group_cohort.members)

    # Determine whether or not this is an intraday analysis.

    intraday = any(substring in data_fractal for substring in PD_INTRADAY_OFFSETS)

    # Get the market data. If we can't get all the data, then
    # predict_history resets to the actual history obtained.

    lookback = predict_history if predict_mode else data_history
    get_market_data(model, market_specs, group, lookback, intraday, local_dir=data_dir)
    get_market_data(model, market_specs, group_cohort, lookback, intraday, local_dir=data_dir)

    # Apply the features to all frames.
    dfs = vapply(group, market_specs, functions)

    # Apply the cohort returns to all frames.
    get_cohort_returns(dfs, group_cohort, trade_fractal)

    # Run an analysis to create the model.

    logger.info("Creating Model")
    # set model targets
    set_model_targets(model, meta_model, dfs, fractals, forecast_period, predict_history)
    # run the AlphaPy model pipeline
    model = main_pipeline(alphapy_specs, model)

    # Run a system

    logger.info("Target           : %s", target)
    logger.info("System Type      : %s", system_type)
    logger.info("Forecast Period  : %s", forecast_period)
    logger.info("Algorithm        : %s", algo)
    logger.info("Prob Minimum     : %s", prob_min)
    logger.info("Prob Maximum     : %s", prob_max)
    logger.info("Trade Fractal    : %s", trade_fractal)
    # create and run the system
    system = System(target, system_type, algo, prob_min, prob_max, forecast_period, trade_fractal)
    tfs = run_system(model, system, forecast_period, group, intraday)
    # generate a portfolio
    if tfs.empty:
        logger.info("No trades to generate a portfolio")
    else:
        trading_specs = market_specs['trading']
        gen_portfolio(model, trading_specs, target, group, tfs)

    # Return the completed model
    return model


#
# Function main
#

def main(args=None):
    r"""MarketFlow Main Program

    Notes
    -----
    (1) Initialize logging.
    (2) Parse the command line arguments.
    (3) Get the market configuration.
    (4) Get the model configuration.
    (5) Create the model object.
    (6) Call the main MarketFlow pipeline.

    Raises
    ------
    ValueError
        Training date must be before prediction date.

    """

    # Argument Parsing

    parser = argparse.ArgumentParser(description="MarketFlow Parser")
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument('--pdate', dest='predict_date',
                        help="prediction date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_argument('--tdate', dest='train_date',
                        help="training date is in the format: YYYY-MM-DD",
                        required=False, type=valid_date)
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--predict', dest='predict_mode', action='store_true')
    parser.add_argument('--train', dest='predict_mode', action='store_false')
    parser.set_defaults(predict_mode=False)
    args = parser.parse_args()

    # Logging

    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="market_flow.log", filemode='a', level=log_level,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Start the pipeline

    logger.info('*'*80)
    logger.info("MarketFlow Start")
    logger.info('*'*80)

    # Set train and predict dates

    if args.train_date:
        train_date = args.train_date
    else:
        train_date = pd.datetime(1900, 1, 1).strftime("%Y-%m-%d")

    if args.predict_date:
        predict_date = args.predict_date
    else:
        predict_date = datetime.date.today().strftime("%Y-%m-%d")

    # Verify that the dates are in sequence.

    if train_date >= predict_date:
        raise ValueError("Training date must be before prediction date")
    else:
        logger.info("Training Date: %s", train_date)
        logger.info("Prediction Date: %s", predict_date)

    # Read model configuration file

    _, model_specs = get_model_config()
    model_specs['predict_mode'] = args.predict_mode
    model_specs['predict_date'] = predict_date
    model_specs['train_date'] = train_date

    # Read AlphaPy root directory

    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)
    else:
        model_specs['alphapy_root'] = alphapy_root

    # Read AlphaPy configuration file
    alphapy_specs = get_alphapy_config(alphapy_root)

    # Read market configuration file
    _, market_specs = get_market_config()

    # Create directories if necessary

    output_dirs = ['config', 'data', 'input', 'model', 'output', 'plots', 'systems']
    for od in output_dirs:
        output_dir = SSEP.join([model_specs['directory'], od])
        if not os.path.exists(output_dir):
            logger.info("Creating directory %s", output_dir)
            os.makedirs(output_dir)

    # Create a model object from the specifications
    model = Model(model_specs)

    # Start the pipeline
    model = market_pipeline(alphapy_specs, model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("MarketFlow End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
