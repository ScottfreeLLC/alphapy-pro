################################################################################
#
# Package   : AlphaPy
# Module    : mflow_main
# Created   : July 11, 2013
#
# Copyright 2020 ScottFree Analytics LLC
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
import os
import pandas as pd
import sys
import yaml

from alphapy.alphapy_main import get_alphapy_config
from alphapy.alphapy_main import main_pipeline
from alphapy.data import get_market_data
from alphapy.frame import write_frame
from alphapy.globals import USEP, BarType
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.globals import PSEP, SSEP
from alphapy.group import Group
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolio
from alphapy.space import Space
from alphapy.system import run_system
from alphapy.system import System
from alphapy.utilities import subtract_days, valid_date
from alphapy.variables import vapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_market_config
#

def get_market_config(alphapy_specs, directory='.'):
    r"""Read the configuration file for MarketFlow.

    Parameters
    ----------
    alphapy_specs : dict
        The specifications for AlphaPy.
    directory : str
        The location of the configuration file.

    Returns
    -------
    specs : dict
        The parameters for controlling MarketFlow.

    """

    logger.info("MarketFlow Configuration")

    # Read the configuration file

    full_path = SSEP.join([directory, 'config', 'market.yml'])
    with open(full_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Store configuration parameters in dictionary

    specs = {}

    #
    # Section: market [this section must be first]
    #

    # Fractals must conform to the pandas offset format
    fractal = cfg['market']['data_fractal']
    try:
        data_fractal_td = pd.to_timedelta(fractal)
    except:
        raise ValueError("Fractal [%s] is an invalid pandas offset" % fractal)
    specs['data_fractal'] = fractal

    data_history = cfg['market']['data_history']
    if not data_history:
        data_history = 0

    start_date = cfg['market']['data_start_date']
    end_date = cfg['market']['data_end_date']

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

    data_directory = cfg['market']['data_directory']
    dir_exists = os.path.isdir(data_directory)
    if dir_exists:
        specs['data_directory'] = data_directory
    else:
        raise ValueError("Directory %s does not exist" % data_directory)
 
    specs['data_history'] = data_history
    specs['data_start_date'] = start_date
    specs['data_end_date'] = end_date

    specs['forecast_period'] = cfg['market']['forecast_period']
    specs['predict_history'] = cfg['market']['predict_history']
    specs['schema'] = cfg['market']['schema']
    specs['subschema'] = cfg['market']['subschema']
    specs['api_key_name'] = cfg['market']['api_key_name']
    specs['api_key'] = cfg['market']['api_key']
    specs['subject'] = cfg['market']['subject']
    specs['target_group'] = cfg['market']['target_group']
    specs['create_model'] = cfg['market']['create_model']
    specs['run_system'] = cfg['market']['run_system']

    # Set API Key environment variable
    if specs['api_key']:
        os.environ[specs['api_key_name']] = specs['api_key']

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
    # Section: portfolio
    #

    logger.info("Getting Portfolio Parameters")
    try:
        specs['portfolio'] = cfg['portfolio']
    except:
        raise ValueError("No Portfolio Parameters Found")

    #
    # Section: system
    #

    logger.info("Getting System Parameters")

    systems = [x for x in alphapy_specs['systems']]
    try:
        specs['system'] = cfg['system']
        system_name = specs['system']['name']
        if system_name in systems:
            specs['system']['longentry'] = alphapy_specs['systems'][system_name]['longentry']
            specs['system']['longexit'] = alphapy_specs['systems'][system_name]['longexit']
            specs['system']['shortentry'] = alphapy_specs['systems'][system_name]['shortentry']
            specs['system']['shortexit'] = alphapy_specs['systems'][system_name]['shortexit']
        else:
            raise ValueError("System %s not found in systems.yml" % system_name)
    except:
        raise ValueError("No System Parameters Found")

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
    logger.info('api_key          = %s', specs['api_key'])
    logger.info('api_key_name     = %s', specs['api_key_name'])
    logger.info('bar_type         = %s', specs['bar_type'])
    logger.info('create_model     = %r', specs['create_model'])
    logger.info('data_directory   = %s', specs['data_directory'])
    logger.info('data_end_date    = %s', specs['data_end_date'])
    logger.info('data_fractal     = %s', specs['data_fractal'])
    logger.info('data_start_date  = %s', specs['data_start_date'])
    logger.info('data_history     = %d', specs['data_history'])
    logger.info('features         = %s', specs['features'])
    logger.info('forecast_period  = %d', specs['forecast_period'])
    logger.info('fractals         = %s', specs['fractals'])
    logger.info('portfolio        = %s', specs['portfolio'])
    logger.info('predict_history  = %s', specs['predict_history'])
    logger.info('run_system       = %r', specs['run_system'])
    logger.info('schema           = %s', specs['schema'])
    logger.info('subject          = %s', specs['subject'])
    logger.info('subschema        = %s', specs['subschema'])
    logger.info('system           = %s', specs['system'])
    logger.info('target_group     = %s', specs['target_group'])

    # Market Specifications
    return specs


#
# Function set_model_targets
#

def set_model_targets(model, dfs, fractals, system_specs, forecast_period, predict_history):
    r"""Set the model return targets.

    First, the target value is lagged for the ``forecast_period``. Each frame
    is split along the ``predict_date`` from the ``analysis``, and finally
    the train and test files are generated.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    dfs : list
        The list of pandas dataframes to analyze.
    system_specs : dict
        The system specifications containing the signals.
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

    # Unpack system specifications

    longentry = system_specs['longentry']
    shortentry = system_specs['shortentry']

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
    # We are creating a target variable based on whether the trade was successful.
    # If the trade is profitable, then the target is 1 else 0.
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
        # shift ROI column back by the number of forecast periods
        target_roi = USEP.join(['roi', str(forecast_period), fractals[0]])
        df[target_roi] = df[target_roi].shift(-forecast_period)
        # filter for signal
        df_signal = pd.DataFrame()
        if longentry:
            col_buy = USEP.join([longentry, fractals[0]])
            df_buy = df[df[col_buy] == True]
            df_buy[target] = df_buy[target_roi] > 0.0
            df_buy.drop(columns=[target_roi], inplace=True)
            df_signal = pd.concat([df_signal, df_buy])
        if shortentry:
            col_sell = USEP.join([shortentry, fractals[0]])
            df_sell = df[df[col_sell] == True]
            df_sell[target] = df_sell[target_roi] < 0.0
            df_sell.drop(columns=[target_roi], inplace=True)
            df_signal = pd.concat([df_signal, df_sell])
        # get frame subsets
        if predict_mode:
            new_predict = df_signal.loc[(df_signal.index >= split_date) & (df_signal.index <= last_date)]
            if len(new_predict) > 0:
                predict_frame = predict_frame.append(new_predict)
            else:
                logger.info("%s Prediction Frame has zero rows. Check prediction date.", symbol.upper())
        else:
            # split data into train and test
            new_train = df_signal.loc[(df_signal.index >= train_date) & (df_signal.index < predict_date)]
            if not new_train.empty:
                # check if target column has NaN values
                nan_count = new_train[target].isnull().sum()
                if nan_count > 0:
                    logger.info("%s has %d train records with a NaN target.", symbol.upper(), nan_count)
                # drop records with NaN values in target column
                new_train = new_train.dropna(subset=[target])
                train_frame = train_frame.append(new_train)
                # get test frame
                new_test = df_signal.loc[(df_signal.index >= predict_date) & (df_signal.index <= last_date)]
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

    # Get model specifications

    predict_mode = model.specs['predict_mode']

    # Get market specifications

    create_model = market_specs['create_model']
    data_history = market_specs['data_history']
    forecast_period = market_specs['forecast_period']
    fractals = market_specs['fractals']
    data_fractal = market_specs['data_fractal']
    functions = market_specs['functions']
    predict_history = market_specs['predict_history']
    schema = market_specs['schema']
    subject = market_specs['subject']
    target_group = market_specs['target_group']
    run_sys = market_specs['run_system']

    # Get system specifications

    system_specs = market_specs['system']
    system_name = system_specs['name']
    algo = system_specs['algo']
    prob_min = system_specs['prob_min']
    prob_max = system_specs['prob_max']
    longentry = system_specs['longentry']
    longexit = system_specs['longexit']
    shortentry = system_specs['shortentry']
    shortexit = system_specs['shortexit']
    holdperiod = system_specs['holdperiod']
    trade_fractal = fractals[0]

    # Set the target group and space

    group = Group.groups[target_group]
    group.space = Space(subject, schema, trade_fractal)
    logger.info("Group Space: %s", group.space)
    logger.info("All Symbols: %s", group.members)

    # Determine whether or not this is an intraday analysis.

    intraday = any(substring in data_fractal for substring in PD_INTRADAY_OFFSETS)

    # Get stock data. If we can't get all the data, then
    # predict_history resets to the actual history obtained.

    lookback = predict_history if predict_mode else data_history
    get_market_data(model, market_specs, group, lookback, intraday)

    # Apply the features to all frames.

    target_roi = USEP.join(['roi', str(forecast_period)])
    market_specs['features'][trade_fractal].append(target_roi)
    if longentry:
        market_specs['features'][trade_fractal].append(longentry)
    if shortentry:
        market_specs['features'][trade_fractal].append(shortentry)
    dfs = vapply(group, market_specs, functions)

    # Run an analysis to create the model.

    if create_model:
        logger.info("Creating Model")
        # set model targets
        set_model_targets(model, dfs, fractals, system_specs, forecast_period, predict_history)
    else:
        logger.info("No Model Created")

    # Run the AlphaPy model pipeline
    model = main_pipeline(alphapy_specs, model)

    # Run a system

    if run_sys:
        logger.info("Running System %s", system_name)
        logger.info("Algorithm        : %s", algo)
        logger.info("Probability Min  : %s", prob_min)
        logger.info("Probability Max  : %s", prob_max)
        logger.info("Long Entry       : %s", longentry)
        logger.info("Long Exit        : %s", longexit)
        logger.info("Short Entry      : %s", shortentry)
        logger.info("Short Exit       : %s", shortexit)
        logger.info("Hold Period      : %s", holdperiod)
        logger.info("Fractal          : %s", trade_fractal)
        # create and run the system
        system = System(system_name, algo, prob_min, prob_max,
                        longentry, longexit, shortentry, shortexit,
                        holdperiod, trade_fractal)
        tfs = run_system(model, system, forecast_period, group, intraday)
        # generate a portfolio
        if tfs.empty:
            logger.info("No trades to generate a portfolio")
        else:
            portfolio_specs = market_specs['portfolio']
            gen_portfolio(model, portfolio_specs, system_name, group, tfs)
    else:
        logger.info("System Not Run")

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

    model_specs = get_model_config()
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
    market_specs = get_market_config(alphapy_specs)

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
