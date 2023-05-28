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
import os
import pandas as pd
import shutil
import sys

from alphapy.alphapy_main import get_alphapy_config
from alphapy.alphapy_main import main_pipeline
from alphapy.data import get_market_data
from alphapy.frame import Frame
from alphapy.frame import frame_name
from alphapy.frame import write_frame
from alphapy.globals import ModelType
from alphapy.globals import LOFF, ROFF, SSEP, USEP
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.group import Group
from alphapy.metalabel import add_vertical_barrier
from alphapy.metalabel import get_bins
from alphapy.metalabel import get_daily_vol
from alphapy.metalabel import get_events
from alphapy.metalabel import get_t_events
from alphapy.mflow_server import get_market_config
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolios
from alphapy.space import Space
from alphapy.system import run_system
from alphapy.system import System
from alphapy.system import SystemRank
from alphapy.transforms import netreturn
from alphapy.utilities import datetime_stamp
from alphapy.utilities import most_recent_file
from alphapy.utilities import subtract_days, valid_date
from alphapy.variables import vapply


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function set_targets_class
#

def set_targets_class(model, df, system_specs):
    r"""Set classification targets

    1. Extract the signal for long and short entries.
    2. Run the Triple Barrier Method analysis.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    df : pandas.DataFrame
        The dataframe to assign metalabels.
    system_specs : dict
        Trade management specifications.

    Returns
    -------
    df_meta : pandas.DataFrame
        The dataframe containing TBM returns, target, and labels

    """

    logger.info("Setting Classification Targets")

    # Unpack model specifications
    target = model.specs['target']

    # Unpack trading specifications

    signal_long = system_specs['signal_long']
    signal_short = system_specs['signal_short']
    predict_history = system_specs['predict_history']
    forecast_period = system_specs['forecast_period']
    profit_factor = system_specs['profit_factor']
    stoploss_factor = system_specs['stoploss_factor']
    minimum_return = system_specs['minimum_return']
    trade_fractal = system_specs['fractal']

    # Find the patterns (signals) in the dataframe.
    
    nrows = df.shape[0]

    if signal_long:
        long_col = USEP.join([signal_long, trade_fractal])
        long_label = 1
        df.loc[df[long_col], 'side'] = long_label
        npats = df[df['side'] == long_label].shape[0]
        logger.info("%d Long Patterns Found in %d Rows", npats, nrows)

    if signal_short:
        short_col = USEP.join([signal_short, trade_fractal])
        short_label = -1
        df.loc[df[short_col], 'side'] = short_label
        npats = df[df['side'] == short_label].shape[0]
        logger.info("%d Short Patterns Found in %d Rows", npats, nrows)

    # Lag the signal.
    df['side'] = df['side'].shift(1)

    # Get closing values for the trading fractal.

    close_col = USEP.join(['close', trade_fractal])
    ds_close = df[close_col]

    # Get daily volatility.
    daily_vol = get_daily_vol(ds_close, p=predict_history)

    # Get the CUSUM events.
    cusum_events = get_t_events(ds_close, threshold=minimum_return)

    # Establish the vertical barriers.
    vertical_barriers = add_vertical_barrier(cusum_events, ds_close, num_days=forecast_period)

    # Set the Triple Barrier Method (TBM) events.

    df_tbm = get_events(ds_close,
                        cusum_events,
                        [profit_factor, stoploss_factor],
                        daily_vol,
                        minimum_return,
                        vertical_barriers,
                        df['side'])

    # Assign labels based on returns.
    df_labels = get_bins(df_tbm, ds_close)

    # Evaluate the primary model.
    pass

    # Filter the dataframe with the events for the secondary model.

    df_meta = df.loc[df_labels.index, :].copy()
    df_meta[target] = df_labels[target]

    return df_meta


#
# Function set_targets_ranking
#

def set_targets_ranking(model, df, ranking_specs):
    r"""Set the Learn-to-Rank (LTR) targets.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    df : pandas.DataFrame
        The dataframe for ranking.
    ranking_specs : dict
        Ranking model specifications.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the shifted target column.

    """

    logger.info("Setting Learn-To-Rank (LTR) Targets")

    # Unpack model specifications
    target = model.specs['target']

    # Unpack ranking specifications
    forecast_period = ranking_specs['forecast_period']

    # Shift the target column
    df[target] = df[target].shift(-forecast_period)

    return df


#
# Function prepare_data
#

def prepare_data(model, dfs, market_specs):
    r"""Prepare the model for training and validation.

    Parameters
    ----------
    model : alphapy.Model
        The model specifications.
    dfs : list
        The list of pandas dataframes to analyze.
    market_specs : dict
        Portfolio and system specifications.

    """

    logger.info("Preparing the Model")

    # Unpack model data

    test_file = model.test_file
    train_file = model.train_file

    # Unpack model specifications

    run_dir = model.specs['run_dir']
    extension = model.specs['extension']
    model_type = model.specs['model_type']
    predict_date = model.specs['predict_date']
    predict_mode = model.specs['predict_mode']
    rank_group_id = model.specs['rank_group_id']
    rank_group_size = model.specs['rank_group_size']
    seed = model.specs['seed']
    separator = model.specs['separator']
    target = model.specs['target']
    train_date = model.specs['train_date']

    # Unpack market specifications

    system_specs = market_specs['system']
    ranking_specs = market_specs['ranking']
    predict_history = market_specs['predict_history']
    forecast_period = market_specs['forecast_period']

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
    # For the Classification Metamodel, we are creating a target variable
    # based on whether the trade was successful. If the trade is profitable,
    # then the target is 1 else 0. Note that we are using the side feature
    # (trade direction) as input into the model.
    #
    # For the Learn-To-Rank Model, the target variable is typically
    # represented by returns over a certain period of time.
    #

    for df_in in dfs:
        # subset each individual frame and add to the master frame
        symbol = df_in['symbol'].iloc[0].upper()
        first_date = df_in.index[0]
        last_date = df_in.index[-1]
        logger.info("Analyzing %s from %s to %s", symbol, first_date, last_date)
        if not df_in.empty:
            # set model targets based on model type
            if model_type == ModelType.ranking:
                df_out = set_targets_ranking(model, df_in, ranking_specs)
            elif model_type == ModelType.classification:
                df_out = set_targets_class(model, df_in, system_specs)
            else:
                raise ValueError("Unsupported Model Type")
            # split the dataframe
            if predict_mode:
                new_predict = df_out.loc[(df_out.index >= split_date) & (df_out.index <= last_date)].copy()
                if len(new_predict) > 0:
                    predict_frame = pd.concat([predict_frame, new_predict])
                else:
                    logger.info("%s Prediction Frame has zero rows. Check prediction date.", symbol)
            else:
                # split data into train and test
                new_train = df_out.loc[(df_out.index >= train_date) & (df_out.index < predict_date)].copy()
                if not new_train.empty:
                    # check if target column has NaN values
                    nan_count = new_train[target].isnull().sum()
                    if nan_count > 0:
                        logger.info("%s has %d train records with a NaN target.", symbol, nan_count)
                    # drop records with NaN values in target column
                    new_train = new_train.dropna(subset=[target])
                    train_frame = pd.concat([train_frame, new_train])
                    # get test frame
                    new_test = df_out.loc[(df_out.index >= predict_date) & (df_out.index <= last_date)]
                    if not new_test.empty:
                        # check if target column has NaN values
                        nan_count = new_test[target].isnull().sum()
                        forecast_check = forecast_period - 1
                        if nan_count != forecast_check:
                            logger.info("%s has %d test records with a NaN target.", symbol, nan_count)
                        # drop records with NaN values in target column
                        new_test = new_test.dropna(subset=[target])
                        # append selected records to the test frame
                        test_frame = pd.concat([test_frame, new_test])
                    else:
                        logger.info("%s Testing Frame has zero rows. Check prediction date.", symbol)
                else:
                    logger.info("%s Training Frame has zero rows. Check data source.", symbol)
        else:
            logger.info("%s Dataframe is empty", symbol)

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

    # Take random samples of each dataframe.

    if model_type == ModelType.ranking:
        logger.info("Random Sampling %d Rows for Ranking", rank_group_size)
        if not test_frame.empty:
            test_frame = test_frame.groupby(rank_group_id).sample(n=rank_group_size, random_state=seed)
        if not predict_mode and not train_frame.empty:
            train_frame = train_frame.groupby(rank_group_id).sample(n=rank_group_size, random_state=seed)

    # Write out the frames for input into the AlphaPy pipeline

    directory = SSEP.join([run_dir, 'input'])
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

    # Get model specifications

    model_type = model.specs['model_type']
    predict_mode = model.specs['predict_mode']
    target = model.specs['target']

    # Get data specifications

    cohort_group = market_specs['cohort_group']
    data_fractal = market_specs['data_fractal']
    data_history = market_specs['data_history']
    data_source = market_specs['data_source']
    forecast_period = market_specs['forecast_period']
    fractals = market_specs['fractals']
    functions = market_specs['functions']
    predict_history = market_specs['predict_history']
    subject = market_specs['subject']
    target_group = market_specs['target_group']
    trade_fractal = fractals[0]

    # Get AlphaPy specifications

    data_dir = alphapy_specs['data_dir']
    systems = alphapy_specs['systems']

    # Get section specifications

    system_specs = market_specs['system']
    system_name = system_specs['system_name']
    signal_long = systems[system_name]['long']
    signal_short = systems[system_name]['short']

    # Augment the system specifications.

    system_specs['predict_history'] = predict_history
    system_specs['forecast_period'] = forecast_period
    system_specs['signal_long'] = signal_long
    system_specs['signal_short'] = signal_short
    system_specs['fractal'] = trade_fractal

    ranking_specs = market_specs['ranking']
    ranking_specs['system_name'] = 'ranking'
    ranking_specs['forecast_period'] = forecast_period
    ranking_specs['fractal'] = trade_fractal

    portfolio_specs = market_specs['portfolio']

    # Set the target group and space

    group = Group.groups[target_group]
    group.space = Space(subject, data_source, trade_fractal)
    logger.info("Group Space: %s", group.space)
    logger.info("All Symbols: %s", group.members)

    # Set the cohort group and space

    try:
        group_cohort = Group.groups[cohort_group]
        group_cohort.space = Space(subject, data_source, trade_fractal)
        logger.info("Cohort Group Space: %s", group_cohort.space)
        logger.info("Cohort Symbols: %s", group_cohort.members)
    except:
        group_cohort = None

    # Determine whether or not this is an intraday analysis.

    intraday = any(substring in data_fractal for substring in PD_INTRADAY_OFFSETS)

    # Get the market data. If we can't get all the data, then
    # predict_history resets to the actual history obtained.

    lookback = predict_history if predict_mode else data_history
    local_dir = SSEP.join([data_dir, subject, trade_fractal])
    get_market_data(alphapy_specs, model, market_specs, group,
                    lookback, intraday, local_dir=local_dir)
    if group_cohort:
        get_market_data(alphapy_specs, model, market_specs, group_cohort,
                        lookback, intraday, local_dir=local_dir)

    # Apply the features to all frames, including the signals just for the
    # target fractal.

    market_specs['features'][trade_fractal].extend([signal_long, signal_short])
    dfs = vapply(group, market_specs, functions)

    # Apply the cohort returns to all frames.

    if group_cohort:
        get_cohort_returns(dfs, group_cohort, trade_fractal)

    # Prepare the data based on the model type.
    prepare_data(model, dfs, market_specs)

    # Run the AlphaPy model pipeline.
    model = main_pipeline(alphapy_specs, model)

    # Run the system designated for that model type.

    df_trades = pd.DataFrame()
    if model_type == ModelType.ranking:
        logger.info("System Name     : %s", ranking_specs['system_name'])
        logger.info("Forecast Period : %s", forecast_period)
        logger.info("Algorithm       : %s", ranking_specs['algo'])
        logger.info("Long Rank       : %s", ranking_specs['long_rank'])
        logger.info("Long Score      : %s", ranking_specs['long_score'])
        logger.info("Short Rank      : %s", ranking_specs['short_rank'])
        logger.info("Short Score     : %s", ranking_specs['short_score'])
        logger.info("Trade Fractal   : %s", trade_fractal)
        system = SystemRank(**ranking_specs)
    else:
        logger.info("System Name      : %s", system_specs['system_name'])
        logger.info("Signal Long      : %s", signal_long)
        logger.info("Signal Short     : %s", signal_short)
        logger.info("Target           : %s", target)
        logger.info("Forecast Period  : %s", forecast_period)
        logger.info("Predict History  : %s", predict_history)
        logger.info("Profit Factor    : %s", system_specs['profit_factor'])
        logger.info("Stop Loss Factor : %s", system_specs['stoploss_factor'])
        logger.info("Minimum Return   : %s", system_specs['minimum_return'])
        logger.info("Algorithm        : %s", system_specs['algo'])
        logger.info("Probability Min  : %s", system_specs['prob_min'])
        logger.info("Probability Max  : %s", system_specs['prob_max'])
        logger.info("Trade Fractal    : %s", trade_fractal)
        system = System(**system_specs)

    # Run the system and generate the portfolio.

    df_trades, df_baseline = run_system(model, system, group, intraday)
    if df_trades.empty:
        logger.info("No trades to generate a portfolio")
    else:
        gen_portfolios(model, system_name, portfolio_specs, group,
                       df_trades, df_baseline)

    # Return the completed model.
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
    parser.add_argument('--rundir', dest='run_dir',
                        help="run directory is in the format: run_YYYYMMDD_hhmmss",
                        required=False)
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
        train_date = datetime.datetime(1900, 1, 1).strftime("%Y-%m-%d")

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

    # If not in prediction mode, then create the training infrastructure.

    if not model_specs['predict_mode']:
        # create the directory infrastructure if necessary
        output_dirs = ['config', 'data', 'runs']
        for od in output_dirs:
            output_dir = SSEP.join([model_specs['directory'], od])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # create the run directory
        dt_stamp = datetime_stamp()
        run_dir_name = USEP.join(['run', dt_stamp])
        run_dir = SSEP.join([model_specs['directory'], 'runs', run_dir_name])
        os.makedirs(run_dir)
        # create the subdirectories of the runs directory
        sub_dirs = ['config', 'input', 'model', 'output', 'plots', 'systems']
        for sd in sub_dirs:
            output_dir = SSEP.join([run_dir, sd])
            if not os.path.exists(output_dir):
                logger.info("Creating directory %s", output_dir)
                os.makedirs(output_dir)
        # copy the market file to the config directory
        file_names = ['model.yml', 'market.yml']
        for file_name in file_names:
            src_file = SSEP.join([model_specs['directory'], 'config', file_name])
            dst_file = SSEP.join([run_dir, 'config', file_name])
            shutil.copyfile(src_file, dst_file)
    else:
        run_dir = args.run_dir if args.run_dir else None
        if not run_dir:
            # get latest directory
            search_dir = SSEP.join([model_specs['directory'], 'runs'])
            run_dir = most_recent_file(search_dir, 'run_*')
    model_specs['run_dir'] = run_dir

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
