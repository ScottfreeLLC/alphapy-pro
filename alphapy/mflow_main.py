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

from alphapy.alias import Alias
from alphapy.analysis import Analysis
from alphapy.analysis import run_analysis
from alphapy.data import get_market_data
from alphapy.globals import USEP, BarType
from alphapy.globals import PD_INTRADAY_OFFSETS
from alphapy.globals import PSEP, SSEP
from alphapy.group import Group
from alphapy.variables import Variable
from alphapy.variables import vapply
from alphapy.model import get_model_config
from alphapy.model import Model
from alphapy.portfolio import gen_portfolio
from alphapy.space import Space
from alphapy.system import run_system
from alphapy.system import System
from alphapy.utilities import valid_date

import argparse
import datetime
import logging
import os
import pandas as pd
import yaml


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_market_config
#

def get_market_config():
    r"""Read the configuration file for MarketFlow.

    Parameters
    ----------
    None : None

    Returns
    -------
    specs : dict
        The parameters for controlling MarketFlow.

    """

    logger.info("MarketFlow Configuration")

    # Read the configuration file

    full_path = SSEP.join([PSEP, 'config', 'market.yml'])
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
    # Section: Bar Type, Fractals and Features
    #

    logger.info("Getting Bar Type")

    try:
        specs['bar_type'] = BarType[cfg['bar_type']]
    except:
        logger.info("No valid bar type was specified. Default: time")
        specs['bar_type'] = BarType.time

    logger.info("Getting Fractals")

    if len(cfg['fractals']) > 1 and specs['bar_type'] != BarType.time:
        raise ValueError("Multiple Fractals valid only on time bars")
    
    fractals = {}
    for frac in cfg['fractals']:
        try:
            td = pd.to_timedelta(frac)
            fractals[td] = frac
        except:
            raise ValueError("Fractal [%s] is an invalid pandas offset" % frac)
    # sort by ascending fractal
    fractals_sorted = dict(sorted(fractals.items()))
    # store features sorted by fractal
    feature_fractals = list(fractals_sorted.values())
    # first (lowest) feature fractal must be >= data fractal
    if list(fractals_sorted)[0] < data_fractal_td:
        raise ValueError("Lowest feature fractal [%s] must >= data fractal [%s]" %
                         (feature_fractals[0], specs['data_fractal']))
    # assign to market specifications
    specs['fractals'] = feature_fractals
 
    logger.info("Getting Features")
    specs['features'] = cfg['features']

    # Create the subject/schema/fractal namespace

    sspecs = [specs['subject'], specs['schema'], feature_fractals[0]]
    space = Space(*sspecs)

    #
    # Section: groups
    #

    logger.info("Defining Groups in Space: %s" % space)
    try:
        for g, m in list(cfg['groups'].items()):
            Group(g, space)
            Group.groups[g].add(m)
    except:
        raise ValueError("No Groups Found")

    #
    # Section: aliases
    #

    logger.info("Defining Aliases")
    try:
        for k, v in list(cfg['aliases'].items()):
            Alias(k, v)
    except:
        raise ValueError("No Aliases Found")

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
    try:
        specs['system'] = cfg['system']
    except:
        raise ValueError("No System Parameters Found")

    #
    # Section: variables
    #

    logger.info("Defining User Variables")
    try:
        for k, v in list(cfg['variables'].items()):
            Variable(k, v)
    except:
        raise ValueError("No Variables Found")

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
# Function market_pipeline
#

def market_pipeline(model, market_specs):
    r"""AlphaPy MarketFlow Pipeline

    Parameters
    ----------
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

    logger.info("Running MarketFlow Pipeline")

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

    # Set the target group

    group = Group.groups[target_group]
    logger.info("All Symbols: %s", group.members)

    # Determine whether or not this is an intraday analysis.

    intraday = any(substring in data_fractal for substring in PD_INTRADAY_OFFSETS)

    # Get stock data. If we can't get all the data, then
    # predict_history resets to the actual history obtained.

    lookback = predict_history if predict_mode else data_history
    get_market_data(model, market_specs, group, lookback, intraday)

    # Apply the features to all frames.

    target_roi = USEP.join(['roi', str(forecast_period)])
    market_specs['features'].append(target_roi)
    if longentry:
        market_specs['features'].append(longentry)
    if shortentry:
        market_specs['features'].append(shortentry)
    dfs = vapply(group, market_specs, functions)

    # Run an analysis to create the model.

    if create_model:
        logger.info("Creating Model")
        # run the analysis, which calls the model pipeline
        anal = Analysis(model, group)
        run_analysis(anal, dfs, fractals, system_specs, forecast_period, predict_history)
    else:
        logger.info("No Model Created")

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

    # Read stock configuration file
    market_specs = get_market_config()

    # Read model configuration file

    model_specs = get_model_config()
    model_specs['predict_mode'] = args.predict_mode
    model_specs['predict_date'] = predict_date
    model_specs['train_date'] = train_date

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
    model = market_pipeline(model, market_specs)

    # Complete the pipeline

    logger.info('*'*80)
    logger.info("MarketFlow End")
    logger.info('*'*80)


#
# MAIN PROGRAM
#

if __name__ == "__main__":
    main()
