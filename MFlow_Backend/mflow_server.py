################################################################################
#
# Package   : AlphaPy
# Module    : mflow_server
# Created   : February 21, 2021
#
# Copyright 2022 ScottFree Analytics LLC
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
# HOW TO RUN:
#
# export ALPHAPY_ROOT=/Users/markconway/Projects/alphapy-root
# cd /Users/markconway/Projects/alphapy-3.0.0/alphapy
# uvicorn mflow_server:app --reload
#


#projects = alphapy_request(alphapy_specs, 'projects', alphapy_specs)
#projects = sorted(projects, key=str.casefold)
#projects.insert(0, None)
#project = st.sidebar.selectbox("Select Project", projects)

#
# Imports
#

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
from finviz.portfolio import Portfolio
from finviz.screener import Screener
import finnhub
import logging
import os
import pandas as pd
from pathlib import Path
import sys
import uvicorn
import yaml

from alphapy.globals import BarType
from alphapy.globals import SSEP
from alphapy.group import Group
from alphapy.model import get_model_config


#
# Initialize FastAPI
#

app = FastAPI()


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
    # Section: ranking
    #

    logger.info("Getting Ranking Parameters")
    try:
        specs['ranking'] = cfg['ranking']
    except:
        raise ValueError("No Ranking Parameters Found")

    #
    # Section: data
    #

    data_section = cfg['data']
    specs['data_source'] = data_section['data_source']

    # Fractals must conform to the pandas offset format

    fractal = data_section['data_fractal']
    try:
        data_fractal_td = pd.to_timedelta(fractal)
    except:
        raise ValueError("Fractal [%s] is an invalid pandas offset" % fractal)
    specs['data_fractal'] = fractal

    data_history = data_section['data_history']
    if not data_history:
        data_history = 0

    start_date = data_section['data_start_date']
    end_date = data_section['data_end_date']

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
    specs['data_start_time'] = data_section['data_start_time']
    specs['data_end_time'] = data_section['data_end_time']
    specs['forecast_period'] = data_section['forecast_period']
    specs['predict_history'] = data_section['predict_history']
    specs['subject'] = data_section['subject']
    specs['target_group'] = data_section['target_group']
    specs['cohort_group'] = data_section['cohort_group']

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
    logger.info('cohort_group     = %s', specs['cohort_group'])
    logger.info('data_end_date    = %s', specs['data_end_date'])
    logger.info('data_end_time    = %s', specs['data_end_time'])
    logger.info('data_fractal     = %s', specs['data_fractal'])
    logger.info('data_history     = %d', specs['data_history'])
    logger.info('data_source      = %s', specs['data_source'])
    logger.info('data_start_date  = %s', specs['data_start_date'])
    logger.info('data_start_time  = %s', specs['data_start_time'])
    logger.info('features         = %s', specs['features'])
    logger.info('forecast_period  = %d', specs['forecast_period'])
    logger.info('fractals         = %s', specs['fractals'])
    logger.info('portfolio        = %s', specs['portfolio'])
    logger.info('predict_history  = %d', specs['predict_history'])
    logger.info('ranking          = %s', specs['ranking'])
    logger.info('subject          = %s', specs['subject'])
    logger.info('system           = %s', specs['system'])
    logger.info('target_group     = %s', specs['target_group'])

    # Market Specifications
    return cfg, specs


#
# Function get_finviz_portfolios
#

def get_finviz_portfolios():
    finviz_specs = alphapy_specs['sources']['finviz']
    email = finviz_specs['email']
    api_key = finviz_specs['api_key']
    portfolios = finviz_specs['portfolios']
    print(portfolios)

    groups = {}
    for pf in portfolios:
        portfolio = Portfolio(email, api_key, pf)
        if portfolio:
            df = pd.DataFrame(portfolio.data)
            symbols = df['Ticker'].tolist()
            groups[pf] = symbols
        else:
            error_message = f"Could not find FinViz Portfolio: {pf}"
            st.text(error_message)
    return groups


#
# Function get_market_index_groups
#

def get_market_index_groups():
    url = f"https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    df = pd.read_csv(url)
    df.loc[df['symbol'] == '^NDX', 'name'] = 'Nasdaq 100'
    finnhub_client = finnhub.Client(api_key=alphapy_specs['sources']['finnhub']['api_key'])

    groups = {}
    for _, row in df.iterrows():
        group_symbol = row['symbol']
        group_name = row['name']
        group_dict = finnhub_client.indices_const(symbol=group_symbol)
        groups[group_name] = group_dict['constituents']
    return groups


#
# Function get_market_inputs
#

def get_market_inputs(input_dict, select_dict):

    # Define the market inputs map with input type and default values

    inputs_map = {
        'data_source' : [st.selectbox, select_dict['data_source']],
        'data_directory' : [st.text_input],
        'data_fractal' : [st.text_input, '5min'],
        'data_history' : [st.number_input, 1, 10000],
        'forecast_period' : [st.number_input, 1, 100],
        'predict_history' : [st.number_input, 1, 200],
        'subject' : [st.selectbox, select_dict['subject']],
        'capital' : [st.number_input, 10000, 1000000],
        'margin' : [st.number_input, 0.01, 1.0],
        'cost_bps' : [st.number_input, 0.0, 100.0],
        'algo' : [st.selectbox, select_dict['algo']],
        'prob_min' : [st.number_input, 0.0, 1.0],
        'prob_max' : [st.number_input, 0.0, 1.0],
        'holdperiod' : [st.number_input],
        'bar_type' : [st.selectbox, select_dict['bar_type']],
        'fractals' : [st.selectbox, select_dict['fractals']],
        'features' : [st.multiselect, select_dict['features']]
        }
    
    # Return the mapping information
    return {k:v for k, v in inputs_map.items() if k in input_dict.keys()}


#
# FastAPI Startup
#

@app.on_event("startup")
async def startup_event():
    # Initialize Logging
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="mflow_server.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    # Start the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server Start")
    logger.info('*'*80)
    # Get the AlphaPy environment variable
    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)
    # Finish Startup
    return


#
# Get groups
#

@app.get("/groups")
def request_groups():
    return Group.groups


#
# Get market specifications
#

@app.get("/market_config")
def request_market_config(project_root):
    cfg, specs = get_market_config(project_root)
    return cfg, specs


#
# Get model specifications
#

@app.get("/model_config")
def request_model_config(project_root):
    cfg, specs = get_model_config(project_root)
    return cfg, specs


#
# Get paths
#

@app.get("/paths")
def request_paths(alphapy_specs):
    root_directory = alphapy_specs['mflow']['project_root']
    paths = []
    for path in Path(root_directory).rglob('market.yml'):
        paths.append(path)
    return paths


#
# Get projects
#

@app.get("/projects")
def request_projects(alphapy_specs):
    root_directory = alphapy_specs['mflow']['project_root']
    projects = []
    for path in Path(root_directory).rglob('market.yml'):
        path_str = str(path).split('/')
        projects.append(path_str[-3])
    return projects


#
# Shut down the application
#

@app.on_event("shutdown")
def shutdown_event():
    # Stop the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server End")
    logger.info('*'*80)


#
# Main Program
#

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)