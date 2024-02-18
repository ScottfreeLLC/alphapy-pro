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
# mflow
#


#projects = alphapy_request(alphapy_specs, 'projects', alphapy_specs)
#projects = sorted(projects, key=str.casefold)
#projects.insert(0, None)
#project = st.sidebar.selectbox("Select Project", projects)

#
# Imports
#

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
# Function run_project
#

def run_project(project):

    # Vet the model and market specifications

    project_root = '/'.join([alphapy_specs['mflow']['project_root'], project])
    model_specs, model_dict = alphapy_request(alphapy_specs, 'model_config', project_root)
    market_specs, market_dict = alphapy_request(alphapy_specs, 'market_config', project_root)

    # Determine the source of market groups

    text_ap = 'Market Flow'
    text_fp = 'Finviz Portfolio'
    text_mi = 'Market Index'
    screener = st.sidebar.radio("Group Source", (text_ap, text_fp, text_mi))

    col1, col2, col3, col4 = st.columns(4)

    if screener == text_ap:
        groups = alphapy_request(alphapy_specs, 'groups')
    elif screener == text_fp:
        groups = get_finviz_portfolios()
    elif screener == text_mi:
        groups = get_market_index_groups()

    # Select the group to test

    group_list = list(groups.keys())
    if screener == text_ap:
        group_default = market_specs['data']['target_group']
        group_list.remove(group_default)
        group_list.insert(0, group_default)
    group_text = ' '.join(['Select', screener, 'Group'])
    group = col1.selectbox(group_text, group_list)

    # Select the date range (market:data_start_date and market:data_end_date)
    # If the configuration variable market:data_history is set, then calculate the dates.

    start_date_default = market_specs['data']['data_start_date']
    end_date_default = market_specs['data']['data_end_date']
    data_history_default = market_specs['data']['data_history']
    today = datetime.now()
    if data_history_default and start_date_default and end_date_default:
        from_date = start_date_default
        to_date = end_date_default
    elif data_history_default and start_date_default and not end_date_default:
        from_date = start_date_default
        to_date = today
    elif data_history_default and not start_date_default and end_date_default:
        from_date = end_date_default - timedelta(days=data_history_default)
        to_date = end_date_default
    elif data_history_default and not start_date_default and not end_date_default:
        from_date = today - timedelta(days=data_history_default)
        to_date = today
    elif not data_history_default and start_date_default and end_date_default:
        from_date = start_date_default
        to_date = end_date_default
    elif not data_history_default and start_date_default and not end_date_default:
        from_date = start_date_default
        to_date = today
    elif not data_history_default and not start_date_default and end_date_default:
        from_date = end_date_default - timedelta(days=365)
        to_date = end_date_default
    elif not data_history_default and not start_date_default and not end_date_default:
        from_date = today - timedelta(days=365)
        to_date = today

    col2.date_input('From', from_date)
    col2.date_input('To', to_date)

    # Select the symbols

    group_container = col1.container()
    select_all = col1.checkbox("Select all")
    select_text = "Select one or more symbols:"
    if screener == text_ap:
        symbols = sorted(map(lambda x: x.upper(), groups[group].members))
    else:
        symbols = sorted(map(lambda x: x.upper(), groups[group]))
    
    if select_all:
        selected_symbols = group_container.multiselect(select_text, symbols, symbols)
    else:
        selected_symbols =  group_container.multiselect(select_text, symbols)

    # Modify any settings

    market_setting = col3.selectbox('Select Market Settings Group', market_specs.keys())
    select_dict = {}
    select_dict['data_source'] = alphapy_specs['sources'].keys()
    select_dict['subject'] = apg.SUBJECTS
    select_dict['algo'] = market_dict
    select_dict['bar_type'] = market_dict
    select_dict['fractals'] = market_dict
    select_dict['features'] = market_dict
    market_inputs = get_market_inputs(market_dict, select_dict)

    market_text = ' '.join(['View', market_setting, 'Settings'])
    with col3.expander(market_text):
        st.write(market_text)

    model_settings = col4.selectbox('Select Model Settings Group', model_specs.keys())
    model_text = ' '.join(['View', model_settings, 'Settings'])
    with col4.expander(model_text):
        st.write(model_text)

    # Run the selected action

    run_model_text = ' '.join(['Run', 'Model', project])
    run_system_text = ' '.join(['Run', 'System'])
    get_model_text = ' '.join(['Model', project, 'Results'])
    get_system_text = ' '.join(['System', 'Results'])
    select_action = col2.selectbox("Choose Action",
                        [None, run_model_text, run_system_text, get_model_text, get_system_text])

    status_ph = st.empty()
    status_ph.info("Status")

    in_progress = 'In Progress:'
    completed = 'Completed:'
    if select_action == run_model_text:
        status_text = ' '.join([in_progress, run_model_text])
        status_ph.info(status_text)
        with st.expander('View Log'):
            result = run_command(['mflow'], project_root)
        status_ph.info(' '.join([completed, select_action]))
    elif select_action == run_system_text:
        status_text = ' '.join([in_progress, run_system_text])
        status_ph.info(status_text)
        with st.expander('View Log'):
            result = run_command(['mflow'], project_root)
        status_ph.info(' '.join([completed, select_action]))
    elif select_action == get_model_text:
        st.info("Model Results Placeholder")
    elif select_action == get_system_text:
        st.info("System Results Placeholder")


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