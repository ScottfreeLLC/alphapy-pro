################################################################################
#
# Package   : AlphaPy
# Module    : streamlit_mflow
# Created   : February 21, 2021
#
# Copyright 2021 ScottFree Analytics LLC
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

from datetime import datetime, timedelta
from finviz.portfolio import Portfolio
from finviz.screener import Screener
from itsdangerous import json
import finnhub
import logging
import os
import pandas as pd
import streamlit as st
import sys
from alphapy.streamlit_requests import run_command

from alphapy_main import get_alphapy_config
from streamlit_requests import alphapy_request
from streamlit_requests import run_command


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Function get_finviz_screener_groups
#

def get_finviz_screener_groups():

    #filters = ['sh_price_u5','ta_gap_d5','ta_rsi_os30','ft=3']
    #stocks = Screener(filters=filters, order="price")

    filters = ['exch_nasd', 'idx_sp500']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Performance', order='price')  # Get the performance table and sort it by price ascending

    stock_df = pd.DataFrame(stock_list.data)
    st.write(stock_df)

    return stock_df


#
# Function get_finviz_portfolios
#

def get_finviz_portfolios(alphapy_specs):
    finviz_specs = alphapy_specs['finviz']
    email = finviz_specs['email']
    api_key = finviz_specs['api_key']
    portfolios = finviz_specs['portfolios']

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

@st.cache
def get_market_index_groups(alphapy_specs):
    url = f"https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    df = pd.read_csv(url)
    df.loc[df['symbol'] == '^NDX', 'name'] = 'Nasdaq 100'
    finnhub_client = finnhub.Client(api_key=alphapy_specs['finnhub']['api_key'])

    groups = {}
    for _, row in df.iterrows():
        group_symbol = row['symbol']
        group_name = row['name']
        group_dict = finnhub_client.indices_const(symbol=group_symbol)
        groups[group_name] = group_dict['constituents']
    return groups


#
# Function run_project
#

def run_project(alphapy_specs, project):

    # Vet the model and market specifications

    project_root = '/'.join([alphapy_specs['mflow']['project_root'], project])
    model_specs, _ = alphapy_request(alphapy_specs, 'model_config', project_root)
    market_specs, _ = alphapy_request(alphapy_specs, 'market_config', alphapy_specs, project_root)

    # Determine the source of market groups

    text_ap = 'Market Flow'
    text_fs = 'Finviz Screener'
    text_fp = 'Finviz Portfolio'
    text_mi = 'Market Index'
    screener = st.sidebar.radio("Group Source", (text_ap, text_fs, text_fp, text_mi))

    col1, col2, col3, col4 = st.columns(4)

    if screener == text_ap:
        groups = alphapy_request(alphapy_specs, 'groups')
    elif screener == text_fs:
        groups = get_finviz_screener_groups()
    elif screener == text_fp:
        groups = get_finviz_portfolios(alphapy_specs)
    elif screener == text_mi:
        groups = get_market_index_groups(alphapy_specs)

    group_text = ' '.join(['Select', screener, 'Group'])
    group = col1.selectbox(group_text, groups.keys())

    # Select the system to run

    systems = alphapy_request(alphapy_specs, 'systems', alphapy_specs)
    system = col2.selectbox("Select System", systems)

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

    market_settings = col3.selectbox('Select Market Settings', market_specs.keys())
    market_text = ' '.join(['View', market_settings, 'Settings'])
    with col3.expander(market_text):
        st.write(market_text)

    with col3.expander("View System Signals"):
        df = pd.DataFrame(systems[system].items(), columns=['signal', 'value'])
        df.reset_index(drop=True, inplace=True)
        st.write(df)

    model_settings = col4.selectbox('Select Model Settings', model_specs.keys())
    model_text = ' '.join(['View', model_settings, 'Settings'])
    with col4.expander(model_text):
        st.write(model_text)

    # Run the selected action

    run_model_text = ' '.join(['Run', 'Model', project])
    run_system_text = ' '.join(['Run', 'System', system])
    get_model_text = ' '.join(['Model', project, 'Results'])
    get_system_text = ' '.join(['System', system, 'Results'])
    select_action = col1.selectbox("Choose Action",
                        [None, run_model_text, run_system_text, get_model_text, get_system_text])

    today = datetime.now()
    year_ago = today - timedelta(days=365)
    col2.date_input('From', year_ago)
    col2.date_input('To')

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
# Main Application
#

def app():
 
    # Get the AlphaPy environment variable

    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)
    else:
        # Read the AlphaPy configuration file
        alphapy_specs = get_alphapy_config(alphapy_root)

    projects = alphapy_request(alphapy_specs, 'projects', alphapy_specs)
    projects = sorted(projects, key=str.casefold)
    projects.insert(0, None)
    project = st.sidebar.selectbox("Select Project", projects)

    if project:
        run_project(alphapy_specs, project)
