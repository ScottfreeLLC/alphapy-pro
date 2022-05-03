################################################################################
#
# Package   : AlphaPy
# Module    : streamlit_mflow
# Created   : February 21, 2021
#
# streamlit run streamlit.py
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
import requests
import streamlit as st
import sys

from torch import compiled_with_cxx11_abi

from alphapy_main import get_alphapy_config
from streamlit_util import alphapy_projects, alphapy_request


#
# Initialize logger
#

logger = logging.getLogger(__name__)


def get_alphapy_groups(server_url):
    groups = alphapy_request(server_url, 'groups')
    return groups


def get_finviz_screener_groups():

    #filters = ['sh_price_u5','ta_gap_d5','ta_rsi_os30','ft=3']
    #stocks = Screener(filters=filters, order="price")

    filters = ['exch_nasd', 'idx_sp500']  # Shows companies in NASDAQ which are in the S&P500
    stock_list = Screener(filters=filters, table='Performance', order='price')  # Get the performance table and sort it by price ascending

    stock_df = pd.DataFrame(stock_list.data)
    st.write(stock_df)

    return stock_df


def get_finviz_portfolios():
    port_name = 'Test'
    portfolio = Portfolio('scottfree.analytics@scottfreellc.com', 'TS7$@@6dU9Nad@i', port_name)
    if portfolio:
        df = pd.DataFrame(portfolio.data)
        st.write(df)
    else:
        error_message = f"Could not find FinViz Portfolio: {port_name}"
        st.text(error_message)
    return df


def get_market_index_groups():
    url = f"https://docs.google.com/spreadsheets/d/1Syr2eLielHWsorxkDEZXyc55d6bNx1M3ZeI4vdn7Qzo/export?format=csv"
    df = pd.read_csv(url)
    df.loc[df['symbol'] == '^NDX', 'name'] = 'Nasdaq 100'

    #finnhub_client = finnhub.Client(api_key="c8m153aad3ie52go4qrg")
    #print(finnhub_client.indices_const(symbol = "^GSPC"))

    st.write(df)
    return df


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

    root_directory = alphapy_specs['mflow']['project_root']
    paths, projects = alphapy_projects(root_directory)
    project = st.sidebar.selectbox("Select Project", sorted(projects, key=str.casefold))

    text_ap = 'Market Flow'
    text_fs = 'Finviz Screener'
    text_fp = 'Finviz Portfolio'
    text_mi = 'Market Index'
    screener = st.sidebar.radio("Group", (text_ap, text_fs, text_fp, text_mi))

    col1, col2, col3, col4 = st.columns(4)

    server_url = alphapy_specs['mflow']['server_url']
    if screener == text_ap:
        groups = get_alphapy_groups(server_url)
    elif screener == text_fs:
        groups = get_finviz_screener_groups()
    elif screener == text_fp:
        groups = get_finviz_portfolios()
    elif screener == text_mi:
        groups = get_market_index_groups()

    group_text = ' '.join(['Select', screener, 'Group'])
    group = col1.selectbox(group_text, groups)

    systems = alphapy_request(server_url, 'systems')
    system = col2.selectbox("Select System", systems)

    with col1.expander("View Group Symbols"):
        df = pd.DataFrame(groups[group]['members'])
        df.columns = ['symbol']
        df['symbol'] = df['symbol'].str.upper()
        df.sort_values(by=['symbol'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.write(df)

    with col2.expander("View System Signals"):
        df = pd.DataFrame(systems[system].items(), columns=['signal', 'value'])
        df.reset_index(drop=True, inplace=True)
        st.write(df)

    market_settings = col3.selectbox('Select Market Settings', ['System', 'Portfolio', 'Features'])
    market_text = ' '.join(['View', market_settings, 'Settings'])
    with col3.expander(market_text):
        st.write(market_text)

    model_settings = col4.selectbox('Select Model Settings', ['Model', 'Data'])
    model_text = ' '.join(['View', model_settings, 'Settings'])
    with col4.expander(model_text):
        st.write(model_text)

    run_model_text = ' '.join(['Run', project, 'Model'])
    run_model_button = col1.button(run_model_text)

    run_system_text = ' '.join(['Run', system, 'System'])
    run_system_button = col1.button(run_system_text)

    today = datetime.now()
    year_ago = today - timedelta(days=365)
    col2.date_input('From', year_ago)
    col3.date_input('To')
